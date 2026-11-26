"""
GajahMada AI — External Knowledge Ingestion Pipeline.

10-Step Pipeline:
1. Parse document (TXT, Markdown, HTML, JSON/CSV structured, PDF if pypdf/fitz present,
   DOCX via pure-Python zipfile XML parser, or authorized MemoryRecord).
2. Normalize text.
3. Validate source metadata & non-empty content.
4. Deduplicate via SHA-256 document hash against `SourceRegistry`.
5. Extract metadata (title, headings, page count, hash).
6. Chunk document via `DocumentChunker`.
7. Generate embeddings via active `EmbeddingProvider` with explicit encoder version.
8. Index chunks into `HybridKnowledgeRetriever`.
9. Store source provenance in `SourceRegistry`.
10. Record structured ingestion status & log.
"""
import csv
import hashlib
from html.parser import HTMLParser
import importlib
import io
import json
import os
import re
import time
import xml.etree.ElementTree as ET
import zipfile
from typing import Any, Dict, List, Optional

from gajahmada_ai.knowledge.chunker import DocumentChunker
from gajahmada_ai.knowledge.interfaces import (
    KnowledgeRecord,
    KnowledgeSource,
)
from gajahmada_ai.knowledge.retriever import HybridKnowledgeRetriever
from gajahmada_ai.knowledge.source_registry import SourceRegistry
from gajahmada_ai.memory.models import VerificationStatus, VisibilityPolicy


class _HTMLTextExtractor(HTMLParser):
    def __init__(self):
        super().__init__()
        self._parts: List[str] = []
        self._skip = False

    def handle_starttag(self, tag, attrs):
        if tag.lower() in ("script", "style", "noscript"):
            self._skip = True
        elif tag.lower() in ("p", "div", "br", "li", "h1", "h2", "h3", "h4"):
            self._parts.append("\n")

    def handle_endtag(self, tag):
        if tag.lower() in ("script", "style", "noscript"):
            self._skip = False

    def handle_data(self, data):
        if not self._skip and data:
            self._parts.append(data)

    def get_text(self) -> str:
        return "".join(self._parts)


class KnowledgeIngestionPipeline:
    """Offline-first multi-format external document ingestion pipeline."""

    def __init__(
        self,
        retriever: HybridKnowledgeRetriever,
        source_registry: SourceRegistry,
        chunker: Optional[DocumentChunker] = None,
    ):
        self.retriever = retriever
        self.source_registry = source_registry
        self.chunker = chunker or DocumentChunker()
        self.ingestion_logs: List[Dict[str, Any]] = []

    def _normalize_text(self, text: str) -> str:
        text = text.replace("\r\n", "\n").replace("\r", "\n")
        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text.strip()

    def _parse_html(self, raw: str) -> str:
        parser = _HTMLTextExtractor()
        parser.feed(raw)
        return parser.get_text()

    def _parse_docx_bytes(self, raw_bytes: bytes) -> str:
        with zipfile.ZipFile(io.BytesIO(raw_bytes)) as zf:
            xml_content = zf.read("word/document.xml")
        root = ET.fromstring(xml_content)
        paragraphs: List[str] = []
        for p in root.iter():
            if p.tag.endswith("}p"):
                texts = [t.text for t in p.iter() if t.tag.endswith("}t") and t.text]
                if texts:
                    paragraphs.append("".join(texts))
        return "\n\n".join(paragraphs)

    def _parse_pdf_path(self, path: str) -> str:
        for mod_name in ("pypdf", "PyPDF2"):
            try:
                mod = importlib.import_module(mod_name)
                reader = mod.PdfReader(path)
                pages = [(page.extract_text() or "") for page in reader.pages]
                return "\f".join(pages)
            except ImportError:
                continue
        raise RuntimeError("PDF parsing requires optional `pypdf` or `PyPDF2` package.")

    def _parse_structured(self, raw: str, fmt: str) -> str:
        if fmt == "json":
            obj = json.loads(raw)
            if isinstance(obj, list):
                return "\n\n".join(json.dumps(item, ensure_ascii=False) for item in obj)
            return json.dumps(obj, indent=2, ensure_ascii=False)
        if fmt == "csv":
            reader = csv.DictReader(io.StringIO(raw))
            lines = [", ".join(f"{k}: {v}" for k, v in row.items()) for row in reader]
            return "\n".join(lines)
        return raw

    def ingest_text(
        self,
        content: str,
        source_type: str = "txt",
        title: str = "",
        location: str = "inline",
        author: str = "operator",
        organization: str = "",
        license_name: str = "unspecified",
        usage_restrictions: str = "",
        visibility_policy: str = VisibilityPolicy.LOCAL.value,
        verification_status: str = VerificationStatus.UNVERIFIED.value,
        owner_node_id: str = "local",
        training_approved: bool = False,
        extra_metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        t0 = time.perf_counter()
        fmt = source_type.lower().strip()
        try:
            # 1. Parse
            if fmt == "html":
                parsed = self._parse_html(content)
            elif fmt in ("json", "csv"):
                parsed = self._parse_structured(content, fmt)
            else:
                parsed = content

            # 2. Normalize
            normalized = self._normalize_text(parsed)

            # 3. Validate
            if not normalized or len(normalized) < 4:
                raise ValueError("Document content is empty or too short after normalization.")

            # 4. Deduplicate via SHA-256 hash
            doc_hash = hashlib.sha256(normalized.encode("utf-8")).hexdigest()
            existing = self.source_registry.find_by_hash(doc_hash)
            if existing is not None:
                entry = {
                    "status": "DUPLICATE_SKIPPED",
                    "source_id": existing.source_id,
                    "document_hash": doc_hash,
                    "chunks_indexed": 0,
                    "elapsed_ms": round((time.perf_counter() - t0) * 1000, 2),
                }
                self.ingestion_logs.append(entry)
                return entry

            # 5. Build Source metadata & register
            source = KnowledgeSource(
                source_type=fmt,
                location=location,
                title=title or os.path.basename(location) or "untitled",
                author=author,
                organization=organization,
                license=license_name,
                usage_restrictions=usage_restrictions,
                document_hash=doc_hash,
                verification_status=verification_status,
                visibility_policy=visibility_policy,
                owner_node_id=owner_node_id,
                training_approved=training_approved,
                provenance={"ingested_format": fmt, **(extra_metadata or {})},
            )
            self.source_registry.register(source)

            # 6. Chunk document
            chunks = self.chunker.chunk_document(
                normalized,
                source_id=source.source_id,
                base_metadata={"title": source.title, "source_type": fmt, **(extra_metadata or {})},
            )

            # 7 & 8. Embed & Index chunks
            indexed_ids: List[str] = []
            provider = self.retriever.embedding_provider
            for ch in chunks:
                rec = KnowledgeRecord(
                    content=ch.text,
                    source_id=source.source_id,
                    source_type=fmt,
                    knowledge_layer="external",
                    visibility_policy=visibility_policy,
                    verification_status=verification_status,
                    confidence=0.85 if verification_status in ("VERIFIED", "INDEPENDENTLY_VERIFIED") else 0.65,
                    metadata=ch.metadata,
                    provenance={
                        "document_hash": doc_hash,
                        "location": location,
                        "author": author,
                        "organization": organization,
                        "license": license_name,
                    },
                    embedding=provider.embed(ch.text),
                    embedding_encoder_id=provider.encoder_id,
                    embedding_version=provider.encoder_version,
                    chunk_index=ch.chunk_index,
                    page_number=ch.page_number,
                    heading=ch.heading,
                    owner_node_id=owner_node_id,
                )
                self.retriever.index_record(rec)
                indexed_ids.append(rec.record_id)

            # 9 & 10. Record status
            entry = {
                "status": "INGESTED",
                "source_id": source.source_id,
                "document_hash": doc_hash,
                "chunks_indexed": len(indexed_ids),
                "record_ids": indexed_ids,
                "encoder": provider.identity_metadata(),
                "elapsed_ms": round((time.perf_counter() - t0) * 1000, 2),
            }
            self.ingestion_logs.append(entry)
            return entry

        except Exception as exc:
            err_entry = {
                "status": "FAILED",
                "location": location,
                "source_type": fmt,
                "error": str(exc),
                "elapsed_ms": round((time.perf_counter() - t0) * 1000, 2),
            }
            self.ingestion_logs.append(err_entry)
            return err_entry

    def ingest_file(self, file_path: str, **kwargs) -> Dict[str, Any]:
        ext = os.path.splitext(file_path)[1].lower().lstrip(".")
        if not os.path.exists(file_path):
            return {"status": "FAILED", "location": file_path, "error": f"File not found: {file_path}"}
        try:
            if ext == "pdf":
                text = self._parse_pdf_path(file_path)
                return self.ingest_text(text, source_type="pdf", location=file_path, **kwargs)
            if ext == "docx":
                with open(file_path, "rb") as f:
                    text = self._parse_docx_bytes(f.read())
                return self.ingest_text(text, source_type="docx", location=file_path, **kwargs)
            with open(file_path, "r", encoding="utf-8", errors="replace") as f:
                text = f.read()
            fmt = "markdown" if ext in ("md", "markdown") else (ext or "txt")
            return self.ingest_text(text, source_type=fmt, location=file_path, **kwargs)
        except Exception as exc:
            entry = {"status": "FAILED", "location": file_path, "error": str(exc)}
            self.ingestion_logs.append(entry)
            return entry
