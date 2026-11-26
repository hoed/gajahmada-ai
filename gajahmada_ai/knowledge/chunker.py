"""
GajahMada AI — Structure- & Token-Aware Document Chunker.

Splits external documents into overlapping semantic chunks while preserving:
- Document & source identity (`source_id`, `source_type`)
- Structural headings (`# Heading` / Markdown sections)
- Page numbers when available (e.g. `\f` form-feed or `[PAGE N]` markers)
- Configurable chunk size and overlap
- Optional token-aware sizing via `GNNMBPETokenizer` (with word/char fallback)
"""
from dataclasses import dataclass, field
import re
from typing import Any, Dict, List, Optional


@dataclass
class DocumentChunk:
    text: str
    chunk_index: int
    source_id: str
    heading: str = ""
    page_number: Optional[int] = None
    token_count: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)


class DocumentChunker:
    """Configurable structure-aware & token-aware chunker with clean fallback."""

    def __init__(
        self,
        chunk_size: int = 180,
        chunk_overlap: int = 30,
        tokenizer: Optional[Any] = None,
    ):
        if chunk_size < 16:
            raise ValueError("chunk_size must be >= 16")
        if chunk_overlap < 0 or chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be >= 0 and < chunk_size")
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.tokenizer = tokenizer

    def count_tokens(self, text: str) -> int:
        if self.tokenizer is not None and hasattr(self.tokenizer, "encode"):
            try:
                return len(self.tokenizer.encode(text, add_bos=False, add_eos=False))
            except Exception:
                pass
        return len(re.findall(r"\S+", text))

    def _split_long_segment(self, text: str) -> List[str]:
        words = text.split()
        if not words:
            return []
        if self.count_tokens(text) <= self.chunk_size:
            return [text.strip()]

        step = max(1, self.chunk_size - self.chunk_overlap)
        out: List[str] = []
        idx = 0
        while idx < len(words):
            window = words[idx : idx + self.chunk_size]
            candidate = " ".join(window).strip()
            # If tokenizer produces more tokens than word count, tighten window gracefully
            while len(window) > 8 and self.count_tokens(candidate) > self.chunk_size:
                window = window[: max(8, int(len(window) * 0.8))]
                candidate = " ".join(window).strip()
            if candidate:
                out.append(candidate)
            if idx + len(window) >= len(words):
                break
            idx += max(1, len(window) - self.chunk_overlap)
        return out

    def chunk_document(
        self,
        text: str,
        source_id: str,
        base_metadata: Optional[Dict[str, Any]] = None,
    ) -> List[DocumentChunk]:
        if not text or not text.strip():
            return []

        meta = dict(base_metadata or {})
        # Split pages if form-feed (\f) is present (standard PDF extraction separator)
        raw_pages = text.split("\f") if "\f" in text else [text]
        has_pages = len(raw_pages) > 1 or bool(meta.get("has_pages"))

        chunks: List[DocumentChunk] = []
        chunk_idx = 0
        current_heading = str(meta.get("title", "")).strip()

        for p_idx, page_text in enumerate(raw_pages, start=1):
            page_num = p_idx if has_pages else meta.get("page_number")
            # Split into paragraphs preserving markdown headings
            paragraphs = [p.strip() for p in re.split(r"\n\s*\n", page_text) if p.strip()]

            buffer: List[str] = []
            buffer_tokens = 0

            def _flush():
                nonlocal chunk_idx, buffer, buffer_tokens
                if not buffer:
                    return
                merged = "\n\n".join(buffer).strip()
                for part in self._split_long_segment(merged):
                    chunks.append(
                        DocumentChunk(
                            text=part,
                            chunk_index=chunk_idx,
                            source_id=source_id,
                            heading=current_heading,
                            page_number=page_num,
                            token_count=self.count_tokens(part),
                            metadata={**meta, "heading": current_heading, "page_number": page_num},
                        )
                    )
                    chunk_idx += 1
                buffer = []
                buffer_tokens = 0

            for para in paragraphs:
                heading_match = re.match(r"^(#{1,6})\s+(.+)$", para.splitlines()[0].strip())
                if heading_match:
                    _flush()
                    current_heading = heading_match.group(2).strip()
                    rest = "\n".join(para.splitlines()[1:]).strip()
                    if not rest:
                        continue
                    para = rest

                para_tokens = self.count_tokens(para)
                if buffer and (buffer_tokens + para_tokens) > self.chunk_size:
                    _flush()
                buffer.append(para)
                buffer_tokens += para_tokens

            _flush()

        return chunks
