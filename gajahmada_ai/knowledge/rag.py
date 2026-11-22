import importlib
import json
import os
import re
from typing import Any, List, Optional

_STOPWORDS = {
    "yang", "dan", "di", "ke", "dari", "ini", "itu", "dengan", "untuk", "pada",
    "adalah", "sebagai", "dalam", "atau", "juga", "oleh", "karena", "tentang",
    "apa", "siapa", "kapan", "dimana", "bagaimana", "mengapa", "kenapa", "berapa",
    "jelaskan", "tolong", "buatkan", "sebutkan", "ceritakan", "kamu", "anda", "saya",
}


def _try_import(module_name: str) -> Optional[Any]:
    try:
        return importlib.import_module(module_name)
    except ImportError:
        return None


def _extract_query_terms(text: str) -> List[str]:
    tokens = re.findall(r"[a-z0-9]+", text.lower())
    return [t for t in tokens if len(t) > 2 and t not in _STOPWORDS]


class LocalKnowledge:
    def __init__(self, dimension: int = 384, storage_path: str = "./knowledge_store"):
        self.dimension = dimension
        self.storage_path = storage_path
        self.index = None
        self.vectors: List[List[float]] = []
        self.doc_store: List[str] = []
        self._dirty = False
        self._faiss = _try_import("faiss")
        self._np = _try_import("numpy")
        self._has_faiss = self._faiss is not None and self._np is not None

        if os.environ.get("VERCEL") == "1" and not self.storage_path.startswith("/tmp"):
            self.storage_path = "/tmp/knowledge_store"

        try:
            os.makedirs(self.storage_path, exist_ok=True)
        except OSError:
            self.storage_path = "/tmp/knowledge_store"
            try:
                os.makedirs(self.storage_path, exist_ok=True)
            except OSError:
                pass

        self.load()

    def _rebuild_faiss_index(self):
        if self._has_faiss and self._faiss is not None:
            try:
                self.index = self._faiss.IndexFlatL2(self.dimension)
                if self.vectors and self._np is not None:
                    arr = self._np.array(self.vectors, dtype=self._np.float32)
                    self.index.add(arr)
            except Exception:
                pass

    def save(self):
        try:
            os.makedirs(self.storage_path, exist_ok=True)
            docs_json_path = os.path.join(self.storage_path, "docs.json")
            with open(docs_json_path, "w", encoding="utf-8") as f:
                json.dump({"docs": self.doc_store, "vectors": self.vectors}, f, ensure_ascii=False)

            if self._has_faiss and self._faiss is not None and self.index is not None:
                self._faiss.write_index(self.index, os.path.join(self.storage_path, "index.faiss"))
            self._dirty = False
        except OSError:
            pass

    def load(self):
        docs_json_path = os.path.join(self.storage_path, "docs.json")
        if os.path.exists(docs_json_path):
            try:
                with open(docs_json_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    raw_docs = data.get("docs", [])
                    raw_vecs = data.get("vectors", [])
                    # Deduplicate preserving order (including cleaning legacy timestamped duplicates)
                    seen_keys = set()
                    clean_docs = []
                    clean_vecs = []
                    for i, doc in enumerate(raw_docs):
                        norm_key = re.sub(r"\(waktu sistem:\s*[^)]+\)", "", doc).strip()
                        if norm_key not in seen_keys:
                            seen_keys.add(norm_key)
                            clean_docs.append(doc)
                            if i < len(raw_vecs):
                                clean_vecs.append(raw_vecs[i])
                    self.doc_store = clean_docs
                    self.vectors = clean_vecs
                    if len(clean_docs) != len(raw_docs):
                        self._dirty = True
            except Exception:
                self.doc_store = []
                self.vectors = []

        self._rebuild_faiss_index()
        if self._dirty:
            self.save()

    def ingest(self, text_chunk: str, embedding: List[float]):
        if len(embedding) != self.dimension:
            raise ValueError(f"Embedding must be {self.dimension} dimensions, got {len(embedding)}.")

        if text_chunk in self.doc_store:
            return

        self.doc_store.append(text_chunk)
        self.vectors.append(embedding)

        if self._has_faiss and self.index is not None and self._np is not None:
            try:
                vector = self._np.array([embedding], dtype=self._np.float32)
                self.index.add(vector)
            except Exception:
                pass

        self._dirty = True

    def remove_at(self, doc_idx: int) -> str:
        if doc_idx < 0 or doc_idx >= len(self.doc_store):
            raise IndexError("Document index out of range")
        removed = self.doc_store.pop(doc_idx)
        if doc_idx < len(self.vectors):
            self.vectors.pop(doc_idx)
        self._rebuild_faiss_index()
        self._dirty = True
        self.flush()
        return removed

    def flush(self):
        if self._dirty:
            self.save()

    def search(self, query_embedding: List[float], k: int = 3, query_text: str = "") -> List[str]:
        if not self.doc_store:
            return []

        if len(query_embedding) != self.dimension:
            raise ValueError(f"Query embedding must be {self.dimension} dimensions, got {len(query_embedding)}.")

        q_terms = _extract_query_terms(query_text) if query_text else []

        scored = []
        for idx, doc in enumerate(self.doc_store):
            doc_lower = doc.lower()
            lexical_hits = sum(1 for t in q_terms if t in doc_lower)

            if query_text and q_terms and lexical_hits == 0:
                continue
            if query_text and len(q_terms) >= 3 and lexical_hits < 2 and (lexical_hits / len(q_terms)) < 0.34:
                continue

            vec_dist = 2.0
            if idx < len(self.vectors) and len(self.vectors[idx]) == len(query_embedding):
                vec_dist = sum((a - b) ** 2 for a, b in zip(query_embedding, self.vectors[idx]))

            combined_score = vec_dist - (lexical_hits * 1.25)
            scored.append((combined_score, idx))

        if not scored and not query_text:
            for idx, vec in enumerate(self.vectors):
                if len(vec) == len(query_embedding):
                    dist = sum((a - b) ** 2 for a, b in zip(query_embedding, vec))
                    scored.append((dist, idx))

        scored.sort(key=lambda item: item[0])
        return [self.doc_store[idx] for _, idx in scored[:k] if idx < len(self.doc_store)]
