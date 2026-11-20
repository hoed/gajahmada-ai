import importlib
import json
import os
from typing import Any, List, Optional


def _try_import(module_name: str) -> Optional[Any]:
    try:
        return importlib.import_module(module_name)
    except ImportError:
        return None


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

        if self._has_faiss and self._faiss is not None:
            if os.path.exists(os.path.join(self.storage_path, "index.faiss")):
                self.load()
            else:
                self.index = self._faiss.IndexFlatL2(dimension)
                self.load()
        else:
            self.load()

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
                    self.doc_store = data.get("docs", [])
                    self.vectors = data.get("vectors", [])
            except Exception:
                self.doc_store = []
                self.vectors = []

        if self._has_faiss and self._faiss is not None:
            try:
                index_path = os.path.join(self.storage_path, "index.faiss")
                if os.path.exists(index_path):
                    self.index = self._faiss.read_index(index_path)
                else:
                    self.index = self._faiss.IndexFlatL2(self.dimension)
                    if self.vectors and self._np is not None:
                        arr = self._np.array(self.vectors, dtype=self._np.float32)
                        self.index.add(arr)
            except Exception:
                pass

    def ingest(self, text_chunk: str, embedding: List[float]):
        if len(embedding) != self.dimension:
            raise ValueError(f"Embedding must be {self.dimension} dimensions, got {len(embedding)}.")

        self.doc_store.append(text_chunk)
        self.vectors.append(embedding)

        if self._has_faiss and self.index is not None and self._np is not None:
            try:
                vector = self._np.array([embedding], dtype=self._np.float32)
                self.index.add(vector)
            except Exception:
                pass

        self._dirty = True

    def flush(self):
        if self._dirty:
            self.save()

    def search(self, query_embedding: List[float], k: int = 3) -> List[str]:
        if not self.doc_store:
            return []

        if len(query_embedding) != self.dimension:
            raise ValueError(f"Query embedding must be {self.dimension} dimensions, got {len(query_embedding)}.")

        if self._has_faiss and self.index is not None and self.index.ntotal > 0 and self._np is not None:
            try:
                vector = self._np.array([query_embedding], dtype=self._np.float32)
                _, indices = self.index.search(vector, k)
                results = []
                for idx in indices[0]:
                    if 0 <= idx < len(self.doc_store):
                        results.append(self.doc_store[idx])
                return results
            except Exception:
                pass

        scored = []
        for idx, vec in enumerate(self.vectors):
            if len(vec) == len(query_embedding):
                dist = sum((a - b) ** 2 for a, b in zip(query_embedding, vec))
                scored.append((dist, idx))
        scored.sort(key=lambda item: item[0])
        return [self.doc_store[idx] for _, idx in scored[:k] if idx < len(self.doc_store)]
