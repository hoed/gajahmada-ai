"""
GajahMada Hybrid Knowledge Retrieval Provider Abstraction (Phase 8).

Preserves FAISS CPU as a first-class local/server vector retrieval engine while
providing graceful, honest runtime adaptation for Vercel Free:
- KnowledgeProviderInterface (shared contract)
- FaissLocalProvider (local dev / testing / compatible server with faiss-cpu)
- FaissArtifactProvider (loads prebuilt FAISS artifact ONLY if runtime & size allow)
- LightweightRetrievalProvider (pure-Python fallback; explicitly identifies its method
  and NEVER claims FAISS vector similarity)

Explicit provider statuses:
  ACTIVE | AVAILABLE_LOCALLY | NOT_CONFIGURED | INCOMPATIBLE_RUNTIME | DISABLED
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from enum import Enum
import importlib
import json
import os
from typing import Any, Dict, List, Optional


class ProviderStatus(str, Enum):
    ACTIVE = "ACTIVE"
    AVAILABLE_LOCALLY = "AVAILABLE_LOCALLY"
    NOT_CONFIGURED = "NOT_CONFIGURED"
    INCOMPATIBLE_RUNTIME = "INCOMPATIBLE_RUNTIME"
    DISABLED = "DISABLED"


class KnowledgeProviderInterface(ABC):
    """Shared retrieval provider interface for GajahMada Hybrid Knowledge."""

    name: str
    retrieval_method: str

    @abstractmethod
    def status(self) -> Dict[str, Any]:
        """Return honest runtime status, compatibility, and metadata."""

    @abstractmethod
    def search(self, query_embedding: List[float], query_text: str = "", k: int = 5) -> List[Dict[str, Any]]:
        """Execute search and return ranked results with provenance & provider attribution."""


def _faiss_available() -> bool:
    try:
        importlib.import_module("faiss")
        importlib.import_module("numpy")
        return True
    except Exception:
        return False


class FaissLocalProvider(KnowledgeProviderInterface):
    """
    Preserves existing FAISS CPU (`LocalKnowledge`) for local development,
    local testing, and compatible server environments.
    """

    name = "FaissLocalProvider"
    retrieval_method = "faiss_cpu_index_flat_l2"

    def __init__(self, rag_store: Optional[Any] = None, is_vercel: Optional[bool] = None):
        self.rag_store = rag_store
        self.is_vercel = (os.environ.get("VERCEL") == "1") if is_vercel is None else is_vercel
        self._has_faiss = _faiss_available()

    def status(self) -> Dict[str, Any]:
        if self.is_vercel:
            st = ProviderStatus.INCOMPATIBLE_RUNTIME.value
            reason = "FAISS CPU native binaries exceed Vercel Free 50MB serverless bundle; preserved for local/server runtimes."
        elif not self._has_faiss:
            st = ProviderStatus.NOT_CONFIGURED.value
            reason = "faiss-cpu not installed in current Python environment (install via requirements-local.txt)."
        elif self.rag_store is not None and getattr(self.rag_store, "_has_faiss", False):
            st = ProviderStatus.ACTIVE.value
            reason = "FAISS CPU IndexFlatL2 active in local/server runtime."
        else:
            st = ProviderStatus.AVAILABLE_LOCALLY.value
            reason = "faiss-cpu library is available locally; awaiting LocalKnowledge index mount."

        return {
            "provider": self.name,
            "status": st,
            "retrieval_method": self.retrieval_method,
            "faiss_installed": self._has_faiss,
            "is_vercel": self.is_vercel,
            "indexed_documents": len(self.rag_store.doc_store) if self.rag_store is not None else 0,
            "dimension": getattr(self.rag_store, "dimension", 384),
            "reason": reason,
        }

    def search(self, query_embedding: List[float], query_text: str = "", k: int = 5) -> List[Dict[str, Any]]:
        st = self.status()["status"]
        if st != ProviderStatus.ACTIVE.value or self.rag_store is None:
            return []
        hits = self.rag_store.search(query_embedding, k=k, query_text=query_text)
        return [
            {
                "content": doc,
                "rank": idx + 1,
                "provider": self.name,
                "retrieval_method": self.retrieval_method,
            }
            for idx, doc in enumerate(hits)
        ]


class FaissArtifactProvider(KnowledgeProviderInterface):
    """
    Loads prebuilt FAISS index artifacts ONLY when runtime, artifact file,
    size limit, and metadata compatibility (dimension + encoder_id) are validated.
    """

    name = "FaissArtifactProvider"
    retrieval_method = "faiss_prebuilt_artifact_l2"

    def __init__(
        self,
        artifact_dir: Optional[str] = None,
        max_artifact_mb: float = 25.0,
        expected_dimension: int = 384,
        expected_encoder_id: Optional[str] = None,
        is_vercel: Optional[bool] = None,
    ):
        self.artifact_dir = artifact_dir
        self.max_artifact_mb = max_artifact_mb
        self.expected_dimension = expected_dimension
        self.expected_encoder_id = expected_encoder_id
        self.is_vercel = (os.environ.get("VERCEL") == "1") if is_vercel is None else is_vercel
        self._has_faiss = _faiss_available()
        self._index = None
        self._docs: List[str] = []
        self._meta: Dict[str, Any] = {}
        self._load_error: Optional[str] = None
        self._try_load()

    def _try_load(self) -> None:
        if self.is_vercel or not self._has_faiss or not self.artifact_dir:
            return
        idx_path = os.path.join(self.artifact_dir, "index.faiss")
        meta_path = os.path.join(self.artifact_dir, "artifact_meta.json")
        if not (os.path.exists(idx_path) and os.path.exists(meta_path)):
            return
        size_mb = os.path.getsize(idx_path) / (1024 * 1024)
        if size_mb > self.max_artifact_mb:
            self._load_error = f"Artifact size ({size_mb:.2f} MB) exceeds max_artifact_mb ({self.max_artifact_mb} MB)."
            return
        try:
            with open(meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
            if int(meta.get("dimension", -1)) != self.expected_dimension:
                self._load_error = f"Dimension mismatch: artifact={meta.get('dimension')} expected={self.expected_dimension}."
                return
            if self.expected_encoder_id and meta.get("encoder_id") != self.expected_encoder_id:
                self._load_error = f"Encoder mismatch: artifact={meta.get('encoder_id')} expected={self.expected_encoder_id}."
                return
            faiss = importlib.import_module("faiss")
            self._index = faiss.read_index(idx_path)
            self._docs = list(meta.get("docs", []))
            self._meta = meta
        except Exception as exc:
            self._load_error = f"Failed to load FAISS artifact: {exc}"

    def status(self) -> Dict[str, Any]:
        if self.is_vercel:
            st = ProviderStatus.INCOMPATIBLE_RUNTIME.value
            reason = "FAISS artifact loading disabled on Vercel Free (native library incompatible with 50MB lambda)."
        elif not self._has_faiss:
            st = ProviderStatus.NOT_CONFIGURED.value
            reason = "faiss-cpu not installed."
        elif self._load_error:
            st = ProviderStatus.DISABLED.value
            reason = self._load_error
        elif self._index is not None:
            st = ProviderStatus.ACTIVE.value
            reason = f"Loaded validated FAISS artifact from {self.artifact_dir}."
        else:
            st = ProviderStatus.NOT_CONFIGURED.value
            reason = "No prebuilt FAISS artifact configured."
        return {
            "provider": self.name,
            "status": st,
            "retrieval_method": self.retrieval_method,
            "artifact_dir": self.artifact_dir,
            "metadata": self._meta,
            "reason": reason,
        }

    def search(self, query_embedding: List[float], query_text: str = "", k: int = 5) -> List[Dict[str, Any]]:
        if self._index is None or not self._docs:
            return []
        np = importlib.import_module("numpy")
        vec = np.array([query_embedding], dtype=np.float32)
        dists, idxs = self._index.search(vec, min(k, len(self._docs)))
        out = []
        for rank, (dist, doc_idx) in enumerate(zip(dists[0], idxs[0]), start=1):
            if 0 <= int(doc_idx) < len(self._docs):
                out.append({
                    "content": self._docs[int(doc_idx)],
                    "distance": float(dist),
                    "rank": rank,
                    "provider": self.name,
                    "retrieval_method": self.retrieval_method,
                })
        return out


class LightweightRetrievalProvider(KnowledgeProviderInterface):
    """
    Pure-Python lexical + L2/BM25 retrieval fallback for environments where FAISS CPU
    cannot run (e.g., Vercel Free serverless functions).
    Explicitly declares its method and NEVER claims FAISS vector similarity.
    """

    name = "LightweightRetrievalProvider"
    retrieval_method = "pure_python_lexical_l2_fallback_non_faiss"

    def __init__(self, rag_store: Optional[Any] = None, hybrid_retriever: Optional[Any] = None):
        self.rag_store = rag_store
        self.hybrid_retriever = hybrid_retriever

    def status(self) -> Dict[str, Any]:
        active = self.rag_store is not None or self.hybrid_retriever is not None
        return {
            "provider": self.name,
            "status": ProviderStatus.ACTIVE.value if active else ProviderStatus.NOT_CONFIGURED.value,
            "retrieval_method": self.retrieval_method,
            "uses_faiss": False,
            "indexed_documents": len(self.rag_store.doc_store) if self.rag_store is not None else 0,
            "reason": (
                "Pure-Python lexical + L2/BM25 retrieval active (honest non-FAISS fallback compatible with Vercel Free)."
                if active
                else "No local store attached."
            ),
        }

    def search(self, query_embedding: List[float], query_text: str = "", k: int = 5) -> List[Dict[str, Any]]:
        if self.rag_store is None:
            return []
        hits = self.rag_store.search(query_embedding, k=k, query_text=query_text)
        return [
            {
                "content": doc,
                "rank": idx + 1,
                "provider": self.name,
                "retrieval_method": self.retrieval_method,
                "uses_faiss": False,
            }
            for idx, doc in enumerate(hits)
        ]


class RetrievalProviderRegistry:
    """Coordinates FAISS CPU providers and Lightweight fallback honestly."""

    def __init__(
        self,
        rag_store: Optional[Any] = None,
        hybrid_retriever: Optional[Any] = None,
        artifact_dir: Optional[str] = None,
        is_vercel: Optional[bool] = None,
    ):
        self.faiss_local = FaissLocalProvider(rag_store=rag_store, is_vercel=is_vercel)
        self.faiss_artifact = FaissArtifactProvider(artifact_dir=artifact_dir, is_vercel=is_vercel)
        self.lightweight = LightweightRetrievalProvider(rag_store=rag_store, hybrid_retriever=hybrid_retriever)

    def select_active_provider(self) -> KnowledgeProviderInterface:
        if self.faiss_local.status()["status"] == ProviderStatus.ACTIVE.value:
            return self.faiss_local
        if self.faiss_artifact.status()["status"] == ProviderStatus.ACTIVE.value:
            return self.faiss_artifact
        return self.lightweight

    def report(self) -> Dict[str, Any]:
        selected = self.select_active_provider()
        return {
            "selected_provider": selected.name,
            "selected_method": selected.retrieval_method,
            "providers": [
                self.faiss_local.status(),
                self.faiss_artifact.status(),
                self.lightweight.status(),
            ],
        }
