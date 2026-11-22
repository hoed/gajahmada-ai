import re
from typing import Any, Callable, Dict, List, Optional

from gajahmada_ai.storage.interfaces import StorageBackend
from gajahmada_ai.storage.vector_store import SemanticVectorIndex


class HybridMemoryRetriever:
    """
    Hybrid Vector + Lexical + Provenance-Weighted Memory Retriever.
    """

    _STOPWORDS = {
        "yang", "dan", "di", "ke", "dari", "ini", "itu", "dengan", "untuk", "pada",
        "adalah", "sebagai", "dalam", "atau", "juga", "oleh", "karena", "tentang",
        "apa", "siapa", "kapan", "dimana", "bagaimana", "mengapa", "kenapa", "berapa",
        "jelaskan", "tolong", "buatkan", "sebutkan", "ceritakan", "kamu", "anda", "saya",
        "the", "is", "are", "was", "were", "in", "on", "at", "to", "for", "of", "and",
    }

    def __init__(self, storage: StorageBackend, embed_fn: Callable[[str], List[float]]):
        self.storage = storage
        self.embed_fn = embed_fn
        self.vector_index = SemanticVectorIndex(dimension=384)

    def retrieve(
        self,
        query: str,
        memory_type: Optional[str] = None,
        top_k: int = 5,
        only_verified: bool = False,
    ) -> List[Dict[str, Any]]:
        ver_filter = "VERIFIED" if only_verified else None
        candidates = self.storage.list_records(
            memory_type=memory_type,
            verification_status=ver_filter,
            include_deleted=False,
            limit=500,
        )
        if not candidates:
            return []

        q_emb = self.embed_fn(query)
        q_terms = [t for t in re.findall(r"[a-z0-9]+", query.lower()) if len(t) > 2 and t not in self._STOPWORDS]

        ranked = self.vector_index.rank_records(
            query_embedding=q_emb,
            query_terms=q_terms,
            records=candidates,
            top_k=top_k,
        )
        out = []
        for score, rec in ranked:
            item = dict(rec)
            item["retrieval_score"] = score
            item["id"] = rec["record_id"]
            item["source"] = rec.get("source_provenance", {}).get("source_type", "verified_memory")
            out.append(item)
        return out
