import math
from typing import Any, Dict, List, Tuple


class SemanticVectorIndex:
    """
    384-dimensional Vector Similarity Index supporting Cosine and L2 similarity
    with lexical overlap boosting.
    """

    def __init__(self, dimension: int = 384):
        self.dimension = dimension

    def cosine_similarity(self, vec_a: List[float], vec_b: List[float]) -> float:
        if not vec_a or not vec_b or len(vec_a) != len(vec_b):
            return 0.0
        dot = sum(a * b for a, b in zip(vec_a, vec_b))
        norm_a = math.sqrt(sum(a * a for a in vec_a)) or 1.0
        norm_b = math.sqrt(sum(b * b for b in vec_b)) or 1.0
        return round(dot / (norm_a * norm_b), 6)

    def rank_records(
        self,
        query_embedding: List[float],
        query_terms: List[str],
        records: List[Dict[str, Any]],
        top_k: int = 5,
    ) -> List[Tuple[float, Dict[str, Any]]]:
        scored: List[Tuple[float, Dict[str, Any]]] = []
        for rec in records:
            if rec.get("is_deleted"):
                continue
            content_lower = str(rec.get("content", "")).lower()
            lexical_hits = sum(1 for t in query_terms if t in content_lower)
            emb = rec.get("embedding") or []
            cos_sim = self.cosine_similarity(query_embedding, emb) if len(emb) == len(query_embedding) else 0.0

            if query_terms and lexical_hits == 0 and cos_sim < 0.25:
                continue

            score = round(cos_sim + (lexical_hits * 0.35) + (float(rec.get("confidence", 0.5)) * 0.15), 4)
            scored.append((score, rec))

        scored.sort(key=lambda item: item[0], reverse=True)
        return scored[:top_k]
