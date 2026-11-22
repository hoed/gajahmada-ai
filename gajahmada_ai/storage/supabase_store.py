"""
GajahMada 2.0 — Optional Supabase (PostgreSQL + pgvector) Remote Knowledge & Ledger Store.

Local-First Decentralized Design:
- Every node runs independently with local SQLite (`gajahmada_v2.db`) and local RAG.
- When `SUPABASE_URL` and `SUPABASE_SECRET_KEY` (or `SUPABASE_PUBLISHABLE_KEY`) are present
  in the environment or `.env`, this adapter transparently syncs COLLECTIVE knowledge records
  (content-addressed by SHA-256 `content_hash`) and audit ledger blocks to Supabase PostgREST,
  and enables persistent `pgvector` semantic search across serverless/web deployments (Vercel).
"""

import hashlib
import os
from typing import Any, Dict, List, Optional
import httpx


def load_dotenv_file(dotenv_path: Optional[str] = None) -> None:
    """Zero-dependency `.env` loader (does not overwrite existing environment variables)."""
    if dotenv_path is None:
        here = os.path.dirname(os.path.abspath(__file__))
        dotenv_path = os.path.normpath(os.path.join(here, "..", "..", ".env"))
    if not os.path.exists(dotenv_path):
        return
    try:
        with open(dotenv_path, "r", encoding="utf-8") as f:
            for raw in f:
                line = raw.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                k, v = line.split("=", 1)
                k = k.strip()
                v = v.strip().strip('"').strip("'")
                if k and k not in os.environ:
                    os.environ[k] = v
    except Exception:
        pass


class SupabaseKnowledgeStore:
    """
    Lightweight PostgREST + pgvector client for GajahMada 2.0.
    Uses `httpx` directly so it adds zero heavy SDK dependencies and runs cleanly on Vercel.
    """

    def __init__(
        self,
        url: Optional[str] = None,
        secret_key: Optional[str] = None,
        publishable_key: Optional[str] = None,
        timeout_sec: float = 4.0,
    ):
        load_dotenv_file()
        self.url = (url or os.environ.get("SUPABASE_URL") or os.environ.get("VITE_SUPABASE_URL") or "").rstrip("/")
        self.secret_key = (
            secret_key
            or os.environ.get("SUPABASE_SECRET_KEY")
            or os.environ.get("SUPABASE_SERVICE_ROLE_KEY")
            or ""
        ).strip()
        self.publishable_key = (
            publishable_key
            or os.environ.get("SUPABASE_PUBLISHABLE_KEY")
            or os.environ.get("VITE_SUPABASE_PUBLISHABLE_KEY")
            or os.environ.get("SUPABASE_ANON_KEY")
            or ""
        ).strip()
        self.api_key = self.secret_key or self.publishable_key
        self.timeout = httpx.Timeout(timeout_sec)
        self._last_error: Optional[str] = None
        self._connected: Optional[bool] = None

    @property
    def configured(self) -> bool:
        return bool(self.url and self.api_key)

    def _headers(self, extra: Optional[Dict[str, str]] = None) -> Dict[str, str]:
        h = {
            "apikey": self.api_key,
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        if extra:
            h.update(extra)
        return h

    @staticmethod
    def content_hash(text: str) -> str:
        return hashlib.sha256(text.strip().encode("utf-8")).hexdigest()

    @staticmethod
    def _format_vector(embedding: Optional[List[float]]) -> Optional[str]:
        if not embedding:
            return None
        return "[" + ",".join(f"{float(x):.6f}" for x in embedding) + "]"

    def health(self) -> Dict[str, Any]:
        if not self.configured:
            return {
                "configured": False,
                "connected": False,
                "url": None,
                "key_mode": None,
                "reason": "SUPABASE_URL or key not configured (local-first SQLite mode active)",
            }
        key_mode = "secret" if self.secret_key else "publishable"
        return {
            "configured": True,
            "connected": bool(self._connected),
            "url": self.url,
            "key_mode": key_mode,
            "last_error": self._last_error,
        }

    def ping(self) -> bool:
        if not self.configured:
            self._connected = False
            return False
        try:
            with httpx.Client(timeout=self.timeout) as client:
                resp = client.get(
                    f"{self.url}/rest/v1/knowledge_records?select=content_hash&limit=1",
                    headers=self._headers(),
                )
                if resp.status_code in (200, 206):
                    self._connected = True
                    self._last_error = None
                    return True
                self._connected = False
                self._last_error = f"HTTP {resp.status_code}: {resp.text[:160]}"
                return False
        except Exception as e:
            self._connected = False
            self._last_error = str(e)
            return False

    def upsert_knowledge(
        self,
        content: str,
        embedding: Optional[List[float]] = None,
        source_type: str = "cognitive_brain_qa_update",
        source_url: str = "",
        origin_node_id: str = "local",
        signature: str = "",
        confidence: float = 0.75,
        verification_status: str = "UNVERIFIED",
        visibility: str = "COLLECTIVE",
        allow_local_sync: bool = False,
    ) -> Optional[str]:
        """Content-addressed upsert into `public.knowledge_records`. Returns `content_hash` on success."""
        if not self.configured or not content or not content.strip():
            return None
        if visibility.upper() == "LOCAL" and not allow_local_sync:
            return None

        c_hash = self.content_hash(content)
        payload: Dict[str, Any] = {
            "content_hash": c_hash,
            "content": content.strip(),
            "source_type": source_type,
            "source_url": source_url,
            "origin_node_id": origin_node_id,
            "signature": signature,
            "confidence": float(confidence),
            "verification_status": verification_status,
            "visibility": visibility.upper(),
        }
        vec_str = self._format_vector(embedding)
        if vec_str is not None:
            payload["embedding"] = vec_str

        try:
            with httpx.Client(timeout=self.timeout) as client:
                resp = client.post(
                    f"{self.url}/rest/v1/knowledge_records?on_conflict=content_hash",
                    headers=self._headers({"Prefer": "resolution=merge-duplicates,return=minimal"}),
                    json=payload,
                )
                if resp.status_code in (200, 201, 204):
                    self._connected = True
                    self._last_error = None
                    return c_hash
                self._last_error = f"HTTP {resp.status_code}: {resp.text[:160]}"
                return None
        except Exception as e:
            self._last_error = str(e)
            return None

    def search_knowledge(
        self,
        query_embedding: List[float],
        k: int = 5,
        min_similarity: float = 0.15,
    ) -> List[Dict[str, Any]]:
        """Semantic vector search via `public.match_knowledge` pgvector RPC."""
        if not self.configured or not query_embedding:
            return []
        try:
            with httpx.Client(timeout=self.timeout) as client:
                resp = client.post(
                    f"{self.url}/rest/v1/rpc/match_knowledge",
                    headers=self._headers(),
                    json={
                        "query_embedding": self._format_vector(query_embedding),
                        "match_count": int(k),
                        "min_similarity": float(min_similarity),
                    },
                )
                if resp.status_code == 200:
                    self._connected = True
                    self._last_error = None
                    data = resp.json()
                    return data if isinstance(data, list) else []
                self._last_error = f"HTTP {resp.status_code}: {resp.text[:160]}"
                return []
        except Exception as e:
            self._last_error = str(e)
            return []

    def count_knowledge(self) -> Optional[int]:
        if not self.configured:
            return None
        try:
            with httpx.Client(timeout=self.timeout) as client:
                resp = client.get(
                    f"{self.url}/rest/v1/knowledge_records?select=content_hash",
                    headers=self._headers({"Prefer": "count=exact", "Range": "0-0"}),
                )
                if resp.status_code in (200, 206):
                    cr = resp.headers.get("content-range", "")
                    if "/" in cr:
                        total = cr.split("/")[-1]
                        if total.isdigit():
                            self._connected = True
                            return int(total)
                    return len(resp.json())
        except Exception as e:
            self._last_error = str(e)
        return None

    def upsert_block(self, block_dict: Dict[str, Any]) -> bool:
        """Persist a signed Web3 audit ledger block into `public.ledger_blocks`."""
        if not self.configured or not block_dict:
            return False
        payload = {
            "block_index": int(block_dict.get("index", 0)),
            "timestamp": float(block_dict.get("timestamp", 0.0)),
            "task_hash": str(block_dict.get("task_hash", "")),
            "node_id": str(block_dict.get("node_id", "")),
            "previous_hash": str(block_dict.get("previous_hash", "")),
            "signature": str(block_dict.get("signature", "")),
            "event_type": str(block_dict.get("event_type", "TASK_COMPLETED")),
            "event_ref": str(block_dict.get("event_ref", "")),
            "nonce": int(block_dict.get("nonce", 0)),
            "block_hash": str(block_dict.get("hash", "")),
        }
        try:
            with httpx.Client(timeout=self.timeout) as client:
                resp = client.post(
                    f"{self.url}/rest/v1/ledger_blocks?on_conflict=block_index",
                    headers=self._headers({"Prefer": "resolution=merge-duplicates,return=minimal"}),
                    json=payload,
                )
                if resp.status_code in (200, 201, 204):
                    self._connected = True
                    self._last_error = None
                    return True
                self._last_error = f"HTTP {resp.status_code}: {resp.text[:160]}"
                return False
        except Exception as e:
            self._last_error = str(e)
            return False
