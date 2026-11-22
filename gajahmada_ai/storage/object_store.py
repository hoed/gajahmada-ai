import hashlib
import json
import os
import time
from typing import Any, Dict, Optional


class ContentAddressedObjectStore:
    """
    Content-addressed persistent object store for memory snapshots, backups,
    and large evidence artifacts (referenced by SHA-256 hash on the blockchain ledger).
    """

    def __init__(self, base_dir: str = "./knowledge_store/objects"):
        if os.environ.get("VERCEL") == "1" and not base_dir.startswith("/tmp"):
            base_dir = "/tmp/knowledge_store/objects"
        self.base_dir = base_dir
        os.makedirs(self.base_dir, exist_ok=True)

    def put_json(self, payload: Any) -> Dict[str, Any]:
        raw = json.dumps(payload, sort_keys=True, ensure_ascii=False, indent=2).encode("utf-8")
        sha256_hash = hashlib.sha256(raw).hexdigest()
        file_path = os.path.join(self.base_dir, f"{sha256_hash}.json")
        with open(file_path, "wb") as f:
            f.write(raw)
        return {
            "object_hash": sha256_hash,
            "path": file_path,
            "size_bytes": len(raw),
            "timestamp": round(time.time(), 3),
        }

    def get_json(self, object_hash: str) -> Optional[Any]:
        clean = os.path.basename(object_hash.strip())
        file_path = os.path.join(self.base_dir, f"{clean}.json")
        if not os.path.exists(file_path):
            return None
        with open(file_path, "rb") as f:
            raw = f.read()
        if hashlib.sha256(raw).hexdigest() != clean:
            raise ValueError(f"Object integrity check failed for hash {clean}")
        return json.loads(raw.decode("utf-8"))
