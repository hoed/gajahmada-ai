import base64
import hashlib
import json
import os
import time
from typing import Any, Dict, List, Optional, Set
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import ed25519


class KeyPersistenceError(RuntimeError):
    """Raised when persistent node key or identity state cannot be safely written to disk."""


def _b64url_encode(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).decode("ascii").rstrip("=")


def _b64url_decode(data: str) -> bytes:
    clean = data.strip().rstrip("=")
    pad = "=" * (-len(clean) % 4)
    return base64.urlsafe_b64decode(clean + pad)


def canonical_json_bytes(payload: dict) -> bytes:
    """Deterministic canonical JSON serialization for cryptographic signing."""
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


class NodeIdentity:
    """
    Sovereign Node Identity using Ed25519 asymmetric keypair (RFC 8032).

    Security properties (Round 3 Hardened):
    - Never uses a hardcoded or predictable default seed.
    - Generates an independent cryptographically random Ed25519 keypair per node.
    - Atomically persists private key with strict 0600 permissions; FAILS CLOSED
      (raises KeyPersistenceError) if persistence fails when key_path is configured.
    - Persists key_version, previous_node_ids, revoked_node_ids, signed rotation_history,
      and signed revocation_records to `<key_path>.state.json` so identity governance
      survives process restarts and can be cryptographically synchronized with peers.
    """

    def __init__(self, key_path: Optional[str] = "./node_key.pem", allow_env_seed: bool = False):
        self.key_path = key_path
        self.state_path = f"{key_path}.state.json" if key_path else None
        self.allow_env_seed = allow_env_seed
        self._private_key: Optional[ed25519.Ed25519PrivateKey] = None
        self._public_key: Optional[ed25519.Ed25519PublicKey] = None
        self.node_id: str = ""
        self.key_version: int = 1
        self.created_at: float = time.time()
        self.previous_node_ids: List[str] = []
        self.revoked_node_ids: Set[str] = set()
        self.rotation_history: List[Dict[str, str]] = []
        self.revocation_records: List[Dict[str, Any]] = []
        self.load_or_create()

    def load_or_create(self):
        loaded = False

        if self.key_path and os.path.exists(self.key_path):
            try:
                with open(self.key_path, "rb") as f:
                    self._private_key = serialization.load_pem_private_key(
                        f.read(), password=None
                    )
                loaded = True
            except Exception as exc:
                raise KeyPersistenceError(
                    f"Failed to load existing node private key from '{self.key_path}': {exc}"
                ) from exc

        if not loaded:
            env_seed = os.environ.get("GAJAHMADA_NODE_SEED", "").strip()
            if self.allow_env_seed and env_seed:
                if len(env_seed) < 32 or env_seed == "gajahmada-sovereign-ai-vercel-node-v2":
                    raise ValueError(
                        "Security violation: Predictable or low-entropy GAJAHMADA_NODE_SEED rejected. "
                        "Provide at least 32 characters of high-entropy secret material or omit for random generation."
                    )
                seed_bytes = hashlib.sha256(env_seed.encode("utf-8")).digest()
                self._private_key = ed25519.Ed25519PrivateKey.from_private_bytes(seed_bytes)
            else:
                self._private_key = ed25519.Ed25519PrivateKey.generate()

            self._persist_key()

        self._derive_public_identity()
        self._load_state()

    def _persist_key(self):
        if not self.key_path or not self._private_key:
            return
        try:
            abs_key = os.path.abspath(self.key_path)
            parent_dir = os.path.dirname(abs_key)
            os.makedirs(parent_dir, exist_ok=True)
            pem = self._private_key.private_bytes(
                encoding=serialization.Encoding.PEM,
                format=serialization.PrivateFormat.PKCS8,
                encryption_algorithm=serialization.NoEncryption(),
            )
            tmp_path = f"{abs_key}.tmp"
            fd = os.open(tmp_path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
            with os.fdopen(fd, "wb") as f:
                f.write(pem)
                f.flush()
                os.fsync(f.fileno())
            os.chmod(tmp_path, 0o600)
            os.replace(tmp_path, abs_key)
        except OSError as exc:
            raise KeyPersistenceError(
                f"Critical security failure: unable to persist node private key to '{self.key_path}': {exc}"
            ) from exc

    def _persist_state(self):
        if not self.state_path:
            return
        try:
            abs_state = os.path.abspath(self.state_path)
            os.makedirs(os.path.dirname(abs_state), exist_ok=True)
            payload = {
                "node_id": self.node_id,
                "key_version": self.key_version,
                "created_at": self.created_at,
                "previous_node_ids": list(self.previous_node_ids),
                "revoked_node_ids": sorted(self.revoked_node_ids),
                "rotation_history": list(self.rotation_history),
                "revocation_records": list(self.revocation_records),
            }
            tmp_path = f"{abs_state}.tmp"
            fd = os.open(tmp_path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2)
                f.flush()
                os.fsync(f.fileno())
            os.chmod(tmp_path, 0o600)
            os.replace(tmp_path, abs_state)
        except OSError as exc:
            raise KeyPersistenceError(
                f"Critical security failure: unable to persist identity state to '{self.state_path}': {exc}"
            ) from exc

    def _load_state(self):
        if not self.state_path or not os.path.exists(self.state_path):
            return
        try:
            with open(self.state_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict):
                self.key_version = int(data.get("key_version", self.key_version))
                self.created_at = float(data.get("created_at", self.created_at))
                self.previous_node_ids = list(data.get("previous_node_ids", []))
                self.revoked_node_ids = set(data.get("revoked_node_ids", []))
                self.rotation_history = list(data.get("rotation_history", []))
                self.revocation_records = list(data.get("revocation_records", []))
        except Exception as exc:
            raise KeyPersistenceError(
                f"Failed to load persisted identity state from '{self.state_path}': {exc}"
            ) from exc

    def _derive_public_identity(self):
        self._public_key = self._private_key.public_key()
        raw_pub = self._public_key.public_bytes(
            encoding=serialization.Encoding.Raw,
            format=serialization.PublicFormat.Raw,
        )
        self.node_id = _b64url_encode(raw_pub)

    def sign_message(self, message: bytes) -> str:
        if isinstance(message, str):
            message = message.encode("utf-8")
        signature = self._private_key.sign(message)
        return _b64url_encode(signature)

    def sign_dict(self, payload: dict) -> str:
        return self.sign_message(canonical_json_bytes(payload))

    def verify_message(self, message: bytes, signature: str, node_id: str) -> bool:
        if not signature or not node_id:
            return False
        if node_id in self.revoked_node_ids:
            return False
        if isinstance(message, str):
            message = message.encode("utf-8")
        try:
            raw_sig = _b64url_decode(signature)
            raw_pub = _b64url_decode(node_id)
            if len(raw_pub) != 32 or len(raw_sig) != 64:
                return False
            remote_pub = ed25519.Ed25519PublicKey.from_public_bytes(raw_pub)
            remote_pub.verify(raw_sig, message)
            return True
        except Exception:
            return False

    def verify_dict(self, payload: dict, signature: str, node_id: str) -> bool:
        return self.verify_message(canonical_json_bytes(payload), signature, node_id)

    @staticmethod
    def verify_rotation_certificate(cert: Dict[str, Any]) -> bool:
        """
        Verify a dual-signed key rotation certificate (signed by BOTH old_node_id and new_node_id).
        """
        try:
            old_id = str(cert.get("old_node_id", "")).strip()
            new_id = str(cert.get("new_node_id", "")).strip()
            ts = str(cert.get("timestamp", "")).strip()
            old_sig = str(cert.get("old_signature", "")).strip()
            new_sig = str(cert.get("new_signature", "")).strip()
            if not (old_id and new_id and ts and old_sig and new_sig) or old_id == new_id:
                return False
            statement = f"ROTATE|{old_id}|{new_id}|{ts}".encode("utf-8")
            old_pub = ed25519.Ed25519PublicKey.from_public_bytes(_b64url_decode(old_id))
            new_pub = ed25519.Ed25519PublicKey.from_public_bytes(_b64url_decode(new_id))
            old_pub.verify(_b64url_decode(old_sig), statement)
            new_pub.verify(_b64url_decode(new_sig), statement)
            return True
        except Exception:
            return False

    def apply_peer_rotation_certificate(self, cert: Dict[str, Any]) -> bool:
        """
        Verify a peer's dual-signed rotation certificate and persist revocation of its old key.
        """
        if not self.verify_rotation_certificate(cert):
            return False
        old_id = str(cert["old_node_id"]).strip()
        self.revoked_node_ids.add(old_id)
        self.rotation_history.append(dict(cert))
        self._persist_state()
        return True

    def rotate_key(self) -> Dict[str, str]:
        """
        Rotate to a freshly generated Ed25519 private key, emit a dual-signed
        rotation certificate, and persist both key and rotation/revocation state atomically.
        """
        old_node_id = self.node_id
        old_private_key = self._private_key

        new_private_key = ed25519.Ed25519PrivateKey.generate()
        new_pub = new_private_key.public_key().public_bytes(
            encoding=serialization.Encoding.Raw,
            format=serialization.PublicFormat.Raw,
        )
        new_node_id = _b64url_encode(new_pub)
        timestamp = str(round(time.time(), 3))

        rotation_statement = f"ROTATE|{old_node_id}|{new_node_id}|{timestamp}".encode("utf-8")
        old_sig = _b64url_encode(old_private_key.sign(rotation_statement))
        new_sig = _b64url_encode(new_private_key.sign(rotation_statement))

        self.previous_node_ids.append(old_node_id)
        self.revoked_node_ids.add(old_node_id)
        self._private_key = new_private_key
        self.key_version += 1
        self._derive_public_identity()
        self._persist_key()

        cert = {
            "old_node_id": old_node_id,
            "new_node_id": new_node_id,
            "key_version": str(self.key_version),
            "timestamp": timestamp,
            "old_signature": old_sig,
            "new_signature": new_sig,
        }
        self.rotation_history.append(cert)
        self._persist_state()
        return cert

    def revoke_peer(self, peer_node_id: str, reason: str = "operator_revoked") -> Dict[str, Any]:
        clean = peer_node_id.strip() if peer_node_id else ""
        if not clean:
            return {}
        self.revoked_node_ids.add(clean)
        notice_payload = {
            "action": "REVOKE",
            "revoked_node_id": clean,
            "issuer_node_id": self.node_id,
            "reason": reason,
            "timestamp": round(time.time(), 3),
        }
        notice = {
            **notice_payload,
            "signature": self.sign_dict(notice_payload),
        }
        self.revocation_records.append(notice)
        self._persist_state()
        return notice

    def unrevoke_peer(self, peer_node_id: str):
        if peer_node_id:
            self.revoked_node_ids.discard(peer_node_id.strip())
            self._persist_state()

    def is_revoked(self, peer_node_id: str) -> bool:
        return peer_node_id.strip() in self.revoked_node_ids
