import base64
import hashlib
import json
import os
import time
from typing import Dict, List, Optional, Set
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import ed25519


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

    Security properties:
    - Never uses a hardcoded or predictable default seed.
    - Generates an independent cryptographically random Ed25519 keypair per node.
    - Persists private key with strict 0600 permissions.
    - Supports key rotation with signed rotation certificates, previous key tracking,
      and peer revocation lists.
    """

    def __init__(self, key_path: Optional[str] = "./node_key.pem", allow_env_seed: bool = False):
        self.key_path = key_path
        self.allow_env_seed = allow_env_seed
        self._private_key: Optional[ed25519.Ed25519PrivateKey] = None
        self._public_key: Optional[ed25519.Ed25519PublicKey] = None
        self.node_id: str = ""
        self.key_version: int = 1
        self.created_at: float = time.time()
        self.previous_node_ids: List[str] = []
        self.revoked_node_ids: Set[str] = set()
        self.rotation_history: List[Dict[str, str]] = []
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
            except Exception:
                loaded = False

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

    def _persist_key(self):
        if not self.key_path or not self._private_key:
            return
        try:
            parent_dir = os.path.dirname(os.path.abspath(self.key_path))
            os.makedirs(parent_dir, exist_ok=True)
            pem = self._private_key.private_bytes(
                encoding=serialization.Encoding.PEM,
                format=serialization.PrivateFormat.PKCS8,
                encryption_algorithm=serialization.NoEncryption(),
            )
            fd = os.open(self.key_path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
            with os.fdopen(fd, "wb") as f:
                f.write(pem)
            try:
                os.chmod(self.key_path, 0o600)
            except OSError:
                pass
        except (OSError, IOError):
            pass

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

    def rotate_key(self) -> Dict[str, str]:
        """
        Rotate to a freshly generated Ed25519 private key and emit a signed
        rotation certificate signed by the outgoing private key.
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
        return cert

    def revoke_peer(self, peer_node_id: str):
        if peer_node_id:
            self.revoked_node_ids.add(peer_node_id.strip())

    def unrevoke_peer(self, peer_node_id: str):
        self.revoked_node_ids.discard(peer_node_id.strip())

    def is_revoked(self, peer_node_id: str) -> bool:
        return peer_node_id.strip() in self.revoked_node_ids
