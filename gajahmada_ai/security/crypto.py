import base64
import hashlib
import os
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import ed25519


def _b64url_decode(data: str) -> bytes:
    clean = data.strip().rstrip("=")
    pad = "=" * (-len(clean) % 4)
    return base64.urlsafe_b64decode(clean + pad)


class NodeIdentity:
    """
    Node identity using Ed25519 keypair (RFC 8032).

    - On localhost: key is persisted to key_path with 0600 permissions.
    - On Vercel serverless: uses deterministic seed (GAJAHMADA_NODE_SEED)
      or /tmp/node_key.pem so node_id & signatures stay consistent across
      serverless invocations.
    """

    def __init__(self, key_path: str = "./node_key.pem"):
        self.key_path = key_path
        self._private_key = None
        self._public_key = None
        self.node_id = ""
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
            if os.environ.get("VERCEL") == "1" or os.environ.get("GAJAHMADA_NODE_SEED"):
                seed_str = os.environ.get("GAJAHMADA_NODE_SEED", "gajahmada-sovereign-ai-vercel-node-v2")
                seed_bytes = hashlib.sha256(seed_str.encode("utf-8")).digest()
                self._private_key = ed25519.Ed25519PrivateKey.from_private_bytes(seed_bytes)
            else:
                self._private_key = ed25519.Ed25519PrivateKey.generate()

            if self.key_path:
                try:
                    os.makedirs(os.path.dirname(os.path.abspath(self.key_path)), exist_ok=True)
                    pem = self._private_key.private_bytes(
                        encoding=serialization.Encoding.PEM,
                        format=serialization.PrivateFormat.PKCS8,
                        encryption_algorithm=serialization.NoEncryption(),
                    )
                    fd = os.open(self.key_path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
                    with os.fdopen(fd, "wb") as f:
                        f.write(pem)
                except (OSError, IOError):
                    pass

        self._public_key = self._private_key.public_key()
        raw_pub = self._public_key.public_bytes(
            encoding=serialization.Encoding.Raw,
            format=serialization.PublicFormat.Raw,
        )
        self.node_id = base64.urlsafe_b64encode(raw_pub).decode().rstrip("=")

    def sign_message(self, message: bytes) -> str:
        signature = self._private_key.sign(message)
        return base64.urlsafe_b64encode(signature).decode().rstrip("=")

    def verify_message(self, message: bytes, signature: str, node_id: str) -> bool:
        try:
            raw_sig = _b64url_decode(signature)
            raw_pub = _b64url_decode(node_id)
            remote_pub = ed25519.Ed25519PublicKey.from_public_bytes(raw_pub)
            remote_pub.verify(raw_sig, message)
            return True
        except Exception:
            return False
