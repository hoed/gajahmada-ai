"""
GajahMada Official Python SDK (Stage 5 — Independent AI Ecosystem).

Provides a clean, interoperable client for interacting with GajahMada 2.0 Sovereign Nodes:
- `GajahMadaClient`: HTTP / ASGI-compatible client for /api/v2/* endpoints.
- `NodeKeyPair`: Standalone Ed25519 identity & cryptographic attestation helper.
- `build_signed_gip_message`: Interoperable GIP/2.0 signed envelope builder.
"""

from .client import GajahMadaClient, NodeKeyPair, build_signed_gip_message

__all__ = ["GajahMadaClient", "NodeKeyPair", "build_signed_gip_message"]
__version__ = "2.0.0"
