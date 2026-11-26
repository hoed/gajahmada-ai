"""
GajahMada AI — Verifiable Signed Knowledge Envelope & Organizational Independence (Phase 4 P2).

A public key proves control of a key, NOT that two nodes belong to independent operators.
This module provides:
1. `SignedKnowledgeEnvelope`: Ed25519-signed envelope binding `content_hash`, `source_id`,
   `organization_id`, `sender_node_id`, `timestamp`, `evidence_refs`, `verification_status`,
   `knowledge_version`, and `expires_at`.
2. `count_independent_operators`: counts distinct verified `(organization_id)` values so
   multiple keys belonging to the same organization cannot inflate independence counts.
"""
import hashlib
import time
from typing import Any, Dict, List, Optional, Tuple
from pydantic import BaseModel, Field

from gajahmada_ai.security.crypto import NodeIdentity, canonical_json_bytes


class SignedKnowledgeEnvelope(BaseModel):
    knowledge_version: int = 1
    content: str = Field(..., min_length=1)
    content_hash: str = ""
    source_id: str = Field(..., min_length=2)
    organization_id: str = Field(..., min_length=2)
    sender_node_id: str = ""
    timestamp: float = Field(default_factory=lambda: round(time.time(), 3))
    expires_at: Optional[float] = None
    evidence_refs: List[str] = Field(default_factory=list)
    verification_status: str = "UNVERIFIED"
    signature: str = ""

    def compute_content_hash(self) -> str:
        return hashlib.sha256(self.content.strip().encode("utf-8")).hexdigest()

    def signable_dict(self) -> Dict[str, Any]:
        return {
            "knowledge_version": self.knowledge_version,
            "content_hash": self.content_hash or self.compute_content_hash(),
            "source_id": self.source_id,
            "organization_id": self.organization_id,
            "sender_node_id": self.sender_node_id,
            "timestamp": self.timestamp,
            "expires_at": self.expires_at,
            "evidence_refs": list(self.evidence_refs),
            "verification_status": self.verification_status,
        }

    def sign(self, identity: NodeIdentity) -> "SignedKnowledgeEnvelope":
        self.sender_node_id = identity.node_id
        self.content_hash = self.compute_content_hash()
        self.signature = identity.sign_message(canonical_json_bytes(self.signable_dict()))
        return self

    def verify(self, verifier: NodeIdentity, now: Optional[float] = None) -> Tuple[bool, str]:
        ts_now = now if now is not None else time.time()
        if not self.sender_node_id or not self.signature:
            return False, "missing_sender_or_signature"
        if verifier.is_revoked(self.sender_node_id):
            return False, "sender_node_revoked"
        if self.expires_at is not None and ts_now > self.expires_at:
            return False, "envelope_expired"
        expected_hash = self.compute_content_hash()
        if self.content_hash != expected_hash:
            return False, "content_hash_mismatch"
        if not verifier.verify_message(canonical_json_bytes(self.signable_dict()), self.signature, self.sender_node_id):
            return False, "ed25519_signature_invalid"
        return True, "VALID"


def count_independent_operators(
    envelopes: List[SignedKnowledgeEnvelope],
    verifier: NodeIdentity,
) -> Dict[str, Any]:
    """
    Verify all envelopes and count distinct independent organizations (NOT just distinct public keys).
    """
    valid_orgs = set()
    valid_nodes = set()
    valid_evidence = set()
    rejected: List[Dict[str, str]] = []

    for env in envelopes:
        ok, reason = env.verify(verifier)
        if not ok:
            rejected.append({"sender_node_id": env.sender_node_id[:16], "reason": reason})
            continue
        valid_orgs.add(env.organization_id.strip().lower())
        valid_nodes.add(env.sender_node_id)
        for ref in env.evidence_refs:
            if ref and ref.strip():
                valid_evidence.add(ref.strip())

    return {
        "valid_envelopes": len(envelopes) - len(rejected),
        "rejected_envelopes": len(rejected),
        "distinct_nodes": len(valid_nodes),
        "independent_organizations": len(valid_orgs),
        "independent_evidence_refs": len(valid_evidence),
        "independently_corroborated": len(valid_orgs) >= 2 and len(valid_evidence) >= 2,
    }
