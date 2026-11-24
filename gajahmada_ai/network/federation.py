"""
Stage 4 — Collective Knowledge Network & Multi-Organization Sovereign Federation.

Enables universities, research labs, communities, and organizations to operate their
own independent GajahMada nodes while maintaining strict local ownership of PRIVATE
and LOCAL records, and contributing cryptographically attested findings to COLLECTIVE
knowledge via Ed25519 `node_attestations`.
"""
import json
import os
import time
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field

from gajahmada_ai.security.crypto import NodeIdentity, canonical_json_bytes


class OrganizationProfile(BaseModel):
    org_id: str = Field(..., min_length=2, max_length=64)
    org_name: str = Field(..., min_length=2, max_length=128)
    org_type: str = Field(default="research", description="university | research | community | organization")
    node_id: str = Field(..., min_length=16, max_length=128)
    endpoint: str = Field(..., min_length=7, max_length=256)
    data_sovereignty_policy: str = Field(
        default="STRICT_LOCAL_OWNERSHIP",
        description="PRIVATE and LOCAL records never leave the organization's node.",
    )
    allowed_shared_domains: List[str] = Field(default_factory=lambda: ["science", "history", "open-data"])
    registered_at: float = Field(default_factory=time.time)
    status: str = "ACTIVE"


class FederationRegistry:
    """
    Persistent registry of sovereign participating organizations in the Collective Knowledge Network.
    """

    def __init__(self, storage_path: Optional[str] = None):
        self.storage_path = storage_path
        self._orgs: Dict[str, OrganizationProfile] = {}
        if self.storage_path and os.path.exists(self.storage_path):
            self._load()

    def register_organization(self, profile: OrganizationProfile) -> OrganizationProfile:
        self._orgs[profile.org_id] = profile
        self._save()
        return profile

    def list_organizations(self) -> List[Dict[str, Any]]:
        return [o.model_dump() for o in self._orgs.values()]

    def get_organization(self, org_id: str) -> Optional[OrganizationProfile]:
        return self._orgs.get(org_id)

    def create_cross_org_attestation(
        self,
        verifier_identity: NodeIdentity,
        record_id: str,
        record_hash: str,
        evidence_hash: str,
        source_ref: str,
    ) -> Dict[str, str]:
        """Generate an Ed25519 cryptographic attestation binding verifier -> record -> independent evidence."""
        payload = canonical_json_bytes({
            "record_id": record_id,
            "record_hash": record_hash,
            "evidence_hash": evidence_hash,
            "source_ref": source_ref,
            "verifier_node_id": verifier_identity.node_id,
        })
        return {
            "node_id": verifier_identity.node_id,
            "signature": verifier_identity.sign_message(payload),
            "evidence_hash": evidence_hash,
            "source_ref": source_ref,
        }

    def _save(self):
        if not self.storage_path:
            return
        os.makedirs(os.path.dirname(os.path.abspath(self.storage_path)), exist_ok=True)
        tmp = f"{self.storage_path}.tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump({k: v.model_dump() for k, v in self._orgs.items()}, f, indent=2)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, self.storage_path)

    def _load(self):
        try:
            with open(self.storage_path, "r", encoding="utf-8") as f:
                raw = json.load(f)
            if isinstance(raw, dict):
                for k, v in raw.items():
                    self._orgs[k] = OrganizationProfile(**v)
        except Exception:
            self._orgs = {}
