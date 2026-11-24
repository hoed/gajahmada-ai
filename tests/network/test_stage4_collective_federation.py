"""
Stage 4 Verification: Collective Knowledge Network Multi-Organization Federation.
Verifies that 4 independent organizational nodes (University, Research Lab, Community,
Organization) preserve strict data ownership over PRIVATE/LOCAL records while co-verifying
shared findings into COLLECTIVE knowledge via Ed25519 node_attestations.
"""
import os
import tempfile

from fastapi.testclient import TestClient

from gajahmada_ai.api.server import app, get_runtime_context
from gajahmada_ai.memory.authorization import MemoryAuthorizationService
from gajahmada_ai.memory.models import MemoryType, VerificationStatus, VisibilityPolicy
from gajahmada_ai.network.coordinator import SovereignPeerNode
from gajahmada_ai.network.federation import FederationRegistry, OrganizationProfile


def test_stage4_multi_organization_sovereign_data_ownership_and_collective_attestation():
    with tempfile.TemporaryDirectory(prefix="gm_stage4_") as tmp:
        reg_path = os.path.join(tmp, "federation.json")
        fed = FederationRegistry(storage_path=reg_path)

        univ = SovereignPeerNode("univ-ugm", "http://127.0.0.1:9701", storage_dir=os.path.join(tmp, "univ"))
        lab = SovereignPeerNode("brin-lab", "http://127.0.0.1:9702", storage_dir=os.path.join(tmp, "lab"))
        comm = SovereignPeerNode("komunitas-ai", "http://127.0.0.1:9703", storage_dir=os.path.join(tmp, "comm"))
        org = SovereignPeerNode("ngo-maritim", "http://127.0.0.1:9704", storage_dir=os.path.join(tmp, "org"))

        for oid, oname, otype, node in [
            ("univ-01", "Universitas Nusantara", "university", univ),
            ("lab-01", "Lab Riset Nasional", "research", lab),
            ("comm-01", "Komunitas AI Terbuka", "community", comm),
            ("org-01", "Observatorium Maritim", "organization", org),
        ]:
            fed.register_organization(
                OrganizationProfile(
                    org_id=oid,
                    org_name=oname,
                    org_type=otype,
                    node_id=node.identity.node_id,
                    endpoint=node.endpoint,
                )
            )

        # Reload registry from disk to verify persistence
        fed_reloaded = FederationRegistry(storage_path=reg_path)
        assert len(fed_reloaded.list_organizations()) == 4

        # Each org stores PRIVATE/LOCAL internal research — MUST NOT be readable by external peers or anonymous
        univ_priv = univ.memory.store(
            content="Confidential student cohort dataset #2026-A",
            memory_type=MemoryType.SEMANTIC,
            origin_node_id=univ.identity.node_id,
            visibility_policy=VisibilityPolicy.PRIVATE,
        )
        lab_local = lab.memory.store(
            content="Preliminary unverified lab measurement #99",
            memory_type=MemoryType.SEMANTIC,
            origin_node_id=lab.identity.node_id,
            visibility_policy=VisibilityPolicy.LOCAL,
        )
        authz = MemoryAuthorizationService()
        assert authz.can_read_record(univ_priv, "peer") is False
        assert authz.can_read_record(univ_priv, "anonymous") is False
        assert authz.can_read_record(lab_local, "peer") is False
        assert authz.can_read_record(lab_local, "anonymous") is False
        assert authz.can_read_record(univ_priv, "admin") is True
        assert authz.can_read_record(lab_local, "operator") is True

        # Cross-org Ed25519 attestation promotes a quarantined finding on the hub API to INDEPENDENTLY_VERIFIED
        client = TestClient(app)
        rt = get_runtime_context()
        admin_key = rt["auth"].admin_api_key

        # Register & trust University and Research Lab on the hub node
        for org_node, name in [(univ, "univ-ugm"), (lab, "brin-lab")]:
            r_reg = client.post(
                "/api/v2/nodes/register",
                headers={"X-API-Key": admin_key},
                json={"node_id": org_node.identity.node_id, "endpoint": org_node.endpoint, "node_name": name},
            )
            assert r_reg.status_code == 200

        # Store quarantined observation on hub
        r_mem = client.post(
            "/api/v2/memory",
            headers={"X-API-Key": admin_key},
            json={
                "content": "Coral reef recovery rate at Raja Ampat increased 14.2% in 2026.",
                "visibility_policy": "LOCAL",
                "verification_status": "UNVERIFIED",
            },
        )
        assert r_mem.status_code == 200
        rec = r_mem.json()

        att_univ = fed.create_cross_org_attestation(
            univ.identity, rec["record_id"], rec["integrity_hash"], "sha256-univ-dive-log-2026", "https://univ.ac.id/reef-2026"
        )
        att_lab = fed.create_cross_org_attestation(
            lab.identity, rec["record_id"], rec["integrity_hash"], "sha256-lab-satellite-2026", "https://lab.go.id/sat-reef-2026"
        )

        r_prom = client.post(
            f"/api/v2/knowledge/promote/{rec['record_id']}",
            headers={"X-API-Key": admin_key},
            json={
                "independent_sources": ["https://univ.ac.id/reef-2026", "https://lab.go.id/sat-reef-2026"],
                "verification_method": "independent_node_verification",
                "node_attestations": [att_univ, att_lab],
            },
        )
        assert r_prom.status_code == 200
        assert r_prom.json()["epistemic_status"] == "INDEPENDENTLY_VERIFIED"
        assert r_prom.json()["record"]["visibility_policy"] == "COLLECTIVE"
