"""
Stage 5 Verification: Official Python SDK (`sdk.gajahmada_sdk`), GIP/2.0 RFC Interoperability,
and Federation Endpoint Integration.
"""
from fastapi.testclient import TestClient

from gajahmada_ai.api.server import app, get_runtime_context
from gajahmada_ai.network.protocol import GIPMessageType
from sdk.gajahmada_sdk import GajahMadaClient, NodeKeyPair, build_signed_gip_message


def test_stage5_sdk_full_lifecycle_and_federation_api():
    rt = get_runtime_context()
    admin_key = rt["auth"].admin_api_key
    sdk = GajahMadaClient(api_key=admin_key, http_client=TestClient(app))

    # 1. Health & Diagnose
    h = sdk.health()
    assert h["status"] == "healthy"
    assert h["ledger"]["type"] == "tamper_evident_local_signed_event_ledger"
    d = sdk.diagnose()
    assert d["version"] == "2.0.0"

    # 2. Hybrid Ask via SDK
    res = sdk.ask("Hitung 33 * 3 + 1")
    assert "100" in res["conclusion"]

    # 3. Store LOCAL memory & register a federation organization via SDK
    kp = NodeKeyPair()
    org = sdk.register_organization(
        org_id="sdk-test-univ",
        org_name="SDK Test University Node",
        org_type="university",
        node_id=kp.node_id,
        endpoint="http://127.0.0.1:9801",
    )
    assert org["org_id"] == "sdk-test-univ"
    fed = sdk.list_federation()
    assert any(o["org_id"] == "sdk-test-univ" for o in fed["organizations"])

    # 4. Signed GIP-IDENTITY handshake via SDK helper
    signed_msg = build_signed_gip_message(
        kp,
        GIPMessageType.GIP_IDENTITY,
        payload={"node_name": "sdk-peer", "endpoint": "http://127.0.0.1:9801"},
    )
    gip_resp = sdk.send_gip(signed_msg)
    assert gip_resp["accepted"] is True
    assert gip_resp["trust_state"] == "PENDING"
