# GIP/2.0 — GajahMada Intelligence Protocol Specification (RFC Draft v2.0)

**Status:** Open Interoperability Specification (Stage 5 — Independent AI Ecosystem)  
**Transport:** Authenticated HTTPS JSON (libp2p QUIC-ready abstraction)  
**Cryptography:** Ed25519 (RFC 8032), SHA-256, Canonical JSON (`sort_keys=True, separators=(",", ":")`)

---

## 1. Wire Envelope Format

Every GIP/2.0 message MUST conform to the following JSON object:

```json
{
  "protocol_version": "GIP/2.0",
  "message_id": "gip-<14-hex>",
  "message_type": "GIP-IDENTITY | GIP-DISCOVERY | GIP-CAPABILITY | GIP-TASK | GIP-KNOWLEDGE | GIP-VERIFY | GIP-CONSENSUS | GIP-SYNC",
  "sender_node_id": "<base64url-32-byte-ed25519-pubkey>",
  "recipient_node_id": "* | <base64url-32-byte-ed25519-pubkey>",
  "timestamp": 1763970000.1234,
  "nonce": "<24-hex-cryptographic-random>",
  "payload": {},
  "signature": "<base64url-64-byte-ed25519-signature>"
}
```

### Signature Computation
The `signature` MUST be computed using the sender's Ed25519 private key over the UTF-8 canonical JSON serialization (`sort_keys=True`, `separators=(",", ":")`, `ensure_ascii=False`) of all envelope fields **except** `signature`.

---

## 2. Mandatory Receiver Validation Rules

A conformant GIP/2.0 receiver MUST enforce the following checks in order:
1. `protocol_version == "GIP/2.0"`.
2. Non-empty `sender_node_id`, `signature`, and `nonce`.
3. `sender_node_id` is NOT in the node's persistent revocation list (`node_key.pem.state.json`).
4. `recipient_node_id` is `"*"` or matches the local node's `node_id`.
5. `|now - timestamp| <= max_clock_skew_sec` (default `300s`).
6. Valid Ed25519 signature over the canonical envelope (verified **before** recording nonce to prevent unauthenticated nonce-table flooding).
7. Payload `action` MUST NOT match forbidden remote actions (`exec_shell`, `shell_command`, `override_policy`, `export_private_key`, `delete_private_memory`, `modify_config`).
8. Atomic, restart-persistent replay protection: `(sender_node_id, nonce)` MUST be recorded atomically with TTL `>= 2 * max_clock_skew_sec` (e.g., SQLite WAL `BEGIN IMMEDIATE` + `INSERT OR IGNORE` or shared Redis/PostgreSQL).

---

## 3. Peer Trust Lifecycle (`PeerTrustState`)

Cryptographic signature validity proves **key ownership**, NOT network trust.
- `UNKNOWN` → `PENDING` (upon SSRF-validated `GIP-IDENTITY` / `GIP-DISCOVERY`) → `AUTHENTICATED` → `TRUSTED` (explicit operator promotion via `POST /api/v2/nodes/{node_id}/trust`) → `REVOKED`.
- Operational namespaces (`GIP-TASK`, `GIP-KNOWLEDGE`, `GIP-VERIFY`, `GIP-CONSENSUS`, `GIP-SYNC`) MUST reject any remote peer whose trust state is not `TRUSTED`.

---

## 4. Cryptographic Evidence Attestation Contract (`node_attestations`)

To promote a quarantined record to `INDEPENDENTLY_VERIFIED` across organizations:
- At least **2 distinct `TRUSTED` verifiers** MUST each sign:
  ```json
  {
    "evidence_hash": "<sha256-of-independent-evidence>",
    "record_hash": "<record-integrity-hash>",
    "record_id": "<record-id>",
    "source_ref": "<traceable-source-uri>",
    "verifier_node_id": "<verifier-ed25519-node-id>"
  }
  ```
- At least **2 distinct `evidence_hash` values** MUST be present. Plain URL strings without cryptographic attestations MUST NOT grant `INDEPENDENTLY_VERIFIED`.

---

## 5. Key Rotation & Revocation Certificates

A node rotating its Ed25519 identity MUST emit a **dual-signed rotation certificate** signing `ROTATE|<old_node_id>|<new_node_id>|<timestamp>` with **both** the outgoing and incoming private keys. Peers verify both signatures before revoking `old_node_id` persistently.
