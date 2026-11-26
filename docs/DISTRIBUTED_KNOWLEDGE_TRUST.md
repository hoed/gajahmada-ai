# GajahMada AI — Distributed Knowledge Trust (`DISTRIBUTED_KNOWLEDGE_TRUST.md`)

## Maturity Classification
- **GIP/2.0 8-Namespace Signed Envelopes & SQLite WAL Replay Protection (`gip_replay.db`):** **IMPLEMENTED AND TESTED**
- **`PeerTrustState` Lifecycle (`UNKNOWN → PENDING → AUTHENTICATED → TRUSTED → REVOKED`):** **IMPLEMENTED AND TESTED**
- **`SignedKnowledgeEnvelope` & Organization-Level Independence (`gajahmada_ai/network/signed_knowledge.py`):** **IMPLEMENTED AND TESTED**
- **Multi-Organization `FederationRegistry` & Ed25519 `node_attestations`:** **IMPLEMENTED AND TESTED**
- **WAN libp2p QUIC & BFT Replicated Finality:** **PLANNED**

## Trust Principles
- A valid Ed25519 signature proves **key ownership**, NOT network trust or organizational independence.
- Two distinct public keys belonging to the **same `organization_id`** count as only **1** independent operator in `count_independent_operators()`.
- `SignedKnowledgeEnvelope` cryptographically binds `content_hash`, `source_id`, `organization_id`, `sender_node_id`, `timestamp`, `expires_at`, `evidence_refs`, `verification_status`, and `knowledge_version`, rejecting tampered, expired, or revoked envelopes.
- Local ledger (`AIBlockchain` Schema v2) is explicitly classified as a **tamper-evident local signed event ledger**, not a public BFT distributed consensus blockchain.
