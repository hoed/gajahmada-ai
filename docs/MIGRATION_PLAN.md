# GAJAHMADA 2.0 — Migration & Backward Compatibility Plan (`MIGRATION_PLAN.md`)

---

## 1. Core Migration Principles

1. **Zero Destruction of Working Functionality:** All 20+ existing v1 REST endpoints (`/`, `/api/status`, `/health`, `/ai/chat`, `/ai/ingest`, `/ai/embedding`, `/knowledge/stats`, `/knowledge/search`, `/knowledge/docs/{idx}`, `/node/info`, `/node/peers`, `/node/sync`, `/node/broadcast`, `/node/metrics`, `/node/chain`, `/node/chain/mine`, `/node/chain/verify`, `/security/sign-verify`, `/node/config`, `/node/dependencies`, `/node/logs`, `/ui`) and all 4 existing CLI commands (`start`, `join`, `node-status`, `train-model`) remain operational.
2. **Additive Versioned Architecture (`/api/v2/*`):** All new GajahMada 2.0 capabilities are exposed under `/api/v2/*` while existing v1 routes are upgraded internally to leverage hardened cryptography, deduplicated knowledge storage, and honest telemetry.
3. **Database & Storage Migration Strategy:**
   - **Primary Production Store:** PostgreSQL with `pgvector` (`storage/postgres.py` + `storage/migrations/001_initial_v2_schema.sql`).
   - **Standalone / Offline Local Store:** When `DATABASE_URL` (PostgreSQL) is not configured or the Postgres daemon is stopped, `storage/postgres.py` automatically uses a relational SQLite database (`knowledge_store/gajahmada_v2.db`) with transactional ACID guarantees, full schema parity, and vector similarity search, plus automatic migration of existing `knowledge_store/docs.json` and `knowledge_store/chain.json`.

---

## 2. Backward Compatibility Contract

| Existing Interface | Compatibility Guarantee | V2 Enhancement Under the Hood |
| :--- | :--- | :--- |
| `GET /`, `GET /api/status`, `GET /health` | Exact JSON keys preserved (`status`, `message`, `mode`, `ml_available`, `uptime_sec`, `ui_url`). | Adds `capabilities` object detailing `neural_inference_active`, `symbolic_reasoning_active`, `active_backend`, and `storage_engine`. |
| `POST /ai/chat` | Preserves `response`, `time_taken`, `block_index`, `task_hash`, `signature`, `rag_matches`. | Routes through the Cognitive Reasoning Pipeline, attaches `reasoning_id`, `verification_status`, `confidence`, and `capabilities_used`, and records a signed `TASK_COMPLETED` event on the ledger. |
| `POST /ai/ingest`, `GET /knowledge/stats`, `POST /knowledge/search` | Preserves v1 request/response schemas. | Syncs ingested records into Layer 4 Semantic Memory with SHA-256 content integrity hash, provenance metadata, and Rebuild-on-Delete vector synchronization. |
| `GET /node/chain`, `POST /node/chain/mine`, `POST /node/chain/verify` | Preserves `chain`, `valid`, `height`, `difficulty`, `verified_signatures`. | `is_chain_valid()` now verifies Ed25519 signatures for every signed block; blocks include typed `event_type` and `payload_hash`. |
| `POST /node/sync`, `POST /node/broadcast` | Preserves v1 request/response keys. | Enforces SSRF protection on peer URLs; `/node/broadcast` sends signed GIP messages to `/api/v2/network/gip` (falling back gracefully if peer is v1). |
| `POST /node/config`, `DELETE /node/logs`, `DELETE /knowledge/docs/{idx}` | Preserves v1 response schema. | Adds administrative authentication check when `GAJAHMADA_REQUIRE_AUTH=1` (and always on `/api/v2/*` admin endpoints). Passes updated `max_tokens` and `temperature` to inference engine. |
| Cyberdeck Web UI (`/ui`) | All 8 panels and 4 modals remain 100% functional. | Displays real v2 telemetry (active inference backend, honest neural status, memory tier counts, reasoning confidence, GIP protocol status, and security audit events). |

---

## 3. Sequential Implementation Phases

### PHASE 0: Repository Audit, Security Hardening & Storage/Migration Foundation
1. Create `docs/AUDIT_V2.md`, `docs/SECURITY_AUDIT.md`, `docs/ARCHITECTURE_V2.md`, `docs/MIGRATION_PLAN.md`, `docs/ROADMAP.md`.
2. Fix `NodeIdentity` in `gajahmada_ai/security/crypto.py`: remove hardcoded deterministic default seed, add key rotation, revocation list, and public key verification helpers.
3. Implement `gajahmada_ai/security/ssrf.py`, `gajahmada_ai/security/auth.py`, `gajahmada_ai/security/policies.py`, `gajahmada_ai/security/rate_limit.py`, and `gajahmada_ai/security/audit.py`.
4. Harden `AIBlockchain` (`gajahmada_ai/blockchain/chain.py`) to support typed audit ledger events (`NODE_REGISTERED`, `TASK_CREATED`, `TASK_ACCEPTED`, `TASK_COMPLETED`, `EVIDENCE_SUBMITTED`, `RESULT_VERIFIED`, `KNOWLEDGE_PROPOSED`, `KNOWLEDGE_ACCEPTED`, `KNOWLEDGE_REJECTED`, `MODEL_VERSION_CHANGED`) and verify Ed25519 signatures inside `is_chain_valid()`.
5. Fix `indonesia_corpus.py` dynamic timestamp duplication bug and clean duplicate entries on load.

### PHASE 1: Inference Abstraction & Native Hybrid Cognitive Reasoning
1. Implement `gajahmada_ai/gnnm/` (from-scratch **GNNM** Transformer & BPE tokenizer) and `gajahmada_ai/neural/` (`interfaces.py`, `model_registry.py`, `pytorch_backend.py`, `onnx_backend.py`, `router.py`).
2. Ensure honest capability reporting: when no `GNNM` (`./models/gnnm-*`) or `.onnx` model is loaded, `neural_inference_active` is explicitly `False` and `active_backend` is `"none (symbolic+evidence-only)"`.
3. Implement `gajahmada_ai/reasoning/` (`symbolic.py`, `constraints.py`, `evidence.py`, `contradictions.py`, `inference.py`, `evaluator.py`).
4. Implement `gajahmada_ai/cognition/` (`models.py`, `perception.py`, `intent.py`, `decomposition.py`, `hypotheses.py`, `metacognition.py`, `decision.py`, `reasoning.py`).

### PHASE 2: Persistent 5-Tier Memory & Knowledge Provenance
1. Implement `gajahmada_ai/storage/` (`interfaces.py`, `postgres.py`, `vector_store.py`, `object_store.py`, `migrations/001_initial_v2_schema.sql`).
2. Implement `gajahmada_ai/memory/` (`models.py`, `working.py`, `semantic.py`, `episodic.py`, `procedural.py`, `collective.py`, `knowledge_graph.py`, `provenance.py`, `retrieval.py`, `versioning.py`, `consolidation.py`, `manager.py`).
3. Ensure contradictions are detected, preserved, and linked rather than overwritten.

### PHASE 3: Autonomous Agent Runtime, Planning & Controlled Execution
1. Implement `gajahmada_ai/execution/sandbox.py` with strict tool allowlists, resource budgets, and human approval requirements for sensitive operations.
2. Implement `gajahmada_ai/agents/` (`base.py`, `planner.py`, `researcher.py`, `analyst.py`, `verifier.py`, `scheduler.py`) with explicit states (`CREATED`, `PLANNED`, `APPROVED`, `RUNNING`, `WAITING`, `COMPLETED`, `FAILED`, `CANCELLED`).

### PHASE 4: GajahMada Intelligence Protocol (GIP) & Multi-Node Collective Intelligence
1. Implement `gajahmada_ai/network/` (`protocol.py`, `transport.py`, `discovery.py`, `coordinator.py`, `synchronization.py`, `consensus.py`).
2. Implement all 8 GIP namespaces (`GIP-IDENTITY`, `GIP-DISCOVERY`, `GIP-CAPABILITY`, `GIP-TASK`, `GIP-KNOWLEDGE`, `GIP-VERIFY`, `GIP-CONSENSUS`, `GIP-SYNC`) with Ed25519 signature verification and nonce replay protection.
3. Implement 3+ node distributed task decomposition, delegation, independent verification, and evidence-weighted synthesis.

### PHASE 5: Controlled Learning Subsystem, API v2, CLI v2 & Acceptance Test Suite
1. Implement `gajahmada_ai/learning/` (`feedback.py`, `experience.py`, `evaluator.py`, `adaptation.py`, `experiments.py`, `datasets.py`) across Levels 1, 2, and 3 with regression testing and rollback.
2. Expose all `/api/v2/*` endpoints (`gajahmada_ai/api/v2_routes.py`) and v2 CLI commands (`gajahmada_ai/cli/main.py`).
3. Evolve Cyberdeck UI (`interface/html/`) to display real v2 cognitive, memory, agent, network, and security telemetry.
4. Execute all 15 mandatory acceptance tests and record real benchmarks.
