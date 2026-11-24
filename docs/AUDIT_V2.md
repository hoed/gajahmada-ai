# GAJAHMADA 2.0 — Comprehensive Repository Audit (`AUDIT_V2.md`)

**Audit Date:** 2026-11-22  
**Target Repository:** `/home/hoed/gajahmada`  
**Baseline Commit:** `3417a08` (`main`)  
**Runtime Environment:** Linux, Python 3.14.10, FastAPI 0.143.3, Pydantic 2.14.1, Cryptography 49.0.0, HTTPX 0.28.1, SQLAlchemy 2.0.52, Psycopg 3.3.5, Pytest 9.1.1.

---

## 1. Executive Summary

An exhaustive file-by-file, endpoint-by-endpoint, and dependency-by-dependency audit of the existing GajahMada repository was conducted prior to any source modification.

### Key Findings
1. **Working Baseline Exists:** All 20+ existing FastAPI routes (`/`, `/api/status`, `/health`, `/ai/chat`, `/ai/ingest`, `/ai/embedding`, `/knowledge/stats`, `/knowledge/search`, `/knowledge/docs/{idx}`, `/node/info`, `/node/peers`, `/node/sync`, `/node/broadcast`, `/node/metrics`, `/node/chain`, `/node/chain/mine`, `/node/chain/verify`, `/security/sign-verify`, `/node/config`, `/node/dependencies`, `/node/logs`, `/ui`) return HTTP 200 and drive an interactive 8-panel Cyberdeck HUD (`interface/html/`).
2. **Discrepancy Between Claims and Reality:**
   - **Neural Inference Claim (`README.md`):** Legacy v1 claimed a "Dual-Mode Neural Inference (`AIModelEngine`)" with a "Built-in Neural Synthesis Engine". In reality (`gajahmada_ai/core/engine.py`), when local native `GNNM` weights were absent, `_synthesize_response()` fell back to a regex/keyword rule table returning hardcoded Indonesian history paragraphs while reporting `ml_available = True` and `ml_ready: True`. (Remediated in v2.0 with **GNNM** & honest `neural_inference_active` reporting.)
   - **384-d Semantic Vector RAG Claim:** When `sentence-transformers` is absent, `create_embedding()` projects SHA-256 token hashes into a 384-d vector (`engine.py:343-353`). Furthermore, `indonesia_corpus.py` embeds dynamic `datetime.now()` strings into 3 corpus documents, causing `ensure_initialized()` (`server.py:168-175`) to re-ingest duplicate timestamped documents on every server restart.
   - **Decentralized P2P & Consensus Claim:** `POST /node/broadcast` (`server.py:507`) sends a `POST` request to `f"{peer}/health"`, which only defines `GET /health` (resulting in HTTP 405 Method Not Allowed on real peers) and performs zero signature verification or state mutation on the receiver.
   - **Test Coverage:** Running `pytest` collected **0 items**. There are no automated tests in the repository.

---

## 2. Classification of Existing Functionality

Every component is classified strictly into one of six categories:
`WORKING` | `PARTIALLY WORKING` | `PLACEHOLDER` | `BROKEN` | `UNSAFE` | `NOT IMPLEMENTED`

### 2.1 Directory & Module Classification Matrix

| Path / Module | Classification | Verified Evidence & Audit Notes |
| :--- | :---: | :--- |
| `main.py` | **WORKING** | Entrypoint importing `app` from `gajahmada_ai.api.server` and running Uvicorn on `0.0.0.0:8000`. |
| `config.yaml` | **WORKING** | Defines `node` (`host`, `port`, `name`), `network.bootstrap_peers`, and `ai` (`model_path`, `embedding_model`). |
| `gajahmada_ai/api/server.py` | **PARTIALLY WORKING / UNSAFE** | All REST endpoints execute and return JSON. However, `ML_AVAILABLE = True` is hardcoded; mutating admin endpoints lack authentication; `ensure_initialized()` duplicates dynamic corpus docs on restart; `/node/broadcast` POSTs to `/health`. |
| `gajahmada_ai/core/engine.py` | **PARTIALLY WORKING (v1) -> REMEDIATED (v2)** | `_safe_eval_math` (AST arithmetic evaluator) and 384-d hash-projection embedder work deterministically. Native **GNNM** loading works via `PyTorchNativeBackend` when `./models/gnnm-*` weights are present, with honest degraded capability reporting when weights are absent. |
| `gajahmada_ai/knowledge/rag.py` | **PARTIALLY WORKING** | Pure-Python L2 + lexical overlap search and `docs.json` persistence work. However, deleting a document (`server.py:445`) pops from `doc_store` and `vectors` without rebuilding the FAISS index (`self.index`), causing FAISS index desynchronization if FAISS is enabled. |
| `gajahmada_ai/knowledge/indonesia_corpus.py` | **PARTIALLY WORKING** | Returns 15 historical/modern Indonesian documents. Because 3 documents interpolate `datetime.now().strftime("%Y-%m-%d %H:%M:%S %Z")`, string equality checks in `server.py:169` fail on every restart, polluting `docs.json` with duplicates. |
| `gajahmada_ai/blockchain/chain.py` | **PARTIALLY WORKING** | SHA-256 Proof-of-Work mining (`difficulty=2`), hash linkage, and JSON persistence (`chain.json`) work. However, `is_chain_valid()` does **not** verify Ed25519 signatures, and blocks only store opaque `task_hash` strings rather than typed, verifiable audit events. |
| `gajahmada_ai/security/crypto.py` | **PARTIALLY WORKING / UNSAFE** | Ed25519 RFC 8032 key generation, PEM persistence (`0600`), signing, and verification work. **CRITICAL VULNERABILITY:** Line 46 uses a hardcoded deterministic seed `"gajahmada-sovereign-ai-vercel-node-v2"` whenever `VERCEL == "1"`, allowing anyone to derive the private key and forge signatures. |
| `gajahmada_ai/network/p2p.py` | **PARTIALLY WORKING / UNSAFE** | Async `httpx` client and self-URL filtering work. **UNSAFE:** `add_peer()` only checks `urlparse().scheme in ("http", "https")`, permitting SSRF against cloud metadata (`169.254.169.254`), internal loopback services, or arbitrary ports. **BROKEN:** No receiving gossip/protocol endpoint exists. |
| `gajahmada_ai/cli/main.py` | **PARTIALLY WORKING** | `start`, `join`, `node-status`, and `train-model` commands work via Click. However, `train-model` does not train any model—it merely calls `/node/chain/mine` and `/node/broadcast`. |
| `gajahmada_ai/models/__init__.py` | **PLACEHOLDER** | Empty 0-byte file. No data models or neural model registry implemented. |
| `gajahmada_ai/plugins/__init__.py` | **PLACEHOLDER** | Empty 0-byte file. No plugin architecture implemented. |
| `interface/html/index.html` | **WORKING** | 1334-line Cyberdeck HUD with 8 panels and 4 modals. |
| `interface/html/app.js` | **WORKING** | 2133-line frontend controller binding all 8 panels to backend APIs, with client-side fallback when offline. |
| `interface/html/style.css` | **WORKING** | 1900-line responsive Bento Grid stylesheet. |
| `interface/html/analytics.js` | **WORKING** | Optional Vercel Web Analytics queue helper. |
| `tests/` | **NOT IMPLEMENTED** | No unit, integration, security, cognition, memory, or network tests exist. |

---

## 3. Detailed Functional & Architectural Audit

### 3.1 API Endpoints Baseline Inventory

All existing endpoints were tested with `fastapi.testclient.TestClient`:

| Method & Route | Status | Baseline Response Keys | Issues Identified |
| :--- | :---: | :--- | :--- |
| `GET /` | **WORKING** | `status`, `message`, `mode`, `ml_available`, `uptime_sec`, `ui_url` (or HTML UI) | `ml_available` is always `True` (`engine is not None`), even when no neural model is loaded. |
| `GET /api/status` | **WORKING** | `status`, `message`, `mode`, `ml_available`, `uptime_sec`, `system_time`, `ui_url` | Same misleading `ml_available: True`. |
| `GET /health` | **WORKING** | `ok`, `mode`, `uptime_sec` | Does not report active inference backend or storage health. |
| `POST /ai/chat` | **PARTIALLY WORKING** | `response`, `time_taken`, `block_index`, `task_hash`, `signature`, `rag_matches` | Uses keyword fallback without reporting inference mode; `distribute: bool` field in `ChatRequest` (`server.py:183`) is completely ignored (dead parameter). |
| `POST /ai/ingest` | **PARTIALLY WORKING** | `status`, `tokens`, `total_docs`, `dimension` | Unauthenticated; no provenance, confidence, or deduplication. |
| `POST /ai/embedding` | **WORKING** | `embedding`, `dimension` | Uses hash-projection when `sentence-transformers` is absent without reporting embedding backend type. |
| `GET /knowledge/stats` | **WORKING** | `total_docs`, `dimension`, `engine`, `docs` | None. |
| `POST /knowledge/search` | **WORKING** | `query`, `results`, `count` | Lacks provenance, confidence scores, and memory type filtering. |
| `DELETE /knowledge/docs/{doc_idx}` | **PARTIALLY WORKING / UNSAFE** | `status`, `total_docs` | Unauthenticated destructive action; desynchronizes FAISS index if FAISS is active. |
| `GET /node/info` | **WORKING** | `node_id`, `node_name`, `peers`, `blockchain_height`, `blockchain_valid`, `pow_difficulty`, `knowledge_docs`, `mode`, `ml_ready`, `uptime_sec`, `system_time` | `ml_ready` is always `True` even without a loaded neural model. |
| `GET /node/peers` | **WORKING** | `peers` | Lists raw URL strings without node identity, capabilities, or trust status. |
| `POST /node/sync` | **UNSAFE** | `status`, `added`, `peers_known` | Unauthenticated; vulnerable to SSRF via arbitrary peer URLs. |
| `DELETE /node/peers` | **UNSAFE** | `status`, `peers_known` | Unauthenticated peer removal. |
| `POST /node/broadcast` | **BROKEN / UNSAFE** | `status`, `peers_targeted`, `peers_reachable`, `signature` | Broadcasts `POST /health` to peers (`server.py:507`), which fails with `405 Method Not Allowed` on actual GajahMada peers. |
| `GET /node/metrics` | **WORKING** | `cpu_usage`, `memory_usage`, `disk_read_mb`, `knowledge_docs`, `chain_height`, `pow_difficulty`, `peers_count`, `time` | Falls back to hardcoded `12, 38, 0` if `psutil` fails (`server.py:528`). |
| `GET /node/chain` | **WORKING** | `chain`, `valid`, `height`, `difficulty` | None. |
| `POST /node/chain/mine` | **PARTIALLY WORKING / UNSAFE** | `status`, `block`, `time_taken` | Unauthenticated; synchronous PoW loop on main thread can block event loop if difficulty is raised. |
| `POST /node/chain/verify` | **WORKING** | `valid`, `height`, `difficulty`, `verified_signatures` | Verifies every block signature against `identity.node_id` (`server.py:580`) instead of `b.node_id`'s public key, which works for local blocks (`b.node_id` is base64url public key) because `verify_message` decodes the third argument `b.node_id`. |
| `POST /security/sign-verify` | **WORKING** | `node_id`, `message`, `signature`, `verified`, `algorithm` | Good self-test endpoint. |
| `GET /node/config` | **WORKING** | `node`, `network`, `ai`, `blockchain` | None. |
| `POST /node/config` | **UNSAFE** | `status`, `config` | Unauthenticated administrative mutation of runtime configuration. Furthermore, `temperature` and `max_tokens` in `runtime_config["ai"]` are never passed to `AIModelEngine.generate()`! |
| `GET /node/dependencies` | **PARTIALLY WORKING (v1) -> REMEDIATED (v2)** | `python_version`, `packages`, `endpoints_count` | Legacy v1 reported missing ML packages (`torch`, `faiss-cpu`, `sentence-transformers`) as `"builtin-engine"`, obscuring whether optional ML libraries were installed. Now reports exact status honestly. |
| `GET /node/logs` | **WORKING** | `logs` | Returns last 50 log entries. |
| `DELETE /node/logs` | **UNSAFE** | `status`, `logs` | Unauthenticated log deletion (allows an attacker to wipe audit logs). |

### 3.2 Dead Code, Broken Imports & Duplicated Implementations

1. **Unused / Ignored Parameters & Configs:**
   - `ChatRequest.distribute: bool = False` (`server.py:183`) is accepted by `/ai/chat` but never read.
   - `runtime_config["ai"]["temperature"]` and `runtime_config["ai"]["max_tokens"]` (`server.py:88-89`, updated via `POST /node/config`) are never passed to `engine.generate()`.
2. **Duplicate Stopwords & Keyword Extractors:**
   - `_STOPWORDS` and keyword extraction are duplicated across `gajahmada_ai/core/engine.py` (lines 13-29) and `gajahmada_ai/knowledge/rag.py` (lines 7-24).
3. **Duplicate LLM Synthesis Logic in Frontend & Backend:**
   - `interface/html/app.js` (`synthesizeLocalLLMResponse`, lines 1191-1263) duplicates the backend's `_synthesize_response` (`engine.py:129-336`).
4. **Empty Placeholder Packages:**
   - `gajahmada_ai/models/__init__.py` (0 bytes)
   - `gajahmada_ai/plugins/__init__.py` (0 bytes)
5. **Corpus Timestamp Duplication Bug:**
   - `gajahmada_ai/knowledge/indonesia_corpus.py` generates dynamic timestamps inside 3 corpus paragraphs at import/call time. In `knowledge_store/docs.json`, this has already caused the document count to bloat from 21 unique documents to 29 documents with repeated timestamp variations (`2026-11-21 22:55:44 WIB`, `23:07:49 WIB`, `23:13:38 WIB`).

### 3.3 Dependency Assessment

| Package | Specified in `requirements.txt` | Installed in Environment | Audit Assessment |
| :--- | :---: | :---: | :--- |
| `fastapi` | `>=0.103.0` | `0.143.3` | **WORKING** — Core REST API framework. |
| `uvicorn` | `>=0.23.2` | `0.53.0` | **WORKING** — ASGI server. |
| `pydantic` | `>=2.3.0` | `2.14.1` | **WORKING** — Schema validation and structured models. |
| `cryptography` | `>=41.0.3` | `49.0.0` | **WORKING** — Ed25519 (RFC 8032) and cryptographic primitives. |
| `httpx` | `>=0.25.0` | `0.28.1` | **WORKING** — Async HTTP client for P2P and TestClient. |
| `click` | `>=8.1.7` | `8.3.3` | **WORKING** — CLI framework. |
| `PyYAML` | `>=6.0.1` | `6.0.3` | **WORKING** — Configuration parser. |
| `psutil` | `>=5.9.5` | `7.2.2` | **WORKING** — Real hardware/process telemetry. |
| `numpy` | `>=1.24.0` (`-local`) | `2.4.6` | **WORKING** — Numerical and vector operations. |
| `sqlalchemy` | *(New for V2)* | `2.0.52` | **AVAILABLE** — Relational ORM/Core for PostgreSQL & SQLite persistence. |
| `psycopg` / `psycopg2` | *(New for V2)* | `3.3.5` / `2.9.13` | **AVAILABLE** — PostgreSQL drivers installed. (Local `postgres` daemon is currently stopped; storage layer must support PostgreSQL + pgvector with seamless SQLite/JSON local durable fallback when Postgres daemon is offline). |
| `torch` (for **GNNM**) | `>=2.0.0` (`.venv312`) | Installed in `.venv312` | **OPTIONAL** — Abstracted behind `PyTorchNativeBackend` & `gajahmada_ai/gnnm/` with honest availability reporting (`available: False` when package or `./models/gnnm-*` checkpoint is absent). |
| `onnxruntime` | *(New for V2)* | Not installed | **OPTIONAL** — Abstracted behind `OnnxBackend` with explicit availability detection. |
| `faiss-cpu` | `>=1.7.4` (`-local`) | Not installed | **OPTIONAL** — Pure-Python/NumPy L2 & cosine vector search works natively. |
| `sentence-transformers` | `>=2.2.2` (`-local`) | Not installed | **OPTIONAL** — Must explicitly report whether embeddings come from a neural transformer or deterministic lexical-hash projection. |

---

## 4. Target Architecture Gap Summary

| GajahMada 2.0 Requirement | Current Repository State | Required Evolution |
| :--- | :--- | :--- |
| **Layer 1: Sovereign Foundation** | Partial (`NodeIdentity`, `config.yaml`, `system_logs` in-memory) | Eliminate deterministic default seed; add key rotation/revocation, persistent audit events, runtime lifecycle & honest capability diagnostics. |
| **Layer 2: Distributed Infrastructure** | Broken/Partial (`P2PNetwork` broadcasts to `/health`, no peer auth) | Implement GIP v2.0 signed protocol (`GIP-IDENTITY`, `GIP-DISCOVERY`, `GIP-CAPABILITY`, `GIP-TASK`, `GIP-KNOWLEDGE`, `GIP-VERIFY`, `GIP-CONSENSUS`, `GIP-SYNC`), replay protection, SSRF protection, and real multi-node HTTP transport. |
| **Layer 3: Perception & Knowledge** | Basic (`extract_keywords`, raw string list in `rag.py`) | Add document normalization, entity extraction, claim classification, evidence collection, and source provenance records. |
| **Layer 4: Memory & Knowledge Engine** | Single flat list (`docs.json`) | Implement 5 memory types (`working`, `semantic`, `episodic`, `procedural`, `collective`), versioning, contradiction tracking, confidence metadata, integrity hashes, PostgreSQL/pgvector + durable SQLite storage, and backup/restore. |
| **Layer 5: Hybrid Reasoning Engine** | Hardcoded regex synthesis + basic AST math | Implement sovereign **GNNM** (`gajahmada_ai/gnnm/`) + provider-independent `InferenceBackend` (`pytorch_backend`, `onnx_backend`, `model_registry`, `router`) + `SymbolicReasoner` (logic, constraints, math) + `EvidenceReasoner` (contradictions, provenance) + `MetacognitiveCritic` producing structured `ReasoningResult`. |
| **Layer 6: Autonomous Agent Runtime** | Not implemented | Implement `PlannerAgent`, `ResearcherAgent`, `AnalystAgent`, `VerifierAgent`, `TaskScheduler` with explicit states (`CREATED`..`CANCELLED`), budgets, sandboxing, and human approval gates. |
| **Layer 7: Collective Intelligence** | Not implemented | Implement multi-node task negotiation, delegation, independent verification, conflict resolution, and quorum acknowledgement without trusting majority blindly. |
| **Layer 8: Intelligence Governance** | Not implemented | Implement RBAC/API key authentication, execution policies, knowledge trust policies, model version control, and learning experiment approval/rollback. |
