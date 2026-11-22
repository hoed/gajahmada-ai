# GAJAHMADA 2.0 — Security Vulnerability Assessment (`SECURITY_AUDIT.md`)

**Audit Date:** 2026-11-22  
**Scope:** Full source code review of `gajahmada_ai/`, `main.py`, `config.yaml`, `vercel.json`, and `interface/html/`.

---

## 1. Identified Vulnerabilities in Existing Codebase

### VULN-01 (CRITICAL): Predictable Hardcoded Private Key Seed
- **Location:** `gajahmada_ai/security/crypto.py:45-48`
- **Code:**
  ```python
  if os.environ.get("VERCEL") == "1" or os.environ.get("GAJAHMADA_NODE_SEED"):
      seed_str = os.environ.get("GAJAHMADA_NODE_SEED", "gajahmada-sovereign-ai-vercel-node-v2")
      seed_bytes = hashlib.sha256(seed_str.encode("utf-8")).digest()
      self._private_key = ed25519.Ed25519PrivateKey.from_private_bytes(seed_bytes)
  ```
- **Risk:** Every deployment with `VERCEL=1` (and no explicit secret seed) derives the exact same Ed25519 private key from the public string `"gajahmada-sovereign-ai-vercel-node-v2"`. Any external party can compute this private key and forge valid Ed25519 signatures for blocks and P2P messages.
- **Remediation:** Remove the hardcoded fallback string completely. Always generate a cryptographically random Ed25519 keypair via `Ed25519PrivateKey.generate()` (persisting to `key_path` with `0600` permissions), and only allow seed derivation if an explicit, high-entropy `GAJAHMADA_NODE_SEED` environment variable (>= 32 characters) is deliberately provided by the operator. Support key rotation and revocation lists.

---

### VULN-02 (HIGH): Unauthenticated Administrative & Destructive Endpoints
- **Location:** `gajahmada_ai/api/server.py`
  - `POST /node/config` (line 617) — Allows any anonymous caller to modify node name, PoW difficulty, max tokens, and temperature.
  - `DELETE /knowledge/docs/{doc_idx}` (line 440) — Allows any anonymous caller to delete knowledge base documents.
  - `DELETE /node/logs` (line 665) — Allows any anonymous caller to wipe the system audit log buffer.
  - `POST /node/sync` (line 480) & `DELETE /node/peers` (line 494) — Allows any anonymous caller to inject or remove P2P peers.
- **Risk:** Unauthorized state tampering, knowledge destruction, and audit log erasure.
- **Remediation:**
  1. Introduce role- and capability-based authentication (`gajahmada_ai/security/auth.py` and `policies.py`) supporting `X-API-Key` / `Authorization: Bearer <token>` and Ed25519 signed administrative requests.
  2. Enforce authentication on all administrative and destructive endpoints (`/api/v2/*` admin actions, `POST /node/config`, `DELETE /knowledge/docs/{idx}`, `DELETE /node/logs`, peer mutation, agent execution approval) when security enforcement is active, and require admin credentials on `/api/v2/` protected routes by default.

---

### VULN-03 (HIGH): Server-Side Request Forgery (SSRF) in P2P Peer Registration & Broadcast
- **Location:** `gajahmada_ai/network/p2p.py:23-31` and `gajahmada_ai/api/server.py:480-515`
- **Code:**
  ```python
  parsed = urlparse(clean)
  if parsed.scheme not in ("http", "https") or not parsed.netloc:
      return False
  ```
- **Risk:** An attacker can register internal network targets such as `http://169.254.169.254/latest/meta-data` (cloud metadata service), `http://127.0.0.1:6379` (Redis), or `http://127.0.0.1:5432` (PostgreSQL) via `POST /node/sync` and trigger outbound HTTP requests via `POST /node/broadcast`.
- **Remediation:**
  1. Implement strict SSRF validation (`gajahmada_ai/security/ssrf.py`):
     - Block link-local / cloud metadata IP ranges (`169.254.0.0/16`, `fe80::/10`).
     - Block non-HTTP(S) schemes, credentials in URLs (`user:pass@host`), and dangerous internal ports (`22`, `25`, `3306`, `5432`, `6379`, `27017`).
     - Block private/loopback IP ranges (`127.0.0.0/8`, `10.0.0.0/8`, `172.16.0.0/12`, `192.168.0.0/16`, `::1`) in production mode unless `allow_local_peers=True` is explicitly configured for local multi-node testing on allowed ports.
     - Enforce peer allowlists and revocation lists.

---

### VULN-04 (HIGH): Missing Replay Protection & Peer Authentication in P2P Messages
- **Location:** `gajahmada_ai/api/server.py:502-515` & `gajahmada_ai/network/p2p.py:36-45`
- **Risk:**
  1. Broadcast payloads have no `message_id`, `nonce`, `protocol_version`, or expiration window. Captured signed messages can be replayed indefinitely.
  2. Receiving nodes do not authenticate or verify incoming P2P payloads (in fact, `/node/broadcast` sends `POST /health`).
  3. `AIBlockchain.is_chain_valid()` (`chain.py:90-103`) checks SHA-256 hashes but never verifies `Block.signature` against `Block.node_id`!
- **Remediation:**
  1. Implement the GajahMada Intelligence Protocol (GIP v2.0) with mandatory fields: `protocol_version`, `message_id`, `message_type`, `sender_node_id`, `recipient_node_id`, `timestamp`, `nonce`, `payload`, `signature`.
  2. Verify Ed25519 signature over canonical JSON payload, enforce timestamp skew bounds (e.g., $\pm 300$ seconds), and track seen `(sender_node_id, nonce)` pairs in a sliding replay cache.
  3. Update `AIBlockchain.is_chain_valid()` to cryptographically verify `Block.signature` against `Block.node_id` for all non-genesis blocks.

---

### VULN-05 (MEDIUM): Lack of Rate Limiting, Input Bounds, and Resource Limits
- **Location:** `gajahmada_ai/api/server.py`
- **Risk:**
  1. `EmbeddingRequest` (`POST /ai/ingest`, `POST /ai/embedding`) and `ChatRequest` (`POST /ai/chat`) accept arbitrarily megabyte-sized strings, allowing memory exhaustion.
  2. No rate limiting exists on block mining (`POST /node/chain/mine`) or inference endpoints.
- **Remediation:**
  1. Add Pydantic field length and numeric bounds validation on all API models.
  2. Add a token-bucket / sliding-window rate limiter middleware and dependency for sensitive endpoints.
  3. Enforce bounded execution budgets (`max_iterations`, `max_time_sec`, `max_memory_items`) and sandboxed tool execution in the Autonomous Agent layer.

---

### VULN-06 (MEDIUM): Silent Error Swallowing on Storage & Verification Failures
- **Location:** `gajahmada_ai/blockchain/chain.py:112-125`, `gajahmada_ai/knowledge/rag.py:70-71`
- **Risk:** `except Exception: pass` silently discards disk write failures or corrupted chain files, resetting `self.chain = []` without logging a security event.
- **Remediation:** Route all cryptographic, storage, authentication, and policy violations to an immutable security audit event log (`gajahmada_ai/security/audit.py`).

---

## 2. Security Controls Matrix for GajahMada 2.0

| Security Control | Implementation Module | Verification Test |
| :--- | :--- | :--- |
| **Non-Predictable Ed25519 Identity & Key Rotation** | `gajahmada_ai/security/crypto.py` | `tests/security/test_security.py::test_no_predictable_seed` & `test_key_rotation_and_revocation` |
| **Admin Authentication & RBAC Authorization** | `gajahmada_ai/security/auth.py` | `tests/security/test_security.py::test_unauthenticated_admin_rejected` (TEST 12) |
| **SSRF Prevention & Peer Allowlist/Revocation** | `gajahmada_ai/security/ssrf.py` | `tests/security/test_security.py::test_ssrf_prevention` |
| **GIP Message Signing & Replay Protection** | `gajahmada_ai/network/protocol.py` | `tests/network/test_gip_protocol.py::test_replay_attack_rejected` |
| **Malicious Peer Isolation (No RCE / Memory Tampering)** | `gajahmada_ai/network/coordinator.py` & `security/policies.py` | `tests/security/test_security.py::test_malicious_peer_rejected` (TEST 13) |
| **Sandboxed Tool Execution & Human Approval Gates** | `gajahmada_ai/execution/sandbox.py` | `tests/security/test_security.py::test_sandbox_blocks_unrestricted_access` |
| **Rate Limiting & Payload Size Bounds** | `gajahmada_ai/security/rate_limit.py` | `tests/security/test_security.py::test_rate_limiting_and_input_validation` |
