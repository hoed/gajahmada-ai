# GajahMada — System & Web3 Architecture

## 1. Sovereign Decentralized Node
**GajahMada** is a decentralized peer-to-peer (P2P) AI and Web3 audit network node. Each node executes inference, 384-d semantic vector retrieval (RAG), Ed25519 payload signing, and SHA-256 Proof-of-Work ledger commits on both **Localhost** and **Vercel Serverless**.

## 2. Core Modules (`gajahmada_ai/`)

| Module | File | Responsibility |
| :--- | :--- | :--- |
| **API Gateway** | `gajahmada_ai/api/server.py` | FastAPI REST & Web3 endpoints (16 routes) + `/ui` Tactical HUD host |
| **Core Engine** | `gajahmada_ai/core/engine.py` | `AIModelEngine`: Local LLaMA GGUF + Built-in Neural Synthesis & 384-d Embedder |
| **Knowledge RAG** | `gajahmada_ai/knowledge/rag.py` | `LocalKnowledge`: 384-d FAISS `IndexFlatL2` & JSON-backed L2 Vector Store |
| **Web3 Blockchain** | `gajahmada_ai/blockchain/chain.py` | `AIBlockchain`: Persistent SHA-256 Proof-of-Work ledger (`chain.json`) |
| **P2P Network** | `gajahmada_ai/network/p2p.py` | `P2PNetwork`: Async `httpx` mesh topology, peer sync, & signed broadcast |
| **Security & Crypto** | `gajahmada_ai/security/crypto.py` | `NodeIdentity`: Ed25519 (RFC 8032) keypair, deterministic serverless seed, & verifier |
| **CLI Interface** | `gajahmada_ai/cli/main.py` | `gajahmada` CLI (`start`, `join`, `node-status`, `train-model`) |
| **Tactical Web UI** | `interface/html/` | 8-panel 100vh Cyberdeck HUD (`index.html`, `style.css`, `app.js`, `logo.png`) |

## 3. REST & Web3 API Endpoints
- `GET /` & `GET /health` — Node status, mode (`full` / `serverless`), and uptime
- `POST /ai/chat` — RAG-augmented inference + Ed25519 signature + SHA-256 PoW block mining
- `POST /ai/ingest` & `POST /ai/embedding` — 384-d vector encoding and RAG document assimilation
- `GET /knowledge/stats`, `POST /knowledge/search`, `DELETE /knowledge/docs/{idx}` — Vector store management
- `GET /node/chain`, `POST /node/chain/mine`, `POST /node/chain/verify` — Web3 PoW ledger & signature audit
- `POST /security/sign-verify` — Ed25519 cryptographic self-test
- `GET /node/peers`, `POST /node/sync`, `DELETE /node/peers`, `POST /node/broadcast` — P2P mesh operations
- `GET /node/info`, `GET /node/metrics`, `GET /node/config`, `POST /node/config`, `GET /node/dependencies`, `GET /node/logs`, `DELETE /node/logs` — System telemetry & configuration
