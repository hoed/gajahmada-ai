<div align="center">

<img src="interface/html/logo.png" alt="GajahMada Logo" width="130" />

# GajahMada AI Nusantara (v2.0)

### **Sovereign, Independent, Hybrid, Distributed Intelligence Architecture**

*14-Stage Hybrid Cognitive Pipeline · Provider-Independent Local Neural Inference (`llama.cpp` GGUF / ONNX / PyTorch) · Deterministic Symbolic & Live Specialist Resolvers · Autonomous Internet Knowledge Crawler · 5-Tier Persistent Memory (`SQLite` + Optional `Supabase PostgreSQL + pgvector`) · Autonomous Multi-Agent Runtime · GIP/2.0 Cryptographic Mesh Protocol · Ed25519 Signed Event Audit Ledger*

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-2dd4bf?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-REST_v1_%26_v2-009688?style=for-the-badge&logo=fastapi&logoColor=white)](gajahmada_ai/api/server.py)
[![Supabase pgvector](https://img.shields.io/badge/Supabase-pgvector_384d-3ecf8e?style=for-the-badge&logo=supabase&logoColor=white)](supabase/schema.sql)
[![Cognition](https://img.shields.io/badge/Cognition-14_Stage_Hybrid-818cf8?style=for-the-badge)](gajahmada_ai/cognition/reasoning.py)
[![Protocol](https://img.shields.io/badge/Protocol-GIP%2F2.0_Ed25519-f59e0b?style=for-the-badge)](gajahmada_ai/network/protocol.py)
[![Tests](https://img.shields.io/badge/Pytest-19%2F19_Passing-4ade80?style=for-the-badge)](tests/)

</div>

---

## 🖥️ Tactical Cyberdeck HUD Preview (`/ui`)

![GajahMada Tactical Cyberdeck HUD](interface/html/screenshot-v2.jpg)

> **Live Web Console (`/ui`):** Antarmuka *Bento Grid 100vh* bergaya *Tactical Cyberdeck* yang memvisualisasikan dan mengendalikan seluruh 8 lapisan arsitektur **GajahMada 2.0** secara real-time—mulai dari graf pengetahuan 3D, telemetri log sistem, konsol perintah penalaran hibrida & slash-commands, matriks dependensi, siklus konsensus Web3, protokol jaringan P2P `GIP/2.0`, hingga penambangan blok *Signed Event Audit Ledger*.

---

## ✨ Sorotan & Kapabilitas Utama GajahMada 2.0

1. **🧠 14-Stage Hybrid Cognitive Reasoning Pipeline (`gajahmada_ai.cognition` & `reasoning`):**
   - Mengorkestrasi tahapan `PERCEPTION` → `INTENT CLASSIFICATION` → `MEMORY RETRIEVAL` → `PROBLEM DECOMPOSITION` → `HYPOTHESIS GENERATION` → `PARALLEL REASONING (Neural + Symbolic + Evidence)` → `EVIDENCE FUSION` → `METACOGNITIVE CRITIC` → `SUFFICIENCY EVALUATION` → `DECISION`.
   - Menghasilkan objek terstruktur `ReasoningResult` dengan `EpistemicBreakdown` yang secara tegas memisahkan **fakta terverifikasi (`verified_facts`)**, **klaim belum terverifikasi (`unverified_claims`)**, **hipotesis (`hypotheses`)**, **inferensi (`inferences`)**, **opini (`opinions`)**, dan **informasi yang belum diketahui (`unknown_information`)**.
2. **🔌 Provider-Independent Local Neural Abstraction (`gajahmada_ai.neural`):**
   - Mendukung eksekusi model lokal melalui **`LlamaCppBackend` (GGUF `llama.cpp`)**, **`OnnxBackend` (ONNX Runtime)**, dan **`PyTorchNativeBackend`** dengan auto-discovery model `*.gguf` di `./models/` (mis. `qwen2.5-1.5b-instruct-q4_k_m.gguf`).
   - **Ringkas Secara Default:** Jawaban dirancang singkat dan langsung ke inti (1–3 kalimat), dan otomatis menjabarkan secara mendalam hanya jika pertanyaan meminta detail (`detail`, `lengkap`, `rinci`, `uraikan`, `step by step`).
   - **100% Bebas Ketergantungan Cloud AI Terpusat** & **Honest Capability Reporting:** Melaporkan secara jujur status `neural_inference_active` (`True` hanya ketika bobot model neural nyata dimuat).
3. **📐 Deterministic Symbolic, Live Specialists & Autonomous Internet Crawler (`gajahmada_ai.reasoning` & `knowledge.open_knowledge`):**
   - **Symbolic Reasoner:** Evaluasi ekspresi matematika berbasis AST (`sqrt`, `log`, `sin`, `cos`), deduksi logika *forward-chaining*, dan silogisme tanpa LLM.
   - **Autonomous Multi-Source Crawler:** Crawling pengetahuan lintas sumber (**Wikipedia ID/EN**, **DuckDuckGo**, **Open-Meteo Live Weather**, **Katalog 114 Surah & Ayat Al-Qur'an**, dan **Direct URL Crawl** berproteksi anti-SSRF) lengkap dengan daftar **Sumber & Referensi** di akhir jawaban serta asimilasi otomatis ke Brain setiap interaksi.
   - **Evidence & Contradiction Engine:** Pembobotan reliabilitas sumber (*provenance weighting*) dan deteksi kontradiksi otomatis.
4. **🗄️ 5-Tier Persistent Memory & Hybrid Supabase `pgvector` Store (`gajahmada_ai.memory` & `storage`):**
   - Mengimplementasikan 5 tingkat memori (**Working, Semantic, Episodic, Procedural, Collective**) dengan penyimpanan *local-first* **`SQLite` (`gajahmada_v2.db`)** + **384-d Vector Store (`LocalKnowledge`)**.
   - **Integrasi Opsional Supabase (`PostgreSQL + pgvector`):** Melalui `SupabaseKnowledgeStore` (`gajahmada_ai/storage/supabase_store.py`) dan skema `supabase/schema.sql`, node serverless (seperti Vercel) maupun node lokal dapat menyinkronkan memori kolektif ber-ID **`content_hash` (SHA-256)** + tanda tangan **Ed25519**, pencarian vektor `match_knowledge` 384-d, dan blok audit `ledger_blocks` secara persisten tanpa kehilangan independensi lokal.
5. **🤖 Autonomous Multi-Agent Runtime & Sandboxed Execution (`gajahmada_ai.agents` & `execution`):**
   - Empat agen spesialis dengan kapabilitas dan izin tool tersendiri: **`PlannerAgent`**, **`ResearcherAgent`**, **`AnalystAgent`**, dan **`VerifierAgent`** yang dikoordinasikan oleh **`AutonomousTaskScheduler`**.
   - Siklus status eksekusi eksplisit: `CREATED` → `PLANNED` → `APPROVED` → `RUNNING` → `WAITING` → `COMPLETED` / `FAILED` / `CANCELLED`, dilengkapi batas anggaran iterasi/waktu, *self-correction*, **`SandboxedExecutor`**, dan gerbang persetujuan manusia (*human approval gate*) untuk aksi sensitif.
6. **🌐 GajahMada Intelligence Protocol (`GIP/2.0`) & Collective Intelligence (`gajahmada_ai.network`):**
   - Protokol komunikasi terdistribusi berversi (`GIP/2.0`) dengan 8 namespace: `GIP-IDENTITY`, `GIP-DISCOVERY`, `GIP-CAPABILITY`, `GIP-TASK`, `GIP-KNOWLEDGE`, `GIP-VERIFY`, `GIP-CONSENSUS`, dan `GIP-SYNC`.
   - Dilengkapi tanda tangan digital **Ed25519**, proteksi *nonce replay* (*sliding window*), pelindung **SSRF**, delegasi tugas multi-node (`CollectiveCoordinator`), dan verifikasi kuorum berbasis independensi bukti (tidak mempercayai suara mayoritas buta tanpa bukti).
7. **⛓️ Signed Event Audit Ledger & Sovereign Security (`gajahmada_ai.blockchain` & `security`):**
   - Setiap node memiliki kunci privat **Ed25519 (RFC 8032)** independen (`0600`, tanpa *default seed* statis yang dapat ditebak), mendukung rotasi kunci bersertifikat ganda (`POST /api/v2/system/rotate-key`) dan pencabutan peer (*peer revocation*).
   - `AIBlockchain` mencatat event terstruktur (`NODE_REGISTERED`, `TASK_CREATED`, `TASK_ACCEPTED`, `TASK_COMPLETED`, `EVIDENCE_SUBMITTED`, `RESULT_VERIFIED`, `KNOWLEDGE_PROPOSED`, `KNOWLEDGE_ACCEPTED`, `KNOWLEDGE_REJECTED`, `MODEL_VERSION_CHANGED`) dengan Proof-of-Work SHA-256 dan verifikasi tanda tangan Ed25519.
8. **📈 Controlled 3-Level Closed-Loop Learning Subsystem (`gajahmada_ai.learning`):**
   - Mendukung Level 1 (*Validated Knowledge Acquisition*), Level 2 (*Closed-Loop Procedural Strategy Extraction & Dynamic `retrieval_top_k` Adaptation* yang terhubung langsung ke `POST /ai/chat`), dan Level 3 (*Local Model Experiment Governance*).
   - Setiap adaptasi strategi diuji terhadap *Regression Dataset Suite* dan otomatis ditolak (*rollback*) jika menurunkan akurasi.

---

## 🛡️ 7 Prinsip Arsitektur Kecerdasan Berdaulat (7/7 Terverifikasi)

| # | Prinsip Arsitektur | Implementasi & Verifikasi |
| :-: | :--- | :--- |
| **1** | **Cognitive engine independen per node** | Tiap `SovereignPeerNode` menjalankan `CognitiveReasoningPipeline` 14-tahap secara mandiri & offline. |
| **2** | **Persistent memory privat per node** | SQLite ACID lokal (`gajahmada_v2.db`) + isolasi `PRIVATE`/`LOCAL` (terproteksi di GIP/2.0 & Supabase RLS). |
| **3** | **Neural-symbolic reasoning aktif** | Kombinasi model GGUF lokal (`qwen2.5-1.5b-instruct-q4_k_m.gguf` via `.venv312` `llama-cpp-python`) + AST Symbolic + Evidence Reasoner. |
| **4** | **Strategi adaptif berbasis pengalaman** | `record_outcome_and_adapt()` otomatis mengekstrak strategi ke `PROCEDURAL` memory & menyetel `retrieval_top_k` dengan uji regresi + rollback. |
| **5** | **Kerja sama melalui protokol terbuka** | Protokol **GIP/2.0** (8 namespace, JSON kanonik, Ed25519, proteksi *nonce replay* & anti-SSRF). |
| **6** | **Verifikasi independen antar-node** | `QuorumConsensusManager` memverifikasi tanda tangan Ed25519 & `provenance_hash`, serta menolak mayoritas tanpa bukti primer. |
| **7** | **Tanpa pusat pengendali tunggal** | Tidak ada node master wajib; identitas kriptografis, kebijakan, dan penyimpanan dikuasai masing-masing node. |

---

## 🧩 8 Modul Antarmuka Interaktif (`/ui`)

Setiap panel pada dashboard [interface/html/index.html](interface/html/index.html) terhubung langsung ke fungsi nyata pada backend:

| Panel | Nama Modul | Fungsi Nyata & Interaksi |
| :---: | :--- | :--- |
| **`01`** | **System Architecture & Knowledge Expansion** | • **Graf 3D Interaktif:** Memetakan 7 klaster paket `gajahmada_ai.*` dalam ruang 3D; klik klaster untuk menguji modul.<br>• **Module Configuration (Kiri):** Klik `node.name`, `ai.max_tokens`, `ai.temperature`, atau `chain.pow_diff` untuk mengubah konfigurasi server secara *live* (`POST /node/config`).<br>• **Web3 Server Arch (Kanan) & 6 Kartu Bawah (`6/6`):** Klik kartu `1. API SERVER`, `2. AI CORE`, `3. KNOWLEDGE`, `4. WEB3 CHAIN`, `5. P2P NETWORK`, `6. SECURITY`, atau `TEST ALL` untuk menjalankan benchmark & diagnostik langsung. |
| **`02`** | **System Logs** | Menarik *ring-buffer* log telemetri asli dari server (`GET /node/logs`), dilengkapi tombol **`SYNC`**, **`EXPORT`** (unduh file `.log`), dan **`CLEAR`** (`DELETE /node/logs`). |
| **`03`** | **Command Interface (Hero Console)** | Konsol utama penalaran kognitif (`POST /ai/chat`) dengan *RAG context injection*, pencatatan blok PoW otomatis, bar tombol **`QUICK EXEC`**, mode layar penuh **`⛶ MAX`**, serta dukungan **Slash Commands** (`/status`, `/time`, `/ingest`, `/search`, `/mine`, `/verify`, `/peers`). |
| **`04`** | **Module Dependencies** | Matriks 3 kolom (`APP MODULES`, `DEPENDENCIES`, `REST / WEB3 API`). Klik sel mana saja untuk memeriksa versi paket Python yang terinstal (`GET /node/dependencies`) atau menguji latensi HTTP endpoint terkait. |
| **`05`** | **Web3 Server Arch (Consensus Cycle)** | Cincin donat 6 segmen (`ingest` → `embed` → `infer` → `sign` → `mine` → `gossip`). Klik cincin atau tombol **`▶ RUN CYCLE`** untuk mengeksekusi siklus verifikasi 6 tahap secara berurutan ke backend. |
| **`06`** | **Modular Architecture Matrix** | Grid 4×4 berisi 16 komponen sistem (`AIModelEngine`, `LocalKnowledge`, `AIBlockchain`, `P2PNetwork`, `NodeIdentity`, dll.) dengan kursor pemindai otomatis. Klik komponen mana pun untuk menjalankan *live diagnostic probe*. |
| **`07`** | **Network Status (P2P Mesh)** | Visualisasi topologi *Hub & Spoke* node aktif. Klik **`⚡ BROADCAST`** untuk mengirim paket `GIP/2.0` bertanda tangan Ed25519 ke seluruh peer (`POST /node/broadcast`) atau **`+ PEER`** untuk mengelola koneksi mesh. |
| **`08`** | **System Performance & Ledger** | Menampilkan 7 indikator metrik nyata (`CPU USAGE`, `RAM ALLOC`, `DISK I/O READ`, `RAG VECTORS`, `ED25519 KEY`, `POW DIFFICULTY`, `WEB3 LEDGER`), status mode (`FULL·NEURAL` / `FULL·SYM+EV`), tombol **`⛏ MINE BLOCK`**, dan audit **`✓ CHAIN VALID`**. |

---

## 🏗️ Arsitektur 8 Lapisan & Alur Penalaran Kognitif

```mermaid
flowchart TD
    UI["Tactical Cyberdeck UI (/ui) & CLI (gajahmada)"] <--> API["FastAPI Gateway<br/>/api/v2/* & Legacy v1 Routes"]
    API --> L8["LAYER 8 — INTELLIGENCE GOVERNANCE<br/>RBAC Auth · Execution Policies · Trust Policies · Learning Regression & Rollback"]
    L8 --> L7["LAYER 7 — COLLECTIVE INTELLIGENCE<br/>Multi-Node Task Negotiation · Delegation · Independent Verification · Quorum Consensus"]
    L7 --> L6["LAYER 6 — AUTONOMOUS AGENT RUNTIME<br/>Planner · Researcher · Analyst · Verifier · Scheduler · Sandboxed Execution (CREATED..COMPLETED)"]
    L6 --> L5["LAYER 5 — HYBRID REASONING ENGINE<br/>Neural Backend (llama.cpp / ONNX / PyTorch) + Symbolic Reasoner + Evidence Reasoner + Metacognitive Critic"]
    L5 --> L4["LAYER 4 — 5-TIER MEMORY & KNOWLEDGE ENGINE<br/>Working · Semantic · Episodic · Procedural · Collective · Contradiction Tracking · Versioning"]
    L4 --> L3["LAYER 3 — PERCEPTION & KNOWLEDGE INGESTION<br/>Input Normalization · Intent Classification · Entity Extraction · Source Provenance"]
    L3 --> L2["LAYER 2 — DISTRIBUTED INFRASTRUCTURE (GIP/2.0)<br/>Ed25519 Signed Envelopes · Nonce Replay Protection · SSRF Guard · Capability Discovery"]
    L2 --> L1["LAYER 1 — SOVEREIGN FOUNDATION<br/>Ed25519 Identity & Key Rotation · PostgreSQL+pgvector / SQLite Storage · Signed Event Ledger"]
```

### Struktur Direktori Proyek

```text
gajahmada/
├── main.py                              # Entry point utama (Localhost Uvicorn & Vercel ASGI)
├── config.yaml                          # Konfigurasi node, bootstrap peer, dan path model GGUF
├── .env.example                         # Template variabel lingkungan Supabase (salin ke .env, jangan commit .env)
├── vercel.json                          # Konfigurasi build & routing Vercel Serverless
├── supabase/
│   └── schema.sql                       # Skema PostgreSQL + pgvector (knowledge_records, ledger_blocks, match_knowledge RPC)
├── pytest.ini                           # Konfigurasi test suite Pytest
├── requirements.txt                     # Dependensi inti ringan (Vercel & Localhost)
├── requirements-local.txt               # Dependensi ML penuh opsional (llama-cpp, faiss, sentence-transformers)
├── setup.py                             # Instalasi paket Python & CLI script `gajahmada`
├── run.sh / run.bat                     # Script quick-start Linux/macOS & Windows
├── ARCHITECTURE.md                      # Ringkasan arsitektur sistem
├── docs/                                # Dokumentasi Audit & Arsitektur GajahMada 2.0
│   ├── AUDIT_V2.md                      # Audit lengkap kode sumber & klasifikasi fitur
│   ├── ARCHITECTURE_V2.md               # Spesifikasi 8 lapisan arsitektur & pipa kognitif 14-tahap
│   ├── SECURITY_AUDIT.md                # Audit kerentanan keamanan & matriks kontrol keamanan
│   ├── MIGRATION_PLAN.md                # Rencana migrasi & kontrak kompatibilitas v1 -> v2
│   └── ROADMAP.md                       # Roadmap fase implementasi & hasil benchmark nyata
├── gajahmada_ai/
│   ├── api/
│   │   ├── server.py                    # FastAPI server (v1 endpoints + mount /api/v2/* + /ui host)
│   │   └── v2_routes.py                 # Router /api/v2/{intelligence,reasoning,memory,agents,network,nodes,knowledge,system}
│   ├── core/
│   │   └── engine.py                    # AIModelEngine (Jembatan kompatibilitas v1 ke ModelRouter & SymbolicReasoner v2)
│   ├── cognition/                       # Pipa Penalaran Kognitif 14-Tahap
│   │   ├── models.py                    # ReasoningResult, EpistemicBreakdown, HypothesisItem, EvidenceItem
│   │   ├── perception.py                # Normalisasi input, deteksi bahasa, ekstraksi entitas & klaim
│   │   ├── intent.py                    # Klasifikasi maksud & domain penalaran
│   │   ├── decomposition.py             # Dekomposisi masalah menjadi sub-objektif terverifikasi
│   │   ├── hypotheses.py                # Generator asumsi & hipotesis
│   │   ├── metacognition.py             # Metacognitive Critic & evaluasi kecukupan bukti
│   │   ├── decision.py                  # Pemisahan epistemik (verified_facts, unverified_claims, hypotheses, inferences, opinions, unknowns)
│   │   └── reasoning.py                 # CognitiveReasoningPipeline utama
│   ├── neural/                          # Abstraksi Inferensi Neural Independen
│   │   ├── interfaces.py                # InferenceBackend ABC, ModelCapabilityMetadata, GenerationResult
│   │   ├── llama_backend.py             # LlamaCppBackend (GGUF) & deterministic 384-d embedder
│   │   ├── onnx_backend.py              # OnnxBackend (ONNX Runtime)
│   │   ├── pytorch_backend.py           # PyTorchNativeBackend
│   │   ├── model_registry.py            # ModelRegistry & version history
│   │   └── router.py                    # Task-based ModelRouter & pelaporan kapabilitas jujur
│   ├── reasoning/                       # Penalaran Simbolik, Batasan & Bukti
│   │   ├── symbolic.py                  # SymbolicReasoner (AST Math, Forward-Chaining Rules, Syllogisms)
│   │   ├── constraints.py               # ConstraintEvaluator
│   │   ├── contradictions.py            # ContradictionDetector (negasi, antonim, konflik numerik)
│   │   ├── evidence.py                  # EvidenceReasoner (bobot reliabilitas sumber & relevansi)
│   │   ├── inference.py                 # HybridInferenceCombiner
│   │   └── evaluator.py                 # ReasoningQualityEvaluator
│   ├── memory/                          # Mesin Memori Persisten 5-Tingkat
│   │   ├── models.py                    # MemoryRecord, MemoryType, SourceProvenance, VerificationStatus, VisibilityPolicy
│   │   ├── manager.py                   # MemoryManager terpadu (menegakkan 6 aturan memori GajahMada)
│   │   ├── working.py                   # Tier 1: WorkingMemory
│   │   ├── semantic.py                  # Tier 2: SemanticMemory
│   │   ├── episodic.py                  # Tier 3: EpisodicMemory
│   │   ├── procedural.py                # Tier 4: ProceduralMemory
│   │   ├── collective.py                # Tier 5: CollectiveMemory
│   │   ├── knowledge_graph.py           # KnowledgeGraphManager (relasi entitas)
│   │   ├── provenance.py                # ProvenanceValidator (verifikasi SHA-256 & izin berbagi)
│   │   ├── retrieval.py                 # HybridMemoryRetriever
│   │   ├── versioning.py                # MemoryVersionController (riwayat versi tak berubah)
│   │   └── consolidation.py             # MemoryConsolidator
│   ├── storage/                         # Lapisan Penyimpanan Relasional, Supabase pgvector & Objek
│   │   ├── interfaces.py                # StorageBackend ABC
│   │   ├── postgres.py                  # RelationalMemoryStore (PostgreSQL + pgvector & SQLite ACID fallback)
│   │   ├── supabase_store.py            # SupabaseKnowledgeStore (PostgREST + pgvector 384-d + ledger sync)
│   │   ├── vector_store.py              # SemanticVectorIndex (Cosine & L2 384-d)
│   │   ├── object_store.py              # ContentAddressedObjectStore (SHA-256 JSON snapshots)
│   │   └── migrations/001_initial_v2_schema.sql
│   ├── agents/                          # Runtime Agen Otonom (Layer 6)
│   │   ├── base.py                      # BaseAgent, AgentTask, ExecutionState (CREATED..CANCELLED), ExecutionBudget
│   │   ├── planner.py                   # PlannerAgent
│   │   ├── researcher.py                # ResearcherAgent
│   │   ├── analyst.py                   # AnalystAgent
│   │   ├── verifier.py                  # VerifierAgent
│   │   └── scheduler.py                 # AutonomousTaskScheduler
│   ├── execution/
│   │   └── sandbox.py                   # SandboxedExecutor dengan allowlist tool & human approval gate
│   ├── learning/                        # Pembelajaran Terkontrol 3-Level (Layer 8)
│   │   ├── feedback.py                  # FeedbackCollector
│   │   ├── experience.py                # ExperienceExtractor
│   │   ├── datasets.py                  # RegressionDatasetManager
│   │   ├── evaluator.py                 # LearningExperimentEvaluator
│   │   ├── adaptation.py                # StrategyAdapter dengan rollback instan
│   │   └── experiments.py               # LearningExperimentManager
│   ├── network/                         # Protokol GIP/2.0 & Kecerdasan Kolektif (Layer 2 & 7)
│   │   ├── protocol.py                  # GIPMessage (8 namespace) & GIPProtocolValidator (Replay Protection)
│   │   ├── transport.py                 # AuthenticatedHTTPTransport & TransportBackend
│   │   ├── discovery.py                 # PeerDiscoveryRegistry
│   │   ├── coordinator.py               # SovereignPeerNode & CollectiveCoordinator multi-node
│   │   ├── synchronization.py           # KnowledgeSynchronizer
│   │   ├── consensus.py                 # QuorumConsensusManager
│   │   └── p2p.py                       # P2PNetwork dengan proteksi anti-SSRF
│   ├── blockchain/
│   │   └── chain.py                     # AIBlockchain & Block (Signed Event PoW Audit Ledger)
│   ├── security/                        # Lapisan Keamanan Berdaulat
│   │   ├── crypto.py                    # NodeIdentity Ed25519, rotasi kunci, & pencabutan peer
│   │   ├── auth.py                      # AuthManager RBAC (Admin, Operator, Peer, Reader)
│   │   ├── policies.py                  # GovernancePolicy
│   │   ├── ssrf.py                      # Validator URL/IP/Port anti-SSRF
│   │   ├── rate_limit.py                # SlidingWindowRateLimiter
│   │   └── audit.py                     # SecurityAuditLogger berantai hash SHA-256
│   ├── knowledge/
│   │   ├── rag.py                       # LocalKnowledge (FAISS IndexFlatL2 + JSON Vector Store)
│   │   ├── open_knowledge.py            # Internet Crawler (Wikipedia, DuckDuckGo, Open-Meteo Weather, 114 Surah Al-Qur'an, Direct URL)
│   │   └── indonesia_corpus.py          # Korpus Sejarah Nusantara & Indonesia Modern
│   └── cli/
│       └── main.py                      # CLI commands v1 & v2
├── interface/html/                      # Tactical Cyberdeck Web UI (disajikan di /ui)
└── tests/                               # Suite Pengujian Otomatis (18 Test Suites / 15 Acceptance Criteria)
    ├── unit/test_core_and_neural.py
    ├── cognition/test_cognitive_pipeline.py
    ├── memory/test_memory_persistence.py
    ├── network/test_gip_and_collective.py
    ├── security/test_security_and_governance.py
    ├── integration/test_api_compatibility.py
    └── benchmarks/test_benchmarks.py
```

---

## 🚀 Instalasi & Panduan Menjalankan

### 1. Mode Cepat (Localhost — Berjalan Offline Tanpa GPU / Cloud API)

#### Linux / macOS
```bash
pip install -r requirements.txt
./run.sh
```

#### Windows
```bat
pip install -r requirements.txt
run.bat
```

Buka browser di: **[http://localhost:8000/ui](http://localhost:8000/ui)**

---

### 2. Mode Penuh dengan Model GGUF Lokal (`llama-cpp-python`)

Letakkan file model GGUF di `./models/` (mis. `qwen2.5-1.5b-instruct-q4_k_m.gguf`) dan pasang prebuilt CPU wheel `llama-cpp-python` di `.venv312` (cepat, tanpa kompilasi C++):

```bash
uv venv --python 3.12 .venv312
uv pip install --python .venv312/bin/python -r requirements.txt
uv pip install --python .venv312/bin/python llama-cpp-python \
  --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu

mkdir -p models
curl -L -o models/qwen2.5-1.5b-instruct-q4_k_m.gguf \
  https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct-GGUF/resolve/main/qwen2.5-1.5b-instruct-q4_k_m.gguf

# ./run.sh otomatis mendeteksi .venv312/bin/python dan memuat model GGUF dari config.yaml / ./models/
./run.sh
```

### 3. Integrasi Opsional Supabase (`PostgreSQL + pgvector`) untuk Memori Persisten & Vercel

Agar knowledge Brain dan audit ledger tersimpan permanen di cloud/Vercel sekaligus mendukung pencarian vektor `pgvector` 384-d:

1. Jalankan [supabase/schema.sql](supabase/schema.sql) di **Supabase SQL Editor** (atau via `psql`).
2. Salin `.env.example` menjadi `.env` (untuk lokal) atau tambahkan ke **Vercel → Project Settings → Environment Variables**:
   ```env
   SUPABASE_URL=https://<project-ref>.supabase.co
   SUPABASE_PUBLISHABLE_KEY=<your-publishable-key>
   SUPABASE_SECRET_KEY=<your-secret-key>
   ```
3. **Keamanan RLS (Row Level Security):** Skema `supabase/schema.sql` mengaktifkan RLS pada `knowledge_records` dan `ledger_blocks` — kunci publik (`SUPABASE_PUBLISHABLE_KEY`) hanya dapat membaca baris `visibility = 'COLLECTIVE'` dan **diblokir (`401 / 42501`)** untuk operasi `INSERT/UPDATE/DELETE`; hanya backend dengan `SUPABASE_SECRET_KEY` yang berhak menulis.
4. **Catatan `DATABASE_URL`:** Jika password PostgreSQL mengandung karakter khusus (seperti `@` atau `#`), gunakan URL-encoding (`@` → `%40`, `#` → `%23`).
5. Tanpa `.env`, GajahMada tetap berjalan 100% mandiri menggunakan SQLite lokal (*local-first*).

---

## 💻 Perintah CLI (`gajahmada`)

### Perintah CLI GajahMada 2.0
```bash
# Inisialisasi identitas Ed25519, database memori 5-tier, dan ledger
python -m gajahmada_ai.cli.main init

# Manajemen siklus hidup node
python -m gajahmada_ai.cli.main node init
python -m gajahmada_ai.cli.main node start --config config.yaml
python -m gajahmada_ai.cli.main node status
python -m gajahmada_ai.cli.main node stop

# Diagnostik kapabilitas & penalaran kognitif hibrida 14-tahap
python -m gajahmada_ai.cli.main intelligence diagnose
python -m gajahmada_ai.cli.main intelligence ask "Hitung sqrt(144) + 25 * 4 dan jelaskan verifikasinya"
python -m gajahmada_ai.cli.main reasoning inspect

# Inspeksi & pencadangan (backup) memori persisten 5-tingkat
python -m gajahmada_ai.cli.main memory inspect
python -m gajahmada_ai.cli.main memory backup

# Eksekusi agen otonom (Planner -> Researcher -> Analyst -> Verifier)
python -m gajahmada_ai.cli.main agent run "Investigasi Sumpah Palapa 1336 M dan warisan Majapahit"

# Protokol jaringan GIP/2.0 & pemeriksaan kesehatan sistem
python -m gajahmada_ai.cli.main network status
python -m gajahmada_ai.cli.main network peers
python -m gajahmada_ai.cli.main network discover
python -m gajahmada_ai.cli.main system health
```

### Perintah CLI v1 (Tetap Kompatibel 100%)
```bash
python -m gajahmada_ai.cli.main start
python -m gajahmada_ai.cli.main join http://127.0.0.1:8001
python -m gajahmada_ai.cli.main node-status
python -m gajahmada_ai.cli.main train-model --note "Audit Konsensus V2"
```

---

## 📡 Referensi REST & Web3 API (`/api/v2/*` & Legacy v1)

### 1. Endpoint GajahMada 2.0 (`/api/v2/*`)

| Metode | Endpoint | Deskripsi |
| :---: | :--- | :--- |
| `GET` | `/api/v2/intelligence/diagnose` | Melaporkan secara jujur status backend neural (`llama.cpp`, `onnx`, `gajahmada-native`), mesin simbolik, memori 5-tier, dan agen terdaftar. |
| `POST` | `/api/v2/intelligence/ask` | Menjalankan pipa penalaran kognitif 14-tahap dan mengembalikan `ReasoningResult` lengkap beserta `EpistemicBreakdown` & bukti. |
| `GET/POST` | `/api/v2/reasoning` | Mengambil riwayat jejak penalaran atau mengeksekusi penalaran baru. |
| `GET` | `/api/v2/reasoning/{reasoning_id}` | Menginspeksi detail langkah penalaran, hipotesis, bukti, dan kontradiksi untuk `reasoning_id` tertentu. |
| `GET` | `/api/v2/memory` | Menginspeksi statistik 5 tingkat memori, daftar rekaman publik/lokal, dan daftar kontradiksi yang terdeteksi. |
| `POST` | `/api/v2/memory` | *(Auth Required)* Menyimpan rekaman memori baru dengan hash integritas SHA-256, provenans, dan deteksi kontradiksi otomatis. |
| `PUT` | `/api/v2/memory/{record_id}` | *(Auth Required)* Memperbarui/mengoreksi memori sambil menyimpan versi sebelumnya di tabel audit `gm_memory_versions`. |
| `GET` | `/api/v2/memory/{record_id}/versions` | Mengambil riwayat lengkap seluruh versi rekaman memori. |
| `POST` | `/api/v2/memory/backup` | *(Auth Required)* Mengekspor *snapshot* memori ke *Content-Addressed Object Store* ber-hash SHA-256. |
| `GET` | `/api/v2/knowledge` | Mencari relasi entitas pada *Knowledge Graph* dan daftar kontradiksi pengetahuan. |
| `GET` | `/api/v2/agents` | Menampilkan daftar agen spesialis (`planner`, `researcher`, `analyst`, `verifier`) beserta izin tool dan daftar tugas. |
| `POST` | `/api/v2/agents/run` | Menjalankan objektif melalui runtime agen otonom (`CREATED` → `PLANNED` → `APPROVED` → `RUNNING` → `COMPLETED`). |
| `POST` | `/api/v2/agents/tasks/{task_id}/approve` | *(Auth Required)* Memberikan persetujuan operator manusia untuk tugas agen yang berstatus `WAITING`. |
| `GET` | `/api/v2/network` | Status protokol `GIP/2.0`, daftar URL peer, node yang ditemukan, dan daftar peer yang dicabut (*revoked*). |
| `POST` | `/api/v2/network/gip` | Endpoint penerima pesan protokol `GIP/2.0` bertanda tangan Ed25519 dengan validasi *nonce replay* & batas waktu. |
| `GET/POST` | `/api/v2/nodes` / `/api/v2/nodes/register` | Menampilkan identitas node atau mendaftarkan peer GIP baru dengan validasi anti-SSRF. |
| `GET` | `/api/v2/system/health` | Laporan kesehatan lengkap: kapabilitas kognitif, penyimpanan relasional, validitas ledger, log audit keamanan, & kebijakan. |
| `POST` | `/api/v2/system/rotate-key` | *(Auth Required)* Merotasi kunci privat Ed25519 node dan menerbitkan sertifikat rotasi bertanda tangan ganda. |
| `POST` | `/api/v2/system/learning/experiment` | *(Auth Required)* Mengajukan dan mengevaluasi eksperimen pembelajaran terhadap *regression suite* (otomatis menolak eksperimen yang gagal). |

### 2. Endpoint v1 (Tetap Kompatibel 100%)

| Metode | Endpoint | Deskripsi |
| :---: | :--- | :--- |
| `GET` | `/` & `/api/status` | Status utama node, mode (`full` / `serverless`), kapabilitas neural/simbolik, dan *uptime*. |
| `GET` | `/health` | Pemeriksaan kesehatan layanan (*health check*). |
| `POST` | `/ai/chat` | Eksekusi pertanyaan melalui 14-Stage Cognitive Pipeline dengan konteks RAG 384-d + penambangan blok PoW bertanda tangan Ed25519. |
| `POST` | `/ai/ingest` | Meng-embed teks dokumen ke vektor 384-d dan menyimpannya ke *Knowledge Store* serta *Layer 4 Semantic Memory*. |
| `POST` | `/ai/embedding` | Menghasilkan vektor embedding 384-dimensi yang dinormalisasi dari teks input. |
| `GET` | `/knowledge/stats` | Menampilkan jumlah dokumen RAG, dimensi vektor, tipe indeks, statistik 5-tier memori, dan pratinjau dokumen. |
| `POST` | `/knowledge/search` | Pencarian semantik *top-k* dokumen berdasarkan jarak vektor L2 384-d. |
| `POST` | `/knowledge/crawl` | Crawling pengetahuan langsung dari internet (Wikipedia, DuckDuckGo, Cuaca, URL) & asimilasi otomatis ke Brain. |
| `DELETE` | `/knowledge/docs/{idx}` | Menghapus dokumen tertentu dari *vector store* dan menyinkronkan ulang indeks. |
| `GET` | `/node/info` | Informasi lengkap node (`node_id` Ed25519, nama node, peer, tinggi blok, status `neural_inference_active`, memori). |
| `GET` | `/node/metrics` | Telemetri perangkat keras & sistem (`CPU %`, `RAM %`, `Disk Read MB`, status ledger). |
| `GET` | `/node/chain` | Mengambil seluruh daftar blok pada ledger Proof-of-Work beserta tipe event dan status validitasnya. |
| `POST` | `/node/chain/mine` | Menambang blok audit Proof-of-Work baru secara manual dengan tanda tangan Ed25519. |
| `POST` | `/node/chain/verify` | Mengaudit ulang seluruh hash SHA-256 PoW dan memverifikasi tanda tangan Ed25519 tiap blok. |
| `POST` | `/security/sign-verify` | Uji mandiri (*self-test*) pembuatan & verifikasi tanda tangan digital Ed25519 (RFC 8032). |
| `GET` | `/node/peers` | Daftar peer P2P yang terhubung. |
| `POST` | `/node/sync` | Mendaftarkan URL node peer baru ke dalam topologi mesh P2P (dilindungi oleh validator anti-SSRF). |
| `DELETE` | `/node/peers?peer=...` | Memutuskan tautan peer dari daftar mesh P2P. |
| `POST` | `/node/broadcast` | Menyiarkan pesan protokol `GIP/2.0` bertanda tangan Ed25519 ke seluruh peer aktif. |
| `GET/POST` | `/node/config` | Membaca atau memperbarui konfigurasi runtime (`node_name`, `pow_difficulty`, `max_tokens`, `temperature`). |
| `GET` | `/node/dependencies` | Memeriksa versi Python dan status paket pustaka yang terinstal di server. |
| `GET/DELETE` | `/node/logs` | Mengambil atau membersihkan *ring-buffer* log telemetri server. |

---

## 🧪 Pengujian & Kriteria Penerimaan (19 Automated Tests)

Jalankan seluruh 19 suite pengujian otomatis menggunakan `pytest`:

```bash
pytest -v
```

| Test ID | Kriteria Penerimaan | File Pengujian | Status |
| :---: | :--- | :--- | :---: |
| **TEST 1** | GajahMada dapat berjalan tanpa kredensial penyedia AI eksternal. | `tests/unit/test_core_and_neural.py` | ✅ **PASSED** |
| **TEST 2** | Backend neural lokal (`llama.cpp` GGUF) dapat dimuat dan digunakan saat dikonfigurasi. | `tests/unit/test_core_and_neural.py` | ✅ **PASSED** |
| **TEST 3** | Sistem secara eksplisit melaporkan ketika inferensi neural tidak tersedia (`neural_inference_active: False`). | `tests/unit/test_core_and_neural.py` | ✅ **PASSED** |
| **TEST 4** | Penalaran simbolik (matematika AST, aturan logika, silogisme) beroperasi independen tanpa LLM. | `tests/cognition/test_cognitive_pipeline.py` | ✅ **PASSED** |
| **TEST 5** | Pipa kognitif menghasilkan output terstruktur `ReasoningResult` lengkap dengan `EpistemicBreakdown`. | `tests/cognition/test_cognitive_pipeline.py` | ✅ **PASSED** |
| **TEST 6** | Kesimpulan penting menyertakan referensi bukti (`evidence_id` & `provenance_hash`). | `tests/cognition/test_cognitive_pipeline.py` | ✅ **PASSED** |
| **TEST 7** | Memori persisten bertahan melewati *process restart* beserta riwayat versi dan backup. | `tests/memory/test_memory_persistence.py` | ✅ **PASSED** |
| **TEST 8** | Pengetahuan yang saling bertentangan (*contradictions*) tetap disimpan dan diidentifikasi secara otomatis. | `tests/memory/test_memory_persistence.py` | ✅ **PASSED** |
| **TEST 9** | Node beroperasi penuh secara lokal ketika terputus dari jaringan. | `tests/network/test_gip_and_collective.py` | ✅ **PASSED** |
| **TEST 10** | Dua node independen dapat saling mengotentikasi dan bertukar pesan `GIP/2.0` bertanda tangan Ed25519 (serta menolak *replay attack*). | `tests/network/test_gip_and_collective.py` | ✅ **PASSED** |
| **TEST 11** | Tiga node independen dapat mengoordinasikan, memverifikasi tugas terdistribusi, dan menyimpan ke memori kolektif. | `tests/network/test_gip_and_collective.py` | ✅ **PASSED** |
| **TEST 12** | Permintaan administratif tanpa otentikasi ditolak (`HTTP 401`). | `tests/security/test_security_and_governance.py` | ✅ **PASSED** |
| **TEST 13** | Peer jahat tidak dapat mengeksekusi perintah shell, memalsukan tanda tangan, atau membaca/mengubah memori `PRIVATE`. | `tests/security/test_security_and_governance.py` | ✅ **PASSED** |
| **TEST 14** | Eksperimen pembelajaran yang gagal pada uji regresi otomatis ditolak tanpa memengaruhi model/strategi aktif. | `tests/unit/test_core_and_neural.py` | ✅ **PASSED** |
| **TEST 15** | Seluruh perilaku API v1 yang ada tetap kompatibel sepenuhnya. | `tests/integration/test_api_compatibility.py` | ✅ **PASSED** |
| **EXT 16–19** | SSRF & seed guard, Signed Event PoW Ledger, Real Cognitive Benchmarks, & Closed-Loop Experience Strategy Adaptation. | `tests/` | ✅ **PASSED** |

---

## 💬 Perintah Cepat pada Command Interface (`Panel 03`)

Selain mengetik pertanyaan bahasa alami atau ekspresi matematika/logika, Anda dapat menjalankan perintah sistem langsung di kotak input **Command Interface**:

- `/crawl <topik atau URL>` — Crawling pengetahuan langsung dari internet & menyimpannya ke memori Brain (lokal + Supabase).
- `/status` — Menampilkan laporan langsung CPU, RAM, Disk, jumlah dokumen RAG, peer P2P, dan tinggi rantai blok.
- `/time` — Menampilkan sinkronisasi waktu sistem sekarang (WIB / ISO-8601).
- `/ingest <teks pengetahuan>` — Langsung meng-embed teks ke vektor 384-d dan menyimpannya ke dalam basis pengetahuan RAG, Semantic Memory, & Supabase.
- `/search <kueri pencarian>` — Mencari dokumen paling relevan secara semantik dari *vector store* 384-d.
- `/mine <catatan audit>` — Menambang blok Proof-of-Work baru ke dalam Web3 Signed Event Ledger.
- `/verify` — Menjalankan audit kriptografis penuh terhadap rantai blok SHA-256 dan tanda tangan Ed25519.
- `/peers` — Menampilkan daftar node peer P2P yang terhubung.

---

<div align="center">
  Support and download this repo on GitHub: <a href="https://github.com/hoed/gajahmada-ai">https://github.com/hoed/gajahmada-ai</a>
</div>
