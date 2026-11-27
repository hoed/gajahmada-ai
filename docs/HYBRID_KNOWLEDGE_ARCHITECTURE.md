# GajahMada AI — Hybrid Knowledge Architecture (`HYBRID_KNOWLEDGE_ARCHITECTURE.md`)

## Maturity Classification
- **Three-System Logical Separation (Internal / External / Persistent Memory):** **IMPLEMENTED AND TESTED**
- **Hybrid Knowledge Controller & `GNNMKnowledgeAdapter`:** **IMPLEMENTED AND TESTED**
- **FAISS CPU Preservation & Provider Abstraction (`providers.py`):** **IMPLEMENTED AND TESTED** *(FAISS CPU local/server + honest `LightweightRetrievalProvider` fallback on Vercel Free)*
- **SQLite WAL External Chunk Persistence & `SourceRegistry`:** **IMPLEMENTED AND TESTED**
- **Question–Answer Relevance Gate & Curated Offline World/Space Facts (252 entries):** **IMPLEMENTED AND TESTED**
- **Immutable Per-Request Token-Budget Context Construction:** **IMPLEMENTED AND TESTED**
- **Distributed Remote Node Retrieval Hook:** **PARTIALLY IMPLEMENTED** *(interface & local federation tested; WAN transport planned)*

## Architecture Overview
GajahMada AI separates intelligence knowledge into three independently governed systems:
1. **Internal Knowledge (`gajahmada_ai/gnnm/`):** Native GNNM Transformer weights, updated solely through the isolated `KnowledgeToModelPipeline`.
2. **External Knowledge (`gajahmada_ai/knowledge/`):** Multi-format documents (TXT, MD, HTML, JSON, CSV, DOCX, PDF), curated offline world/space facts (`world_facts.py`), and Knowledge Graph persisted in SQLite WAL (`external_knowledge.db`) + `SourceRegistry` for retrieval without retraining.
3. **Persistent Memory (`gajahmada_ai/memory/`):** 5-tier operational memory (`working`, `semantic`, `episodic`, `procedural`, `collective`) enforced directly by `AccessContext` across 9 scopes, complemented by browser-local `localStorage` / `IndexedDB` clearly distinguished from server persistence.

## Retrieval Providers & FAISS CPU Preservation (`gajahmada_ai/knowledge/providers.py`)
- **`FaissLocalProvider`:** Preserves FAISS CPU `IndexFlatL2` for local development, testing, and compatible server environments.
- **`FaissArtifactProvider`:** Loads validated prebuilt FAISS index artifacts (checking dimension, encoder compatibility, and size limits) when runtime supports it.
- **`LightweightRetrievalProvider`:** Pure-Python lexical + L2/BM25 fallback for Vercel Free (`uses_faiss: False`, never falsely claiming FAISS vector similarity).
- Every provider reports an explicit status: `ACTIVE`, `AVAILABLE_LOCALLY`, `NOT_CONFIGURED`, `INCOMPATIBLE_RUNTIME`, or `DISABLED`.

## Grounding, Relevance & Untrusted Data Defense
`HybridKnowledgeController` enforces retrieval-boundary `KnowledgeAccessPolicy`, detects conflicts via `ContradictionDetector.check_pair`, wraps retrieved chunks in `<untrusted_retrieved_data>` delimiters via immutable per-request `TokenBudgetContextBuilder`, and pairs with `answer_relevance.py` so answers address the actual question asked (ordinals, dates, numbers, qualifiers) or abstain honestly.
