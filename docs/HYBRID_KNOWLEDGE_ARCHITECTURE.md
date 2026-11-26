# GajahMada AI — Hybrid Knowledge Architecture (`HYBRID_KNOWLEDGE_ARCHITECTURE.md`)

## Maturity Classification
- **Three-System Logical Separation (Internal / External / Persistent Memory):** **IMPLEMENTED AND TESTED**
- **Hybrid Knowledge Controller & `GNNMKnowledgeAdapter`:** **IMPLEMENTED AND TESTED**
- **SQLite WAL External Chunk Persistence & `SourceRegistry`:** **IMPLEMENTED AND TESTED**
- **Immutable Per-Request Token-Budget Context Construction:** **IMPLEMENTED AND TESTED**
- **Distributed Remote Node Retrieval Hook:** **PARTIALLY IMPLEMENTED** *(interface & local federation tested; WAN transport pending)*

## Architecture Overview
GajahMada AI separates intelligence knowledge into three distinct, independently governed systems:
1. **Internal Knowledge (`gajahmada_ai/gnnm/`):** Knowledge internalized in GNNM Decoder-Only Transformer weights, updated solely through the isolated `KnowledgeToModelPipeline`.
2. **External Knowledge (`gajahmada_ai/knowledge/`):** Multi-format documents (TXT, Markdown, HTML, JSON, CSV, pure-Python DOCX, optional PDF) ingested via `KnowledgeIngestionPipeline`, chunked by `DocumentChunker`, registered in `SourceRegistry`, and persisted in SQLite WAL (`external_knowledge.db`) for retrieval without retraining.
3. **Persistent Memory (`gajahmada_ai/memory/`):** 5-tier operational memory (`working`, `semantic`, `episodic`, `procedural`, `collective`) managed by `MemoryManager` with direct `AccessContext` enforcement.

`HybridKnowledgeController` coordinates retrieval decisions, agent policies, conflict detection (`ContradictionDetector.check_pair`), and per-request immutable `TokenBudgetContextBuilder` contexts wrapped in `<untrusted_retrieved_data>` delimiters before passing into `ModelRouter` and `CognitiveReasoningPipeline`.
