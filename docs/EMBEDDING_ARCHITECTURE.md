# GajahMada AI — Embedding Architecture (`EMBEDDING_ARCHITECTURE.md`)

## Maturity Classification
- **`HashEmbeddingProvider` (`deterministic-hash-384d` v1.0.0):** **IMPLEMENTED AND TESTED** *(offline fallback)*
- **`NativeSemanticEmbeddingProvider` (`gajahmada-native-semantic-384d` v2.0.0):** **IMPLEMENTED AND TESTED** *(explicitly classified as `heuristic_semantic_projection`, `is_learned_model: False`)*
- **Strict Vector Compatibility Validation & Reversible `EmbeddingMigrationManager`:** **IMPLEMENTED AND TESTED**
- **`LocalLearnedEmbeddingProvider` & `FutureGNNMEmbeddingProvider` Interfaces:** **IMPLEMENTED** *(honestly reports `NOT_CONFIGURED` / raises `RuntimeError` when no trained weights are present)*
- **Locally Trained Contrastive Neural Embedding Weights:** **PLANNED**

## Compatibility Rules (`gajahmada_ai/knowledge/embedding.py`)
Every indexed chunk stores `embedding_encoder_id` and `embedding_version`. `HybridKnowledgeRetriever` verifies `is_compatible_with(encoder_id, encoder_version, vector)` before cosine similarity — incompatible vectors are never silently compared. `EmbeddingMigrationManager` supports batched, reversible re-embedding with rollback snapshots.
