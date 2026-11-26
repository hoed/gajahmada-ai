# GajahMada AI — GNNM Learning Pipeline (`GNNM_LEARNING_PIPELINE.md`)

## Maturity Classification
- **Training Knowledge Filter (`TrainingKnowledgeFilter`):** **IMPLEMENTED AND TESTED**
- **Continued Pretraining & SFT Dataset Separation (`KnowledgeDatasetBuilder`):** **IMPLEMENTED AND TESTED**
- **Strict P0 Training Integrity Guard (`TRAINING_NOT_EXECUTED`, `CHECKPOINT_MISSING`, `CHECKPOINT_INVALID`, `TRAINING_FAILED`):** **IMPLEMENTED AND TESTED**
- **Isolated Checkpoint Copy + Real PyTorch `GNNMTrainer` Execution:** **IMPLEMENTED AND TESTED** *(verified in `.venv312` with PyTorch CPU)*
- **Large-Scale Distributed GPU Pretraining (100M–1B):** **PLANNED**

## Strict Training Integrity Rules (`gajahmada_ai/learning/training_pipeline.py`)
1. Ingestion and normal chat **never** trigger training or modify model weights.
2. A candidate can enter `EVALUATING` **only** when:
   - Base checkpoint passes SHA-256 structural verification (`compute_checkpoint_checksum`).
   - Approved, filtered dataset has `item_count > 0`.
   - PyTorch (`HAS_TORCH`) is installed and `GNNMTrainer` completes without exception.
   - Candidate checkpoint artifacts (`config.json`, `tokenizer.json`, `model.pt`) pass SHA-256 verification.
   - Candidate checkpoint loads in `GNNMInferenceEngine` and completes a real inference smoke check.
3. Missing PyTorch returns `TRAINING_NOT_EXECUTED` — never `APPROVED`.
4. Every run records parent vs. candidate SHA-256 hashes, `weights_changed`, `config_fingerprint`, `dataset_sha256`, `loss_history`, and `training_duration_sec`.
