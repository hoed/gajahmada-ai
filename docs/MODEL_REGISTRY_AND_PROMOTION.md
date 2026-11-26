# GajahMada AI — Model Registry & Runtime Promotion (`MODEL_REGISTRY_AND_PROMOTION.md`)

## Maturity Classification
- **Persistent Atomic `GNNMModelRegistry` (`gajahmada_ai/learning/model_registry.py`):** **IMPLEMENTED AND TESTED**
- **Strict State Machine (`DEVELOPMENT → TRAINING → EVALUATING → APPROVED → ACTIVE → ARCHIVED` / `REJECTED`):** **IMPLEMENTED AND TESTED**
- **SHA-256 Checkpoint Integrity Check Before Promotion:** **IMPLEMENTED AND TESTED**
- **Live `ModelRouter` Staging, Smoke Test, Atomic Switch, & Rollback:** **IMPLEMENTED AND TESTED** *(verified in `.venv312` with PyTorch)*

## Governance Flow
1. Candidates register in `DEVELOPMENT` / `TRAINING` and cannot jump directly to `ACTIVE` (invalid transitions raise `ValueError`).
2. Promotion requires `status == APPROVED`, `operator_approved == True`, and live on-disk SHA-256 matching `checkpoint_sha256`.
3. When coupled to `ModelRouter`, `_stage_and_switch_runtime()` loads the candidate onto a fresh `PyTorchNativeBackend`, runs a real inference smoke check, and swaps atomically under `_switch_lock` — preserving the previous `ACTIVE` backend if staging fails.
4. `rollback()` verifies and restores `rollback_reference` atomically.
