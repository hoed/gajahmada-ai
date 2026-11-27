# GajahMada AI — Unified Model Registry & Runtime Promotion (`MODEL_REGISTRY_AND_PROMOTION.md`)

## Maturity Classification
- **Unified 11-State Platform Registry & Model Factory (`gajahmada_ai/platform/model_factory.py`):** **IMPLEMENTED AND TESTED**
- **Persistent Atomic Runtime `GNNMModelRegistry` (`gajahmada_ai/learning/model_registry.py`):** **IMPLEMENTED AND TESTED**
- **SHA-256 Checkpoint Integrity Check Before Promotion:** **IMPLEMENTED AND TESTED**
- **Live `ModelRouter` Staging, Smoke Test, Atomic Switch, & Rollback:** **IMPLEMENTED AND TESTED** *(verified in `.venv312` with PyTorch)*

## Unified 11-State Lifecycle (`platform/model_factory.py`)
Every model across `BUMBLECORE`, `OPTICORE`, and `TITANFORGE` is tracked across 11 explicit states:

`PLANNED` → `ARCHITECTURE_DESIGNED` → `CONFIGURATION_VALIDATED` → `DATASET_PREPARATION` → `READY_FOR_TRAINING` → `TRAINING` → `TRAINING_COMPLETED` → `EVALUATION` → `APPROVED` → `PRODUCTION` / `DEPRECATED`

- Reconciles strictly against actual on-disk artifacts (`models/gnnm-*`) and `GNNMModelRegistry`.
- Specializations (`NEUROX`, `CODESTRIKE`, `KNOWTRON`) are never represented as separate trained models without verified artifacts and evaluation results.
- On Vercel Free, the Model Factory manages metadata, architecture validation, parameter calculation, and resource estimation while keeping training execution explicitly `DISABLED_ON_VERCEL_FREE`.

## Runtime Promotion & Rollback Governance (`learning/model_registry.py`)
1. Candidates register in `DEVELOPMENT` / `TRAINING` and cannot jump directly to `ACTIVE` (invalid transitions raise `ValueError`).
2. Promotion requires `status == APPROVED`, `operator_approved == True`, and live on-disk SHA-256 matching `checkpoint_sha256`.
3. When coupled to `ModelRouter`, `_stage_and_switch_runtime()` loads the candidate onto a fresh `PyTorchNativeBackend`, runs a real inference smoke check, and swaps atomically under `_switch_lock` — preserving the previous `ACTIVE` backend if staging fails.
4. `rollback()` verifies and restores `rollback_reference` atomically.
