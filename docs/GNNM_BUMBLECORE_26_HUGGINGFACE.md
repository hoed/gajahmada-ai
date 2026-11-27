# GajahMada GNNM BumbleCore 300M — Checkpoint & Hugging Face Distribution (Release 2.6)

Implemented in [export.py](file:///home/hoed/gajahmada/gajahmada_ai/gnnm/bumblecore/export.py).

## 1. Honest Training Status & Untrained Guard
- `classify_training_status()` classifies every checkpoint honestly as `UNTRAINED_OR_RANDOM_INIT`, `SMOKE_TEST_ONLY`, or `TRAINED_WITH_PROVENANCE`.
- `export_inference_artifact()` validates `config.vocab_size == tokenizer.vocab_size`, rejects `NaN`/`Inf` tensors before writing, auto-loads `run_metadata.json`, and supports `require_trained=True` to refuse exporting untrained or smoke checkpoints.
- `generate_hf_model_card()` embeds the exact `training_status` and explicitly warns when weights are untrained or smoke-only; `validate_hf_model_card()` validates YAML frontmatter, license (`mit`), `library_name: gajahmada-gnnm`, required sections, and rejects forged `AutoModelForCausalLM` or `TRAINED_WITH_PROVENANCE` claims.

## 2. Full Export/Import & Safetensors Parity
- `verify_export_import_parity()` reloads source and exported checkpoints and verifies exact tensor key sets, shapes (`shapes_match`), dtypes (`dtypes_match`), numerical values (`values_match`), config/tokenizer equivalence, finite deterministic logits (`max_logit_diff <= 1e-6`), and full `model.safetensors` parity when `safetensors` is installed.

## 3. Recursive Safety Scan, Bundle Validation & Upload Auth Gate
- `scan_export_directory_recursive()` walks the entire export tree (`followlinks=False`), rejects all symbolic links, rejects all forbidden patterns (`trainer_state.pt`, `.env`, `.pem`, `.key`, `.db`, `.sqlite`, `.log`, `fingerprints.db`), and permits ONLY:
  `config.json`, `tokenizer.json`, `generation_config.json`, `model.pt`, `model.safetensors`, `README.md`, `artifact_manifest.json`, `LICENSE`.
- `validate_export_bundle()` combines the recursive safety scan, live SHA-256 verification of `artifact_manifest.json`, `config.json ↔ tokenizer.json ↔ generation_config.json` consistency, and `validate_hf_model_card()`.
- `prepare_hf_upload()` defaults to `private=True`, `dry_run=True`, `confirm_upload=False` — **never** uploads automatically. Live upload (`dry_run=False, confirm_upload=True`) additionally enforces `validate_export_bundle()`, rejects untrained/smoke bundles unless `allow_untrained=True`, and requires `HF_TOKEN`.
- Exported artifacts remain explicitly marked `format: "custom_gnnm_native"` (`transformers_auto_compatible: false`).
