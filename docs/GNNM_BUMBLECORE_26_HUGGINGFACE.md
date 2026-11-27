# GajahMada GNNM BumbleCore 300M — Checkpoint & Hugging Face Distribution (Release 2.6)

Implemented in [export.py](file:///home/hoed/gajahmada/gajahmada_ai/gnnm/bumblecore/export.py).

## 1. Explicit Release Artifact Allowlist
`scan_export_directory_recursive()` walks the entire export tree (`followlinks=False`), rejects all symbolic links, rejects all forbidden patterns (`trainer_state.pt`, `.env`, `.pem`, `.key`, `.db`, `.sqlite`, `.log`), and permits ONLY:
`config.json`, `tokenizer.json`, `generation_config.json`, `model.pt`, `model.safetensors`, `README.md`, `artifact_manifest.json`, `LICENSE`.

## 2. Auditable Upload Manifest & Auth Gate
- Every allowed file is recorded with `relative_path`, `size_bytes`, `sha256`, `artifact_type`, and `included: true`.
- `prepare_hf_upload()` defaults to `private=True`, `dry_run=True`, `confirm_upload=False` — never uploads automatically without explicit user approval and `HF_TOKEN`.
- Exported artifacts remain explicitly marked `format: "custom_gnnm_native"` (`transformers_auto_compatible: false`).
