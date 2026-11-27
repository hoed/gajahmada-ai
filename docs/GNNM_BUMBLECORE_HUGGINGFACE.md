# GajahMada GNNM BumbleCore 300M — Hugging Face Distribution Guide

- **Target Repository (Unclaimed Until Created):** `GajahMada/GNNM-BumbleCore-300M`
- **Format Honesty:** Exported artifacts are explicitly marked `format: "custom_gnnm_native"` and `transformers_auto_compatible: false`. They load natively via `GNNMInferenceEngine` (never falsely labeled as standard Hugging Face `AutoModelForCausalLM`).
- **Artifacts Exported:** `config.json`, `tokenizer.json`, `generation_config.json`, `model.pt` (+ `model.safetensors` when `safetensors` is installed), `README.md` model card, `artifact_manifest.json` (SHA-256 checksums).
- **Excluded Files:** `trainer_state.pt`, `.env`, `node_key.pem`, `admin_api.key` are strictly blocked by `prepare_hf_upload()`.
- **Publication Policy:** Defaults to `private=True` and `dry_run=True`. Actual upload requires explicit user authorization (`confirm_upload=True, dry_run=False`) and `HF_TOKEN`.
