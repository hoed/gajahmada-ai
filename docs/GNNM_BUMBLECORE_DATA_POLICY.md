# GajahMada GNNM BumbleCore 300M — Data Governance & Licensing Policy

## 1. Core Principles
1. **No Implicit Public-Web License Assumption:** Publicly accessible web pages are **not** automatically licensed for model training or weight redistribution. Every corpus source must declare an explicit license (`MIT`, `CC-BY-4.0`, `CC-BY-SA-4.0`, `CC0`, `Apache-2.0`, or verified institutional permission) in its manifest.
2. **No Private Data Publication:** Private user conversations, internal node keys, API keys, and unconsented personal data are strictly excluded from training corpora and never published.
3. **PII & Secret Redaction:** `BumbleCoreDataPipeline` scans and redacts emails, Indonesian phone numbers, 16-digit IDs (NIK), and API key prefixes (`sk-`, `gm-admin-`, `ghp_`, `hf_`) during deterministic preprocessing.
4. **Zero Train/Validation/Test Leakage:** Splits (`90% train / 5% validation / 5% test`) are assigned deterministically from SHA-256 `document_hash` after exact & near-duplicate filtering so no document appears across multiple splits.
5. **Pretraining vs. SFT Separation:** Pretraining corpora and supervised instruction-tuning (SFT) conversations are maintained as separate versioned datasets with independent manifests and checkpoints.

## 2. Required Document Metadata Schema
Every accepted training record preserves:
- `dataset_id`, `dataset_version`, `preprocessing_version`
- `source_name`, `source_url`, `license`, `collection_date`
- `language` (`id`, `en`, `mixed`), `document_hash` (SHA-256), `split`, `quality_score`
