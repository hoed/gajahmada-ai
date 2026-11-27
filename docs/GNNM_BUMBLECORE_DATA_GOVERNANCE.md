# GajahMada GNNM BumbleCore 300M — Data Governance Specification (Release 2.6)

Implemented in [dataset_governance.py](file:///home/hoed/gajahmada/gajahmada_ai/gnnm/bumblecore/dataset_governance.py) and [streaming_loader.py](file:///home/hoed/gajahmada/gajahmada_ai/gnnm/bumblecore/streaming_loader.py).

## 1. Strict Dataset Separation
- **Smoke-Test Data:** Built-in seed corpus is permitted **strictly** for unit tests, architecture tests, `--smoke` runs, and pipeline validation. Full pretraining and full SFT **refuse** to execute without an approved external dataset + verified manifest (`DATASET_GOVERNANCE_FAILED`).
- **Pretraining Data:** External JSONL corpus + versioned `DatasetManifest` (`approval_status == "approved"`, approved license, verified SHA-256 split hashes, measured document & token counts).
- **SFT Data:** Separate external multi-turn JSONL instruction dataset + manifest meeting explicit minimum sample threshold (`min_sft_examples`), instruction schema validation, and train/validation split.
- **Validation & Held-Out Evaluation Data:** Disjoint splits enforced by SHA-256 `document_hash` AND `source_doc_id` / `source_url` so no document or source chunk leaks across splits (`DOCUMENT_LEAKAGE`, `SOURCE_LEAKAGE`).

## 2. Manifest Schema, Path Safety & Integrity Verification
- `DatasetManifest` records: `dataset_id`, `dataset_version`, `dataset_description`, `source_name`, `license`, `language`, `language_distribution`, `created_at`, `approval_status`, `approval_evidence`, `document_count`, `token_count`, `split_hashes`, and `manifest_checksum`.
- `_resolve_safe_split_path()` strictly rejects absolute paths, `../` directory traversal, and symlink escapes outside the dataset root.
- `validate_dataset_and_manifest()` verifies mandatory `split_hashes` against live file SHA-256 digests, rejects `TOKEN_COUNT_MISMATCH`, rejects `DOCUMENT_ID_COLLISION`, and rejects cross-split `DOCUMENT_LEAKAGE` / `SOURCE_LEAKAGE`.

## 3. Approved License Allowlist
Unknown, missing, or unreviewed licenses are **never** treated as approved. Allowed SPDX/verified licenses:
`MIT`, `Apache-2.0`, `CC-BY-4.0`, `CC-BY-SA-4.0`, `CC0-1.0`, `BSD-3-Clause`, `BSD-2-Clause`, `MPL-2.0`, `ODC-By-1.0`.

## 4. Bounded-Memory Streaming Preprocessing & Training Loader
- `BoundedStreamingPreprocessor` streams raw JSONL line-by-line, stores exact SHA-256 & 64-bit SimHash fingerprints in an on-disk **SQLite WAL** index (`fingerprints.db`), writes accepted records incrementally to `train.jsonl`, `validation.jsonl`, and `test.jsonl`, and counts actual tokenizer tokens incrementally without holding the full corpus in RAM.
- `StreamingPretrainDataset` & `StreamingSFTDataset` stream directly into `GNNMTrainer` with lazy tokenization, streaming sequence packing, bounded shuffle buffers, `skip_sequences` cursor restoration on resume, and RSS/throughput telemetry.
