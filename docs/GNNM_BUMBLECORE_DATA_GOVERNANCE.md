# GajahMada GNNM BumbleCore 300M — Data Governance Specification (Release 2.6)

## 1. Strict Dataset Separation
- **Smoke-Test Data:** Built-in seed corpus is permitted **strictly** for unit tests, architecture tests, `--smoke` runs, and pipeline validation. Full pretraining and full SFT **refuse** to execute without an approved external dataset + verified manifest.
- **Pretraining Data:** External JSONL corpus + versioned `DatasetManifest` (`approval_status == "approved"`, approved license, verified SHA-256 hash, measured document & token counts).
- **SFT Data:** Separate external multi-turn JSONL instruction dataset + manifest meeting explicit minimum sample threshold (`min_sft_examples`), instruction schema validation, and train/validation split.
- **Validation & Held-Out Evaluation Data:** Disjoint splits enforced by SHA-256 `document_hash` AND `source_doc_id` / `source_url` so no document or source chunk leaks across splits.

## 2. Approved License Allowlist
Unknown, missing, or unreviewed licenses are **never** treated as approved. Allowed SPDX/verified licenses:
`MIT`, `Apache-2.0`, `CC-BY-4.0`, `CC-BY-SA-4.0`, `CC0-1.0`, `BSD-3-Clause`, `BSD-2-Clause`, `MPL-2.0`, `ODC-By-1.0`.

## 3. Bounded-Memory Streaming Preprocessing
`BoundedStreamingPreprocessor` (`gajahmada_ai/gnnm/bumblecore/dataset_governance.py`) streams JSONL line-by-line, stores exact SHA-256 & 64-bit MinHash/simhash fingerprints in an on-disk **SQLite WAL** index (`fingerprints.db`), writes accepted records incrementally to split JSONL files (`train.jsonl`, `validation.jsonl`, `test.jsonl`), and counts actual tokenizer tokens incrementally without holding the full corpus in RAM.
