# GajahMada GNNM BumbleCore 300M — Evaluation Framework (Release 2.6)

Implemented in [evaluation.py](file:///home/hoed/gajahmada/gajahmada_ai/gnnm/bumblecore/evaluation.py) (`bumblecore-eval-v2.6`).

## 1. Three Strictly Separated Evaluation Levels
1. **Level A — Functional & Unit Checks (`functional`):** Structural & deterministic execution checks.
2. **Level B — Model Capability Benchmarks (`capability`):** Uses objective metrics — `exact_match`, `normalized_exact_match`, `token_f1`, `numerical_exact`, `json_valid`, plus `val_loss` / `perplexity`. Keyword-only scoring is never presented as standalone capability proof.
3. **Level C — Safety & Robustness (`safety`):** Prompt injection resistance, sensitive credential/key leakage, and honest abstention/uncertainty handling.

## 2. Benchmark Integrity Rules
- Empty held-out benchmarks immediately return `status: "EMPTY_BENCHMARK_FAILED"` (never fake `0.0` or `1.0`).
- Every evaluation report records `checkpoint_sha256`, `tokenizer_sha256`, `dataset_identity`, `dataset_version`, `benchmark_version`, `seed`, `actual_sample_count`, `metric_definitions`, `levels`, and `failure_cases`.
- Unexecuted dimensions are explicitly labeled `NOT_EXECUTED`.
