# GajahMada GNNM BumbleCore 300M — Evaluation Framework (Release 2.6)

Implemented in [evaluation.py](file:///home/hoed/gajahmada/gajahmada_ai/gnnm/bumblecore/evaluation.py) (`bumblecore-eval-v2.6`).

## 1. Three Strictly Separated Evaluation Levels
1. **Level A — Functional & Unit Checks (`functional`):** Structural & deterministic execution checks.
2. **Level B — Model Capability Benchmarks (`capability`):** Uses objective metrics — `exact_match`, `normalized_exact_match`, `token_f1` (strict `>= 0.45` threshold with zero keyword-matching fallback on the independent suite), `numerical_exact`, `json_valid`, plus `val_loss` / `perplexity`.
3. **Level C — Safety & Robustness (`safety`):** Prompt injection resistance, sensitive credential/key leakage (`safety_forbidden`), and honest abstention/uncertainty handling (`abstention`).

## 2. Independent Bilingual Benchmark Suite & Anti-Leakage Controls
- **`INDEPENDENT_BENCHMARK_CASES` (15 bilingual ID/EN cases):** Disjoint from the built-in seed corpus, spanning Indonesian comprehension, English comprehension, arithmetic/reasoning, structured JSON output, abstention, and safety.
- **Benchmark Dataset Integrity (`compute_benchmark_dataset_sha256()`):** Computes a deterministic SHA-256 digest over the exact evaluated benchmark cases.
- **Training-Leakage Detection (`check_benchmark_training_leakage()`):** Compares benchmark prompts/references against training corpus text and flags `BENCHMARK_LEAKAGE_DETECTED` (`clean: false`) if any overlap is found — automatically failing Gate E.
- **Output Stability (`measure_output_stability()`):** Measures deterministic repeatability across repeated evaluations.

## 3. Benchmark Integrity & Gate E Coupling
- Empty held-out benchmarks immediately return `status: "EMPTY_BENCHMARK_FAILED"` (never fake `0.0` or `1.0`).
- Every evaluation report records `model_identity`, `actual_parameters`, `checkpoint_sha256`, `tokenizer_sha256`, `benchmark_dataset_sha256`, `training_leakage_check`, `dataset_identity`, `dataset_version`, `benchmark_version`, `seed`, `actual_sample_count`, `metric_definitions`, `levels`, and `failure_cases`.
- Unexecuted dimensions are explicitly labeled `NOT_EXECUTED`.
- `verify_evaluation_evidence()` in [gate_evidence.py](file:///home/hoed/gajahmada/gajahmada_ai/gnnm/bumblecore/gate_evidence.py) requires `status == "COMPLETED"`, `training_leakage_check.clean == True`, valid `benchmark_dataset_sha256`, minimum overall & safety scores, and a `checkpoint_sha256` matching a verified non-smoke 300M checkpoint.
