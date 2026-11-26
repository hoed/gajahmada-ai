# GajahMada AI — Layered Evaluation Framework (`EVALUATION_FRAMEWORK.md`)

## Maturity Classification
- **6-Dimension `ModelEvaluationGate` (`gajahmada_ai/learning/evaluation_gate.py`):** **IMPLEMENTED AND TESTED**
- **Train/Validation Dataset Leakage Detection:** **IMPLEMENTED AND TESTED**
- **Baseline Regression Delta Comparison:** **IMPLEMENTED AND TESTED**
- **Large-Scale External Multilingual Benchmark Suites (1,000+ Held-Out Items):** **PLANNED**

## Evaluation Dimensions (`gm-eval-v2.1`)
1. **Functional:** Non-empty output, zero runtime exceptions.
2. **Language:** Indonesian (`id_lang`), English (`en_lang`), and mixed Indonesian-English (`mixed_lang`).
3. **Knowledge:** Factual accuracy, citation accuracy, hallucination rate, and explicit **abstention** when evidence is insufficient.
4. **Reasoning:** Multi-step evidence sufficiency logic.
5. **Security:** Prompt-injection & malicious `<untrusted_retrieved_data>` resilience (`1.0` required).
6. **Regression & Leakage:** Rejects candidates with `dataset_leakage_detected`, perplexity degradation (`> 1.15x` baseline), or accuracy regression (`> 0.05` below baseline).
