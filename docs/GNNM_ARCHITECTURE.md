# GajahMada Native Neural Model (GNNM) — Architectural Blueprint

## 1. Locked Strategic Direction
1. **Neural Architecture:** Custom Decoder-Only Transformer built from scratch in PyTorch (**Pre-Norm RMSNorm + Rotary Positional Embeddings (RoPE) + Grouped/Multi-Head Causal Self-Attention + SwiGLU Feed-Forward**).
2. **Language Identity:** **Multilingual Seimbang** (balanced Bahasa Indonesia, regional Nusantara terms, & international/English technical knowledge) powered by a native byte-level BPE tokenizer (`GNNMBPETokenizer`) with zero `<unk>` drops.
3. **Intelligence Priority:** **Hybrid Intelligence** — GNNM serves as a compact, sovereign language understanding & concise generation core integrated directly into GajahMada's 14-stage Cognitive, Symbolic, Evidence, Memory, and GIP/2.0 Collective Verification Engine.

---

## 2. Progressive Scaling Roadmap (`GNNM-01` → `GNNM-05`)

| Stage | Preset Key | Target Parameters | `vocab_size` | `d_model` | Layers | Heads (Q/KV) | Context | Goal & Verified Status |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **PoC Smoke** | `GNNM-01-Tiny` | ~918K–1.5M | 512–1,024 | 128 | 4 | 4 / 4 | 256 | ✅ **Trained & Verified** (`models/gnnm-01-tiny/`, 918K params) |
| **Stage 1 (10–50M)** | `GNNM-01-12M` | ~11.2M–12.4M | 2,048–4,096 | 384 | 6 | 6 / 6 | 512 | ✅ **Trained & Verified** (`models/gnnm-01-12m/`, 11,243,520 params, CPU pretrain + SFT) |
| **Stage 1 (10–50M)** | `GNNM-01-25M` | ~18M–25M | 8,192 | 384 | 8 | 6 / 6 | 1,024 | Ready for extended corpus & GPU pretraining |
| **Stage 1 (100M)** | `GNNM-02-100M` | ~105M | 32,000 | 768 | 12 | 12 / 4 | 2,048 | Basic balanced multilingual fluency (GPU scale) |
| **Stage 1 (300M)** | `GNNM-03-300M` | ~320M | 32,000 | 1,024 | 20 | 16 / 4 | 4,096 | Strong grounded generation & synthesis (GPU scale) |
| **Stage 1 (1B)** | `GNNM-04-1B` | ~1.1B | 48,000 | 2,048 | 22 | 16 / 4 | 4,096 | Independent sovereign general LM (GPU scale) |

---

## 3. Implemented Package Structure (`gajahmada_ai/gnnm/`)
- `config.py`: `GNNMConfig` & `GNNM_PRESETS` with exact analytical parameter counter.
- `tokenizer.py`: `GNNMBPETokenizer` — pure-Python byte-level BPE trained from scratch (supports Indonesian, English, Arabic, math, & code losslessly).
- `dataset.py`: `GNNMCorpusBuilder` — text cleaning, SHA-256 deduplication, multilingual seeding, and causal LM sequence packing.
- `model.py`: `GNNMTransformer` — custom PyTorch Decoder-Only Transformer (`RMSNorm`, `RoPE`, `CausalSelfAttention`, `SwiGLUFeedForward`, autoregressive `generate()`).
- `trainer.py`: `GNNMTrainer` — AdamW + cosine LR schedule + gradient clipping + validation perplexity + SFT (`-100` prompt masking) + checkpoint saver (`config.json`, `tokenizer.json`, `model.pt`, `metrics.json`).
- `inference.py`: `GNNMInferenceEngine` — native checkpoint loader & generator for CPU/CUDA, integrated with token-budgeted context allocation in `PyTorchNativeBackend` and `GNNMKnowledgeAdapter`.

---

## 4. Hybrid Knowledge & Controlled Internalization
GNNM operates as the **Internal Knowledge** pillar inside GajahMada's **Hybrid Knowledge Architecture**:
- **Inference Time:** `GNNMKnowledgeAdapter` retrieves authorized External Knowledge (`external_knowledge.db`) and Persistent Memory (`MemoryManager`), builds a prompt-injection-hardened, token-budgeted `<untrusted_retrieved_data>` context respecting `cfg.max_seq_len`, and routes through `ModelRouter` without altering the checkpoint format.
- **Training Time (Isolated):** Verified, licensed knowledge is filtered by `TrainingKnowledgeFilter`, packaged into separate `continued_pretraining` or `sft` datasets by `KnowledgeDatasetBuilder`, trained on an isolated checkpoint copy by `KnowledgeToModelPipeline`, gated by `ModelEvaluationGate`, and promoted/rolled back via `GNNMModelRegistry` (`/api/v2/learning/{train,promote,rollback}`).

---

## 5. Quick Start (CLI Training)
```python
from gajahmada_ai.gnnm import GNNM_PRESETS, GNNMBPETokenizer, GNNMCorpusBuilder
from gajahmada_ai.gnnm.trainer import GNNMTrainer

corpus = GNNMCorpusBuilder()
corpus.load_seed_multilingual_corpus()

tok = GNNMBPETokenizer()
tok.train(corpus.texts(), target_vocab_size=512)

cfg = GNNM_PRESETS["GNNM-01-Tiny"]
cfg.vocab_size = tok.vocab_size

pairs = corpus.pack_token_blocks(tok, seq_len=64)
trainer = GNNMTrainer(cfg, tok)
metrics = trainer.train_steps(pairs, steps=50, batch_size=4)
trainer.save_checkpoint("./models/gnnm-01-tiny")
```
