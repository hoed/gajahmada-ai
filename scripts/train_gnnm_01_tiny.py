#!/usr/bin/env python3
"""
Train GajahMada Native Neural Model — Stage 0 Proof-of-Concept (`GNNM-01-Tiny`).
Produces `./models/gnnm-01-tiny/` containing:
  - `config.json`    (Transformer architecture configuration)
  - `tokenizer.json` (Native Byte-Level BPE multilingual vocabulary)
  - `model.pt`       (Self-trained PyTorch weights — 100% from scratch)
"""

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from gajahmada_ai.gnnm import GNNM_PRESETS, GNNMBPETokenizer, GNNMCorpusBuilder
from gajahmada_ai.gnnm.inference import GNNMInferenceEngine
from gajahmada_ai.gnnm.trainer import GNNMTrainer


def main():
    output_dir = os.path.join(ROOT, "models", "gnnm-01-tiny")

    print("=== 1. Building Balanced Multilingual Corpus (ID + EN) ===")
    corpus = GNNMCorpusBuilder()
    n_docs = corpus.load_seed_multilingual_corpus()
    print(f"Loaded {n_docs} unique deduplicated documents.")

    print("=== 2. Training Native Byte-Level BPE Tokenizer ===")
    tok = GNNMBPETokenizer()
    tok_stats = tok.train(corpus.texts(), target_vocab_size=512, min_frequency=2)
    print(f"Tokenizer ready: {tok_stats}")

    print("=== 3. Packing Causal LM Training Sequences ===")
    seq_len = 64
    pairs = corpus.pack_token_blocks(tok, seq_len=seq_len)
    print(f"Packed {len(pairs)} training blocks (seq_len={seq_len}).")

    print("=== 4. Initializing & Pretraining GNNM-01-Tiny From Scratch ===")
    cfg = GNNM_PRESETS["GNNM-01-Tiny"]
    cfg.vocab_size = tok.vocab_size
    trainer = GNNMTrainer(cfg, tok, lr=2e-3)
    metrics = trainer.train_steps(pairs, steps=150, batch_size=8)
    print(f"Training complete: {metrics}")

    saved = trainer.save_checkpoint(output_dir)
    print(f"Checkpoint saved to: {saved}")

    print("=== 5. Verifying Native Autoregressive Inference & PyTorchNativeBackend Integration ===")
    from gajahmada_ai.neural.pytorch_backend import PyTorchNativeBackend

    backend = PyTorchNativeBackend(model_path=output_dir)
    assert backend.load_model(), f"Backend load failed: {backend.health()}"
    res_id = backend.generate("GajahMada", max_tokens=24, temperature=0.5)
    res_en = backend.generate("GajahMada is", max_tokens=24, temperature=0.5)
    print(f"Backend Health: {backend.health()}")
    print(f"[ID Sample] GajahMada -> {res_id.text}")
    print(f"[EN Sample] GajahMada is -> {res_en.text}")


if __name__ == "__main__":
    main()
