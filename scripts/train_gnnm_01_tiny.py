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
    trainer = GNNMTrainer(cfg, tok, lr=1e-3)
    metrics = trainer.train_steps(pairs, steps=60, batch_size=4)
    print(f"Training complete: {metrics}")

    saved = trainer.save_checkpoint(output_dir)
    print(f"Checkpoint saved to: {saved}")

    print("=== 5. Verifying Native Autoregressive Inference ===")
    engine = GNNMInferenceEngine(output_dir)
    sample_id = engine.generate("GajahMada adalah", max_new_tokens=28, temperature=0.7)
    sample_en = engine.generate("GajahMada is a sovereign", max_new_tokens=28, temperature=0.7)
    print(f"[ID Sample] GajahMada adalah -> {sample_id}")
    print(f"[EN Sample] GajahMada is a sovereign -> {sample_en}")


if __name__ == "__main__":
    main()
