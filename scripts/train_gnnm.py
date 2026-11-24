#!/usr/bin/env python3
"""
GajahMada Native Neural Model (GNNM) — Scalable CLI Pretraining + SFT Pipeline.

Examples:
  # Fast local CPU proof-of-concept (Pretrain + Val Perplexity + SFT)
  .venv312/bin/python scripts/train_gnnm.py --preset GNNM-01-Tiny --steps 80 --sft-steps 40

  # Scale to GNNM-01-25M on GPU with custom multilingual `.jsonl` / `.txt` corpus directory
  .venv312/bin/python scripts/train_gnnm.py --preset GNNM-01-25M --corpus-dir ./data/corpus --vocab-size 8192 --steps 5000 --sft-steps 500
"""

import argparse
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from gajahmada_ai.gnnm import GNNM_PRESETS, GNNMBPETokenizer, GNNMCorpusBuilder
from gajahmada_ai.gnnm.trainer import GNNMTrainer
from gajahmada_ai.neural.pytorch_backend import PyTorchNativeBackend


def main():
    parser = argparse.ArgumentParser(description="Train GajahMada Native Neural Model (GNNM) from scratch")
    parser.add_argument("--preset", default="GNNM-01-Tiny", choices=list(GNNM_PRESETS.keys()))
    parser.add_argument("--corpus-dir", default="", help="Optional directory of .txt/.jsonl multilingual files")
    parser.add_argument("--jsonl", default="", help="Optional single .jsonl corpus file")
    parser.add_argument("--vocab-size", type=int, default=512)
    parser.add_argument("--seq-len", type=int, default=64)
    parser.add_argument("--steps", type=int, default=80)
    parser.add_argument("--sft-steps", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--output", default="", help="Output checkpoint directory")
    args = parser.parse_args()

    out_dir = args.output or os.path.join(ROOT, "models", args.preset.lower())

    corpus = GNNMCorpusBuilder()
    seed_n = corpus.load_seed_multilingual_corpus()
    extra_n = 0
    if args.corpus_dir:
        extra_n += corpus.load_from_directory(args.corpus_dir)
    if args.jsonl:
        extra_n += corpus.load_from_jsonl(args.jsonl)
    print(f"[Corpus] Seed={seed_n}, External={extra_n}, Total Unique={len(corpus.documents)}")

    tok = GNNMBPETokenizer()
    tok_stats = tok.train(corpus.texts(), target_vocab_size=args.vocab_size, min_frequency=2)
    print(f"[Tokenizer] {tok_stats}")

    pairs = corpus.pack_token_blocks(tok, seq_len=args.seq_len)
    split_idx = max(1, int(len(pairs) * 0.9))
    train_pairs = pairs[:split_idx]
    val_pairs = pairs[split_idx:] or pairs[-4:]

    cfg = GNNM_PRESETS[args.preset]
    cfg.vocab_size = tok.vocab_size
    cfg.max_seq_len = max(cfg.max_seq_len, args.seq_len)

    trainer = GNNMTrainer(cfg, tok, lr=args.lr)
    pre_metrics = trainer.train_steps(train_pairs, steps=args.steps, batch_size=args.batch_size)
    val_metrics = trainer.evaluate_loss(val_pairs)
    print(f"[Pretrain] {pre_metrics} | [Val] {val_metrics}")

    sft_metrics = None
    if args.sft_steps > 0:
        sft_examples = corpus.build_seed_instruction_examples()
        sft_pairs = corpus.pack_instruction_pairs(tok, sft_examples, seq_len=args.seq_len)
        sft_metrics = trainer.train_sft(sft_pairs, steps=args.sft_steps, batch_size=min(4, len(sft_pairs)))
        print(f"[SFT] {sft_metrics}")

    all_metrics = {"pretrain": pre_metrics, "validation": val_metrics, "sft": sft_metrics}
    trainer.save_checkpoint(out_dir, metrics=all_metrics)
    print(f"[Saved] Checkpoint written to {out_dir}")

    backend = PyTorchNativeBackend(model_path=out_dir)
    assert backend.load_model()
    sample = backend.generate("Siapa kamu?", max_tokens=24, temperature=0.4)
    print(f"[Verify Backend] active={backend.health()['neural_inference_active']} | sample='{sample.text}'")


if __name__ == "__main__":
    main()
