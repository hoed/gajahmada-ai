import os
import tempfile

from gajahmada_ai.gnnm import GNNM_PRESETS, GNNMBPETokenizer, GNNMCorpusBuilder
from gajahmada_ai.gnnm.model import HAS_TORCH


def test_gnnm_config_presets_and_parameter_estimation():
    tiny = GNNM_PRESETS["GNNM-01-Tiny"]
    poc = GNNM_PRESETS["GNNM-01-25M"]
    m100 = GNNM_PRESETS["GNNM-02-100M"]

    assert 500_000 < tiny.estimate_parameters() < 3_000_000
    assert 10_000_000 < poc.estimate_parameters() < 30_000_000
    assert 80_000_000 < m100.estimate_parameters() < 150_000_000
    assert poc.language_profile == "multilingual-balanced-id-en"
    assert poc.intelligence_mode == "hybrid-cognitive"


def test_gnnm_native_bpe_tokenizer_multilingual_lossless():
    corpus = GNNMCorpusBuilder()
    added = corpus.load_seed_multilingual_corpus()
    assert added >= 15

    # Verify SHA-256 deduplication blocks duplicates
    assert corpus.add_document(corpus.texts()[0]) is False

    tok = GNNMBPETokenizer()
    stats = tok.train(corpus.texts(), target_vocab_size=340, min_frequency=2)
    assert stats["merges_learned"] > 10
    assert tok.vocab_size == 7 + 256 + stats["merges_learned"]

    samples = [
        "GajahMada adalah arsitektur kecerdasan berdaulat Nusantara.",
        "Decoder-Only Transformer with RoPE, RMSNorm, and SwiGLU!",
        "الحمد لله رب العالمين — E = mc² | def solve(x): return x * 2",
    ]
    for s in samples:
        ids = tok.encode(s, add_bos=True, add_eos=True)
        assert ids[0] == tok.bos_id and ids[-1] == tok.eos_id
        assert tok.decode(ids, skip_special_tokens=True) == s

    # Verify persistence round-trip
    with tempfile.TemporaryDirectory() as tmpdir:
        tok_path = os.path.join(tmpdir, "tokenizer.json")
        tok.save(tok_path)
        reloaded = GNNMBPETokenizer.load(tok_path)
        assert reloaded.vocab_size == tok.vocab_size
        assert reloaded.encode(samples[0]) == tok.encode(samples[0])

    # Verify sequence packing for causal LM pretraining
    pairs = corpus.pack_token_blocks(tok, seq_len=32)
    assert len(pairs) >= 5
    x0, y0 = pairs[0]
    assert len(x0) == 32 and len(y0) == 32
    assert x0[1:] == y0[:-1]


def test_gnnm_transformer_forward_if_torch_available():
    if not HAS_TORCH:
        return
    from gajahmada_ai.gnnm.model import GNNMTransformer, torch

    cfg = GNNM_PRESETS["GNNM-01-Tiny"]
    model = GNNMTransformer(cfg)
    x = torch.randint(0, cfg.vocab_size, (2, 16))
    y = torch.randint(0, cfg.vocab_size, (2, 16))
    logits, loss = model(x, targets=y)
    assert logits.shape == (2, 16, cfg.vocab_size)
    assert loss is not None and float(loss.item()) > 0
    # Verify proper std=0.02 initialization keeps initial CE loss near ln(vocab_size) (~6.9)
    assert float(loss.item()) < 12.0


def test_gnnm_jsonl_corpus_and_sft_prompt_masking():
    import json

    corpus = GNNMCorpusBuilder()
    with tempfile.TemporaryDirectory() as tmpdir:
        jpath = os.path.join(tmpdir, "sample.jsonl")
        with open(jpath, "w", encoding="utf-8") as f:
            f.write(json.dumps({"text": "GajahMada menyatukan Nusantara dengan Sumpah Palapa 1336.", "lang": "id"}) + "\n")
            f.write(json.dumps({"text": "GajahMada menyatukan Nusantara dengan Sumpah Palapa 1336.", "lang": "id"}) + "\n")
            f.write(json.dumps({"text": "Sovereign distributed AI uses Ed25519 cryptographic identities.", "lang": "en"}) + "\n")
        added = corpus.load_from_jsonl(jpath)
        assert added == 2  # duplicate line automatically deduplicated

    tok = GNNMBPETokenizer()
    tok.train(corpus.texts(), target_vocab_size=300, min_frequency=2)
    sft_examples = corpus.build_seed_instruction_examples()
    sft_pairs = corpus.pack_instruction_pairs(tok, sft_examples, seq_len=64)
    assert len(sft_pairs) == len(sft_examples)
    x0, y0 = sft_pairs[0]
    assert len(x0) == 64 and len(y0) == 64
    # Prompt tokens are masked with -100 while assistant tokens have valid >= 0 targets
    assert y0[0] == -100
    assert any(t >= 0 for t in y0)
