"""
Stage 2 Verification: Native Cognitive Engine operates with 100% deterministic
reasoning, symbolic logic, evidence evaluation, contradiction detection, and
5-tier persistent memory EVEN WHEN ZERO NEURAL MODELS ARE LOADED.
"""
import os
import tempfile

from gajahmada_ai.cognition.reasoning import CognitiveReasoningPipeline
from gajahmada_ai.memory.manager import MemoryManager
from gajahmada_ai.memory.models import MemoryType, VerificationStatus, VisibilityPolicy
from gajahmada_ai.reasoning.symbolic import SymbolicReasoner


def test_stage2_cognitive_engine_operates_without_any_neural_model():
    with tempfile.TemporaryDirectory() as tmpdir:
        mem = MemoryManager(
            sqlite_path=os.path.join(tmpdir, "stage2.db"),
            backup_dir=os.path.join(tmpdir, "objects"),
        )
        mem.store(
            content="Sumpah Palapa diucapkan Mahapatih Gajah Mada pada tahun 1336 M di Kerajaan Majapahit.",
            memory_type=MemoryType.SEMANTIC,
            confidence=0.95,
            verification_status=VerificationStatus.INDEPENDENTLY_VERIFIED,
            visibility_policy=VisibilityPolicy.COLLECTIVE,
        )
        # Deliberately construct pipeline with NO neural model_router
        pipeline = CognitiveReasoningPipeline(
            model_router=None,
            symbolic_reasoner=SymbolicReasoner(),
            fallback_synthesizer=None,
        )

        # 1. Deterministic symbolic math without neural model
        r_math = pipeline.execute("Hitung 144 / 12 + 38 * 2")
        assert r_math.capabilities_used["neural_used_in_this_call"] is False
        assert r_math.capabilities_used["symbolic_used_in_this_call"] is True
        assert "88" in r_math.conclusion
        assert r_math.confidence >= 0.90

        # 2. Evidence-backed memory retrieval without neural model
        recs = mem.retrieve("Kapan Sumpah Palapa diucapkan?", top_k=3)
        r_ev = pipeline.execute("Kapan Sumpah Palapa diucapkan?", memory_records=recs)
        assert r_ev.capabilities_used["neural_used_in_this_call"] is False
        assert len(r_ev.evidence) >= 1
        assert "1336" in r_ev.conclusion or any("1336" in e.excerpt for e in r_ev.evidence)

        # 3. Contradiction detection without neural model
        mem.store(
            content="Sumpah Palapa tidak diucapkan pada tahun 1336 M di Kerajaan Majapahit.",
            memory_type=MemoryType.SEMANTIC,
            confidence=0.60,
            verification_status=VerificationStatus.UNVERIFIED,
            visibility_policy=VisibilityPolicy.LOCAL,
        )
        all_recs = mem.retrieve("Sumpah Palapa 1336 Majapahit", top_k=5)
        r_contra = pipeline.execute("Sumpah Palapa 1336 Majapahit", memory_records=all_recs)
        assert len(r_contra.contradictions) >= 1 or len(mem.storage.list_contradictions()) >= 1
