from gajahmada_ai.cognition.reasoning import CognitiveReasoningPipeline
from gajahmada_ai.neural.router import ModelRouter
from gajahmada_ai.reasoning.symbolic import SymbolicReasoner


def test_4_symbolic_reasoning_operates_independently_of_llm():
    """TEST 4: Symbolic reasoning operates independently of the LLM."""
    reasoner = SymbolicReasoner()

    # 1. Mathematical & algebraic evaluation
    res_math = reasoner.reason("Berapa hasil sqrt(144) + 25 * 4?")
    assert res_math.applicable is True
    assert res_math.math_value == 112.0

    # 2. Propositional forward-chaining rule deduction
    res_rules = reasoner.reason(
        "Verify block compliance",
        facts=["has_ed25519_signature", "signature_verified", "pow_hash_valid", "prev_hash_linked"],
    )
    assert res_rules.applicable is True
    assert "cryptographically_authenticated" in res_rules.derived_facts
    assert "ledger_block_valid" in res_rules.derived_facts

    # 3. Categorical syllogism & Modus Ponens
    res_syllogism = reasoner.reason(
        "All sovereign_nodes are cryptographic_actors. All cryptographic_actors are verifiable_peers. gajahmada_alpha is a sovereign_nodes."
    )
    assert res_syllogism.applicable is True
    assert any("gajahmada_alpha is verifiable_peers" in d for d in res_syllogism.derived_facts)


def test_5_cognitive_pipeline_produces_structured_reasoning_results():
    """TEST 5: The cognitive pipeline can produce structured reasoning results."""
    router = ModelRouter(gguf_model_path="./models/missing.gguf")
    pipeline = CognitiveReasoningPipeline(model_router=router)

    result = pipeline.execute(
        objective="Calculate 256 / 16 + 10 and explain verification",
        memory_records=[
            {
                "id": "mem-test-01",
                "content": "Deterministic arithmetic verification requires exact AST evaluation.",
                "source": "symbolic_proof",
                "verification_status": "VERIFIED",
            }
        ],
    )
    assert result.reasoning_id.startswith("rsn-")
    assert result.objective
    assert result.interpretation
    assert len(result.assumptions) >= 1
    assert len(result.hypotheses) >= 1
    assert len(result.reasoning_steps) == 11
    assert 0.0 <= result.confidence <= 1.0
    assert 0.0 <= result.uncertainty <= 1.0
    assert result.verification_status == "VERIFIED"
    assert len(result.epistemic_breakdown.verified_facts) >= 1


def test_6_important_conclusions_contain_evidence_references():
    """TEST 6: Important conclusions contain evidence references."""
    pipeline = CognitiveReasoningPipeline()
    evidence_records = [
        {
            "id": "mem-palapa-1336",
            "content": "Mahapatih Gajah Mada mengucapkan Sumpah Palapa pada tahun 1336 M di Majapahit untuk menyatukan Nusantara.",
            "source": "sovereign_core_corpus",
            "verification_status": "VERIFIED",
        }
    ]
    result = pipeline.execute(
        objective="Kapan Mahapatih Gajah Mada mengucapkan Sumpah Palapa?",
        memory_records=evidence_records,
    )
    assert result.verification_status == "VERIFIED"
    assert "mem-palapa-1336" in result.memory_references
    assert len(result.evidence) == 1
    assert result.evidence[0].evidence_id == "mem-palapa-1336"
    assert result.evidence[0].provenance_hash
