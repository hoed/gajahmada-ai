#!/usr/bin/env python3
"""
Stage 5 — Open Reproducible Benchmark Suite (Stages 1-5).
Produces a verified, honest JSON benchmark report at `docs/BENCHMARK_REPORT.json`.
"""
import json
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from fastapi.testclient import TestClient

from gajahmada_ai.api.server import app, get_runtime_context
from gajahmada_ai.cognition.reasoning import CognitiveReasoningPipeline
from gajahmada_ai.reasoning.symbolic import SymbolicReasoner
from scripts.run_gip_network_simulation import (
    run_3_node_verification,
    run_10_node_mesh_verification,
    run_100_node_quorum_stress_test,
)
from sdk.gajahmada_sdk import GajahMadaClient


def run_all_benchmarks():
    t_start = time.perf_counter()

    # Stage 1: Check native GNNM checkpoint metadata
    gnnm_checkpoints = {}
    models_dir = os.path.join(ROOT, "models")
    if os.path.isdir(models_dir):
        for name in sorted(os.listdir(models_dir)):
            cfg_p = os.path.join(models_dir, name, "config.json")
            met_p = os.path.join(models_dir, name, "metrics.json")
            if os.path.exists(cfg_p):
                with open(cfg_p, "r", encoding="utf-8") as f:
                    cfg = json.load(f)
                metrics = {}
                if os.path.exists(met_p):
                    with open(met_p, "r", encoding="utf-8") as f:
                        metrics = json.load(f)
                gnnm_checkpoints[name] = {
                    "estimated_parameters": cfg.get("estimated_parameters"),
                    "vocab_size": cfg.get("vocab_size"),
                    "n_layers": cfg.get("n_layers"),
                    "d_model": cfg.get("d_model"),
                    "metrics": metrics,
                }

    # Stage 2: Neural-Off Cognitive Engine latency & accuracy
    t2 = time.perf_counter()
    pipe = CognitiveReasoningPipeline(model_router=None, symbolic_reasoner=SymbolicReasoner())
    math_cases = [("Hitung 12 * 12", "144"), ("Hitung 500 - 125", "375"), ("Hitung 81 / 9 + 11", "20")]
    passed_s2 = sum(1 for q, expected in math_cases if expected in pipe.execute(q).conclusion)
    stage2_ms = round((time.perf_counter() - t2) * 1000, 2)

    # Stage 3: 3 / 10 / 100-node GIP Distributed Intelligence
    s3_3 = run_3_node_verification()
    s3_10 = run_10_node_mesh_verification()
    s3_100 = run_100_node_quorum_stress_test()

    # Stage 5: SDK Round-trip benchmark
    t5 = time.perf_counter()
    rt = get_runtime_context()
    sdk_client = GajahMadaClient(api_key=rt["auth"].admin_api_key, http_client=TestClient(app))
    health = sdk_client.health()
    ask_res = sdk_client.ask("Hitung 25 * 4 + 10")
    stage5_ms = round((time.perf_counter() - t5) * 1000, 2)

    report = {
        "benchmark_suite": "GajahMada 2.0 Open Ecosystem Benchmark (Stages 1-5)",
        "generated_at_epoch": round(time.time(), 3),
        "total_elapsed_ms": round((time.perf_counter() - t_start) * 1000, 2),
        "stage_1_gnnm_checkpoints": gnnm_checkpoints,
        "stage_2_neural_off_cognition": {
            "passed": passed_s2,
            "total": len(math_cases),
            "accuracy": round(passed_s2 / len(math_cases), 4),
            "elapsed_ms": stage2_ms,
        },
        "stage_3_gip_distributed_intelligence": {
            "3_nodes": s3_3,
            "10_nodes": s3_10,
            "100_nodes": s3_100,
        },
        "stage_5_sdk_interoperability": {
            "health_status": health.get("status"),
            "ledger_type": health.get("ledger", {}).get("type"),
            "ask_verified": "110" in ask_res.get("conclusion", ""),
            "elapsed_ms": stage5_ms,
        },
    }

    out_path = os.path.join(ROOT, "docs", "BENCHMARK_REPORT.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report, indent=2))
    return report


if __name__ == "__main__":
    run_all_benchmarks()
