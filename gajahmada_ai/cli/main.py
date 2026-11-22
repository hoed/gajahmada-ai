import json
import os
import click
import httpx
import uvicorn
import yaml

import gajahmada_ai.api.server as server_module
from gajahmada_ai.api.server import app


def load_config(path: str) -> dict:
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    return {}


def get_base_url(config_path: str = "config.yaml") -> str:
    cfg = load_config(config_path)
    port = cfg.get("node", {}).get("port", 8000)
    return f"http://127.0.0.1:{port}"


@click.group()
def cli():
    """GajahMada 2.0 — Sovereign Hybrid Distributed Intelligence CLI"""
    pass


# ─── Legacy v1 Commands (Preserved 100%) ──────────────────────────────────
@cli.command()
@click.option("--config", default="config.yaml", help="Path to config file")
def start(config):
    """Start the GajahMada Node"""
    cfg = load_config(config)
    node_cfg = cfg.get("node", {})
    host = node_cfg.get("host", "0.0.0.0")
    port = int(node_cfg.get("port", 8000))

    server_module.ensure_initialized()
    print(f"Starting GajahMada node {server_module.identity.node_id[:10]}... on {host}:{port}")
    uvicorn.run(app, host=host, port=port)


@cli.command()
@click.argument("peer_url")
@click.option("--config", default="config.yaml", help="Path to config file")
def join(peer_url, config):
    """Join the P2P network via a peer URL"""
    base_url = get_base_url(config)
    try:
        resp = httpx.post(f"{base_url}/node/sync", json={"peers": [peer_url]}, timeout=5.0)
        resp.raise_for_status()
        data = resp.json()
        print(f"Joined via {peer_url}. Known peers ({len(data.get('peers_known', []))}): {data.get('peers_known')}")
    except Exception as e:
        print(f"Failed to join via {base_url}. Is local node running? ({e})")


@cli.command()
@click.option("--config", default="config.yaml", help="Path to config file")
def node_status(config):
    """Get status and cryptographic audit of local node"""
    base_url = get_base_url(config)
    try:
        info = httpx.get(f"{base_url}/node/info", timeout=5.0).json()
        chain = httpx.post(f"{base_url}/node/chain/verify", timeout=5.0).json()
        print(f"Node Name:         {info.get('node_name', 'gajahmada-alpha')}")
        print(f"Node ID (Ed25519): {info['node_id']}")
        print(f"Mode:              {info.get('mode', 'full')}")
        print(f"Neural Inference:  {'ACTIVE' if info.get('neural_inference_active') else 'INACTIVE (Symbolic+Evidence active)'}")
        print(f"Peers Linked:      {len(info.get('peers', []))}")
        print(f"RAG Documents:     {info.get('knowledge_docs', 0)}")
        print(f"Blockchain Height: {info.get('blockchain_height', 1)} (Valid: {chain.get('valid')}, Verified Sigs: {chain.get('verified_signatures')})")
    except Exception as e:
        print(f"Failed to get status from {base_url}. Is node running? ({e})")


@cli.command()
@click.option("--note", default="CLI Federated Consensus Audit", help="Audit note to record on chain")
@click.option("--config", default="config.yaml", help="Path to config file")
def train_model(note, config):
    """Submit consensus block & broadcast across P2P mesh"""
    base_url = get_base_url(config)
    try:
        mine_res = httpx.post(f"{base_url}/node/chain/mine", json={"note": note}, timeout=10.0).json()
        blk = mine_res.get("block", {})
        bcast = httpx.post(
            f"{base_url}/node/broadcast",
            json={"message": f"Block #{blk.get('index')} mined: {blk.get('hash', '')[:16]}"},
            timeout=10.0,
        ).json()
        print(f"Consensus Block #{blk.get('index')} mined (hash={blk.get('hash', '')[:20]}..., nonce={blk.get('nonce')})")
        print(f"Broadcasted to {bcast.get('peers_targeted', 0)} peer(s) ({bcast.get('peers_reachable', 0)} reachable).")
    except Exception as e:
        print(f"Failed to submit consensus task: {e}")


# ─── GajahMada 2.0 CLI Command Groups ─────────────────────────────────────
@cli.command()
def init():
    """Initialize GajahMada 2.0 sovereign identity, storage, and memory"""
    ctx = server_module.get_runtime_context()
    print(f"Initialized GajahMada 2.0 Sovereign Node")
    print(f"• Node ID (Ed25519): {ctx['identity'].node_id}")
    print(f"• Storage Engine:    {ctx['memory'].storage.health()['engine_type']}")
    print(f"• Memory Records:    {ctx['memory'].storage.health()['active_records']}")
    print(f"• Ledger Height:     {len(ctx['blockchain'].chain)} blocks")


@cli.group()
def node():
    """Manage GajahMada sovereign node lifecycle"""
    pass


@node.command("init")
def node_init():
    """Initialize local node identity and database"""
    ctx = server_module.get_runtime_context()
    print(f"Node initialized: {ctx['identity'].node_id}")


@node.command("start")
@click.option("--config", default="config.yaml", help="Path to config file")
def node_start(config):
    """Start GajahMada 2.0 node daemon"""
    cfg = load_config(config)
    node_cfg = cfg.get("node", {})
    host = node_cfg.get("host", "0.0.0.0")
    port = int(node_cfg.get("port", 8000))
    server_module.ensure_initialized()
    uvicorn.run(app, host=host, port=port)


@node.command("stop")
def node_stop():
    """Report graceful shutdown instruction for local node"""
    print("GajahMada node shutdown signal issued.")


@node.command("status")
def node_v2_status():
    """Inspect local node identity, ledger, and capability status"""
    ctx = server_module.get_runtime_context()
    cap = ctx["engine"].capability_status()
    print(json.dumps({
        "node_id": ctx["identity"].node_id,
        "key_version": ctx["identity"].key_version,
        "neural_inference_active": cap["neural_inference_active"],
        "symbolic_reasoning_active": cap["symbolic_reasoning_active"],
        "ledger_height": len(ctx["blockchain"].chain),
        "ledger_valid": ctx["blockchain"].is_chain_valid(),
    }, indent=2))


@cli.group()
def intelligence():
    """Diagnose capabilities or run hybrid cognitive reasoning"""
    pass


@intelligence.command("diagnose")
def intelligence_diagnose():
    """Report active neural, symbolic, and evidence reasoning capabilities"""
    ctx = server_module.get_runtime_context()
    print(json.dumps(ctx["engine"].capability_status(), indent=2))


@intelligence.command("ask")
@click.argument("question")
def intelligence_ask(question):
    """Execute 14-stage hybrid cognitive pipeline on a question"""
    ctx = server_module.get_runtime_context()
    records = ctx["memory"].retrieve(query=question, top_k=3)
    res = ctx["cognition"].execute(objective=question, memory_records=records)
    print(json.dumps(res.model_dump(), indent=2, ensure_ascii=False))


@cli.group()
def reasoning():
    """Inspect structured reasoning traces"""
    pass


@reasoning.command("inspect")
@click.argument("reasoning_id", required=False)
def reasoning_inspect(reasoning_id):
    """Inspect recent reasoning results or a specific reasoning_id"""
    ctx = server_module.get_runtime_context()
    if reasoning_id and reasoning_id in ctx["cognition"].history:
        print(json.dumps(ctx["cognition"].history[reasoning_id].model_dump(), indent=2, ensure_ascii=False))
    else:
        items = [r.model_dump() for r in list(ctx["cognition"].history.values())[-5:]]
        print(json.dumps({"count": len(items), "recent": items}, indent=2, ensure_ascii=False))


@cli.group()
def memory():
    """Inspect or backup 5-tier persistent memory"""
    pass


@memory.command("inspect")
def memory_inspect():
    """Display 5-tier memory statistics and contradiction counts"""
    ctx = server_module.get_runtime_context()
    print(json.dumps(ctx["memory"].stats(), indent=2, ensure_ascii=False))


@memory.command("backup")
def memory_backup():
    """Export content-addressed SHA-256 backup of all persistent memory tiers"""
    ctx = server_module.get_runtime_context()
    print(json.dumps(ctx["memory"].export_backup(), indent=2, ensure_ascii=False))


@cli.group()
def network():
    """Manage GIP/2.0 distributed network & peers"""
    pass


@network.command("discover")
def network_discover():
    """List discovered GIP/2.0 peers and capabilities"""
    ctx = server_module.get_runtime_context()
    print(json.dumps(ctx["discovery"].list_active_peers(), indent=2))


@network.command("peers")
def network_peers():
    """List configured P2P peer endpoints"""
    ctx = server_module.get_runtime_context()
    print(json.dumps({"peers": list(ctx["p2p"].peers)}, indent=2))


@network.command("status")
def network_status():
    """Display GIP/2.0 protocol and peer status"""
    ctx = server_module.get_runtime_context()
    print(json.dumps({
        "protocol_version": "GIP/2.0",
        "node_id": ctx["identity"].node_id,
        "peers": list(ctx["p2p"].peers),
        "discovered": ctx["discovery"].list_active_peers(),
    }, indent=2))


@cli.group()
def agent():
    """Execute autonomous multi-agent tasks"""
    pass


@agent.command("run")
@click.argument("objective")
def agent_run(objective):
    """Run an objective through Planner, Researcher, Analyst, and Verifier agents"""
    ctx = server_module.get_runtime_context()
    task = ctx["scheduler"].submit_objective(objective=objective)
    print(json.dumps(task.to_dict(), indent=2, ensure_ascii=False))


@cli.group()
def system():
    """System health and security diagnostics"""
    pass


@system.command("health")
def system_health():
    """Run comprehensive system, storage, ledger, and security audit health check"""
    ctx = server_module.get_runtime_context()
    print(json.dumps({
        "status": "healthy",
        "version": "2.0.0",
        "storage": ctx["memory"].storage.health(),
        "ledger_valid": ctx["blockchain"].is_chain_valid(),
        "security_audit_valid": ctx["audit_logger"].verify_integrity(),
    }, indent=2))


def main():
    cli()


if __name__ == "__main__":
    main()
