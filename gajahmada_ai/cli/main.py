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
    """GajahMada Sovereign Decentralized Node CLI"""
    pass


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


def main():
    cli()


if __name__ == "__main__":
    main()
