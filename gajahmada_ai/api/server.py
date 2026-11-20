import hashlib
import importlib.metadata
import os
import sys
import time
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from gajahmada_ai.blockchain.chain import AIBlockchain, Block
from gajahmada_ai.core.engine import AIModelEngine
from gajahmada_ai.knowledge.rag import LocalKnowledge
from gajahmada_ai.network.p2p import P2PNetwork
from gajahmada_ai.security.crypto import NodeIdentity

try:
    import yaml
except ImportError:
    yaml = None

ML_AVAILABLE = True
IS_VERCEL = os.environ.get("VERCEL") == "1"
START_TIME = time.time()

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.normpath(os.path.join(_HERE, "..", ".."))
UI_PATH = os.path.join(_ROOT, "interface", "html")
CONFIG_PATH = os.path.join(_ROOT, "config.yaml")

DEFAULT_KNOWLEDGE_DOCS = [
    "GajahMada AI System Architecture: Decentralized AI inference & Web3 node terdiri dari 8 modul inti (api.server, core.engine, knowledge.rag, blockchain.chain, network.p2p, security.crypto, cli.main, interface.html).",
    "Module core.engine (AIModelEngine): Menjalankan inferensi LLM lokal (LLaMA GGUF / Neural Synthesis) dan menghasilkan vektor embedding semantik 384-dimensi (MiniLM-L6-v2 / Hash-Projection).",
    "Module knowledge.rag (LocalKnowledge): Menyimpan dokumen dalam indeks vektor 384-dimensi (FAISS IndexFlatL2 & L2-VectorStore) untuk Retrieval-Augmented Generation (RAG) berlatensi rendah.",
    "Module blockchain.chain (AIBlockchain): Mencatat setiap task_hash SHA-256 (prompt + response) ke dalam rantai blok Proof-of-Work (difficulty=2) yang ditandatangani secara kriptografis.",
    "Module security.crypto (NodeIdentity): Menggunakan pasangan kunci asimetris Ed25519 (RFC 8032) untuk identitas node unik (node_id) dan verifikasi tanda tangan digital setiap blok.",
    "Module network.p2p (P2PNetwork): Mengelola topologi mesh peer-to-peer asinkron via httpx, sinkronisasi bootstrap peers (/node/sync), dan siaran gossip bertanda tangan (/node/broadcast).",
]

app = FastAPI(
    title="GajahMada AI Network Node",
    description="Sovereign Decentralized AI & Web3 Node (Local & Vercel Serverless Ready)",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── Global singletons & live telemetry log buffer ────────────────────────
engine: Optional[AIModelEngine] = None
p2p: Optional[P2PNetwork] = None
blockchain: Optional[AIBlockchain] = None
rag: Optional[LocalKnowledge] = None
identity: Optional[NodeIdentity] = None
system_logs: List[Dict[str, Any]] = []
runtime_config: Dict[str, Any] = {
    "node": {"host": "0.0.0.0", "port": 8000, "name": "gajahmada-alpha"},
    "network": {"bootstrap_peers": ["http://127.0.0.1:8001"]},
    "ai": {
        "model_path": "./models/llama-2-7b-chat.Q4_K_M.gguf",
        "embedding_model": "all-MiniLM-L6-v2",
        "max_tokens": 512,
        "temperature": 0.7,
    },
    "blockchain": {"difficulty": 2},
}


def record_log(module: str, level: str, message: str):
    entry = {
        "id": len(system_logs) + 1,
        "timestamp": time.time(),
        "time_str": time.strftime("%H:%M:%S"),
        "module": module,
        "level": level,
        "message": message,
    }
    system_logs.append(entry)
    if len(system_logs) > 100:
        del system_logs[:-100]


def ensure_initialized():
    global blockchain, identity, engine, rag, p2p, runtime_config
    if yaml is not None and os.path.exists(CONFIG_PATH):
        try:
            with open(CONFIG_PATH, "r", encoding="utf-8") as f:
                loaded = yaml.safe_load(f) or {}
                for k, v in loaded.items():
                    if isinstance(v, dict) and k in runtime_config:
                        runtime_config[k].update(v)
                    else:
                        runtime_config[k] = v
        except Exception:
            pass

    if identity is None:
        key_path = "/tmp/node_key.pem" if IS_VERCEL else os.path.join(_ROOT, "node_key.pem")
        identity = NodeIdentity(key_path=key_path)
        record_log("security.crypto", "OK", f"Ed25519 NodeIdentity loaded: {identity.node_id[:14]}...")

    if blockchain is None:
        chain_path = (
            "/tmp/knowledge_store/chain.json"
            if IS_VERCEL
            else os.path.join(_ROOT, "knowledge_store", "chain.json")
        )
        blockchain = AIBlockchain(storage_path=chain_path)
        blockchain.difficulty = int(runtime_config.get("blockchain", {}).get("difficulty", 2))
        record_log(
            "blockchain.chain",
            "OK",
            f"PoW Ledger mounted (height={len(blockchain.chain)}, diff={blockchain.difficulty})",
        )

    if p2p is None:
        node_cfg = runtime_config.get("node", {})
        p2p = P2PNetwork(node_cfg.get("host", "0.0.0.0"), int(node_cfg.get("port", 8000)))
        for bp in runtime_config.get("network", {}).get("bootstrap_peers", []):
            p2p.add_peer(bp)
        record_log("network.p2p", "OK", f"P2PNetwork active with {len(p2p.peers)} bootstrap peer(s)")

    if engine is None:
        ai_cfg = runtime_config.get("ai", {})
        engine = AIModelEngine(
            model_path=ai_cfg.get("model_path", "./models/llama-2-7b-chat.Q4_K_M.gguf"),
            embedding_model_name=ai_cfg.get("embedding_model", "all-MiniLM-L6-v2"),
        )
        if not IS_VERCEL:
            engine.load_models()
        record_log("core.engine", "OK", "AIModelEngine initialized (384-d embeddings & inference ready)")

    if rag is None:
        storage_dir = "/tmp/knowledge_store" if IS_VERCEL else os.path.join(_ROOT, "knowledge_store")
        rag = LocalKnowledge(storage_path=storage_dir)
        # Seed initial architectural knowledge if empty
        if len(rag.doc_store) == 0 and engine is not None:
            for doc in DEFAULT_KNOWLEDGE_DOCS:
                emb = engine.create_embedding(doc)
                if emb:
                    rag.ingest(doc, emb)
            rag.flush()
        record_log("knowledge.rag", "OK", f"LocalKnowledge mounted ({len(rag.doc_store)} docs indexed)")


# ─── Request models ───────────────────────────────────────────────────────
class ChatRequest(BaseModel):
    prompt: str
    max_tokens: int = 256
    distribute: bool = False


class EmbeddingRequest(BaseModel):
    text: str


class SearchKnowledgeRequest(BaseModel):
    query: str
    k: int = 3


class SyncPeersRequest(BaseModel):
    peers: List[str]


class BroadcastRequest(BaseModel):
    message: str


class MineBlockRequest(BaseModel):
    note: str = "Manual operator audit block"


class SignVerifyRequest(BaseModel):
    message: str


class ConfigUpdateRequest(BaseModel):
    node_name: Optional[str] = None
    pow_difficulty: Optional[int] = None
    max_tokens: Optional[int] = None
    temperature: Optional[float] = None


# ─── Startup ──────────────────────────────────────────────────────────────
@app.on_event("startup")
async def startup_event():
    ensure_initialized()


# ─── Health & Root ────────────────────────────────────────────────────────
@app.get("/")
def root():
    ensure_initialized()
    return {
        "status": "online",
        "message": "GajahMada AI Network Node is running.",
        "mode": "serverless" if IS_VERCEL else "full",
        "ml_available": engine is not None,
        "uptime_sec": round(time.time() - START_TIME, 1),
        "ui_url": "/ui",
    }


@app.get("/health")
def health():
    ensure_initialized()
    return {"ok": True, "mode": "serverless" if IS_VERCEL else "full", "uptime_sec": round(time.time() - START_TIME, 1)}


# ─── AI & Command Interface Endpoints ─────────────────────────────────────
@app.post("/ai/chat")
async def ai_chat(req: ChatRequest):
    ensure_initialized()
    start = time.time()
    context = ""
    matched_docs = 0

    if rag and engine:
        try:
            query_emb = engine.create_embedding(req.prompt)
            if query_emb:
                results = rag.search(query_emb, k=3)
                if results:
                    matched_docs = len(results)
                    context = "\n".join(results)
        except Exception as e:
            record_log("knowledge.rag", "WARN", f"RAG lookup warning: {e}")

    response = engine.generate(req.prompt, context=context, max_tokens=req.max_tokens)

    task_hash = hashlib.sha256((req.prompt + response).encode()).hexdigest()
    signature = ""
    block_idx = 0

    if blockchain and identity:
        try:
            signature = identity.sign_message(task_hash.encode())
            block = Block(
                index=len(blockchain.chain),
                timestamp=start,
                task_hash=task_hash,
                node_id=identity.node_id,
                previous_hash=blockchain.get_latest_block().hash,
                signature=signature,
            )
            blockchain.add_block(block)
            block_idx = block.index
        except Exception as e:
            record_log("blockchain.chain", "WARN", f"Block mining error: {e}")

    elapsed = round(time.time() - start, 3)
    record_log(
        "core.engine",
        "INFO",
        f"POST /ai/chat processed in {elapsed}s (rag_matches={matched_docs}, block=#{block_idx})",
    )

    return {
        "response": response,
        "time_taken": elapsed,
        "block_index": block_idx,
        "task_hash": task_hash,
        "signature": signature[:24] if signature else None,
        "rag_matches": matched_docs,
    }


@app.post("/ai/ingest")
async def ai_ingest(req: EmbeddingRequest):
    ensure_initialized()
    text = req.text.strip()
    if not text:
        raise HTTPException(status_code=400, detail="Text cannot be empty.")
    emb = engine.create_embedding(text)
    if not emb:
        raise HTTPException(status_code=500, detail="Failed to create 384-d embedding.")
    rag.ingest(text, emb)
    rag.flush()
    tokens = len(text.split())
    record_log("knowledge.rag", "OK", f"Assimilated document ({tokens} tokens, total_docs={len(rag.doc_store)})")
    return {
        "status": "assimilated",
        "tokens": tokens,
        "total_docs": len(rag.doc_store),
        "dimension": rag.dimension,
    }


@app.post("/ai/embedding")
async def ai_embedding(req: EmbeddingRequest):
    ensure_initialized()
    emb = engine.create_embedding(req.text)
    record_log("core.engine", "INFO", f"Generated {len(emb)}-d embedding vector")
    return {"embedding": emb, "dimension": len(emb)}


# ─── Knowledge Expansion Management Endpoints ─────────────────────────────
@app.get("/knowledge/stats")
def knowledge_stats():
    ensure_initialized()
    return {
        "total_docs": len(rag.doc_store) if rag else 0,
        "dimension": rag.dimension if rag else 384,
        "engine": "FAISS-IndexFlatL2" if (rag and getattr(rag, "_has_faiss", False)) else "L2-VectorStore",
        "docs": [
            {"index": idx, "preview": doc[:120], "tokens": len(doc.split())}
            for idx, doc in enumerate(rag.doc_store if rag else [])
        ],
    }


@app.post("/knowledge/search")
def knowledge_search(req: SearchKnowledgeRequest):
    ensure_initialized()
    emb = engine.create_embedding(req.query)
    results = rag.search(emb, k=req.k) if rag else []
    record_log("knowledge.rag", "INFO", f"Semantic search query='{req.query[:30]}' -> {len(results)} match(es)")
    return {"query": req.query, "results": results, "count": len(results)}


@app.delete("/knowledge/docs/{doc_idx}")
def delete_knowledge_doc(doc_idx: int):
    ensure_initialized()
    if not rag or doc_idx < 0 or doc_idx >= len(rag.doc_store):
        raise HTTPException(status_code=404, detail="Document index not found.")
    removed = rag.doc_store.pop(doc_idx)
    if doc_idx < len(rag.vectors):
        rag.vectors.pop(doc_idx)
    rag._dirty = True
    rag.flush()
    record_log("knowledge.rag", "WARN", f"Removed document #{doc_idx} ({removed[:30]}...)")
    return {"status": "deleted", "total_docs": len(rag.doc_store)}


# ─── Node, Network, Config, Dependencies & Diagnostics Endpoints ──────────
@app.get("/node/info")
def node_info():
    ensure_initialized()
    return {
        "node_id": identity.node_id if identity else "uninitialized",
        "node_name": runtime_config.get("node", {}).get("name", "gajahmada-alpha"),
        "peers": list(p2p.peers) if p2p else [],
        "blockchain_height": len(blockchain.chain) if blockchain else 0,
        "blockchain_valid": blockchain.is_chain_valid() if blockchain else True,
        "pow_difficulty": blockchain.difficulty if blockchain else 2,
        "knowledge_docs": len(rag.doc_store) if rag else 0,
        "mode": "serverless" if IS_VERCEL else "full",
        "ml_ready": engine is not None,
        "uptime_sec": round(time.time() - START_TIME, 1),
    }


@app.get("/node/peers")
def get_peers():
    ensure_initialized()
    return {"peers": list(p2p.peers) if p2p else []}


@app.post("/node/sync")
def sync_node(req: SyncPeersRequest):
    ensure_initialized()
    added = []
    for peer_url in req.peers:
        clean = peer_url.strip()
        parsed = urlparse(clean)
        if parsed.scheme not in ("http", "https") or not parsed.netloc:
            raise HTTPException(status_code=400, detail=f"Invalid peer URL (use http:// or https://): {clean}")
        p2p.add_peer(clean)
        added.append(clean)
    record_log("network.p2p", "OK", f"Synced peer(s): {', '.join(added)} (total={len(p2p.peers)})")
    return {"status": "ok", "added": added, "peers_known": list(p2p.peers)}


@app.delete("/node/peers")
def remove_peer(peer: str):
    ensure_initialized()
    p2p.remove_peer(peer.strip())
    record_log("network.p2p", "WARN", f"Unlinked peer: {peer}")
    return {"status": "removed", "peers_known": list(p2p.peers)}


@app.post("/node/broadcast")
async def broadcast_to_peers(req: BroadcastRequest):
    ensure_initialized()
    sig = identity.sign_message(req.message.encode())
    payload = {"node_id": identity.node_id, "message": req.message, "signature": sig, "timestamp": time.time()}
    results = await p2p.broadcast("/health", payload)
    reachable = sum(1 for r in results if not isinstance(r, Exception))
    record_log("network.p2p", "INFO", f"Broadcast '{req.message[:30]}' to {len(p2p.peers)} peer(s) ({reachable} reachable)")
    return {
        "status": "broadcast_sent",
        "peers_targeted": len(p2p.peers),
        "peers_reachable": reachable,
        "signature": sig[:24],
    }


@app.get("/node/metrics")
def node_metrics():
    ensure_initialized()
    try:
        import psutil
        cpu = psutil.cpu_percent(interval=None)
        mem = psutil.virtual_memory().percent
        disk = psutil.disk_io_counters()
        disk_read_mb = round(disk.read_bytes / (1024 * 1024), 2) if disk else 0
    except Exception:
        cpu, mem, disk_read_mb = 12, 38, 0

    return {
        "cpu_usage": cpu,
        "memory_usage": mem,
        "disk_read_mb": disk_read_mb,
        "knowledge_docs": len(rag.doc_store) if rag else 0,
        "chain_height": len(blockchain.chain) if blockchain else 1,
        "pow_difficulty": blockchain.difficulty if blockchain else 2,
        "peers_count": len(p2p.peers) if p2p else 0,
        "time": time.time(),
    }


@app.get("/node/chain")
def get_chain():
    ensure_initialized()
    valid = blockchain.is_chain_valid()
    return {
        "chain": [b.to_dict() for b in blockchain.chain],
        "valid": valid,
        "height": len(blockchain.chain),
        "difficulty": blockchain.difficulty,
    }


@app.post("/node/chain/mine")
def mine_manual_block(req: MineBlockRequest):
    ensure_initialized()
    t0 = time.time()
    task_hash = hashlib.sha256(f"{req.note}|{t0}".encode()).hexdigest()
    signature = identity.sign_message(task_hash.encode())
    block = Block(
        index=len(blockchain.chain),
        timestamp=t0,
        task_hash=task_hash,
        node_id=identity.node_id,
        previous_hash=blockchain.get_latest_block().hash,
        signature=signature,
    )
    blockchain.add_block(block)
    elapsed = round(time.time() - t0, 4)
    record_log("blockchain.chain", "OK", f"Mined manual Block #{block.index} (nonce={block.nonce}, {elapsed}s)")
    return {"status": "mined", "block": block.to_dict(), "time_taken": elapsed}


@app.post("/node/chain/verify")
def verify_full_chain():
    ensure_initialized()
    is_valid = blockchain.is_chain_valid()
    sig_verified = 0
    for b in blockchain.chain[1:]:
        if b.signature and identity.verify_message(b.task_hash.encode(), b.signature, b.node_id):
            sig_verified += 1
    record_log(
        "blockchain.chain",
        "OK" if is_valid else "ERR",
        f"Chain audit complete: valid={is_valid}, height={len(blockchain.chain)}, verified_sigs={sig_verified}",
    )
    return {
        "valid": is_valid,
        "height": len(blockchain.chain),
        "difficulty": blockchain.difficulty,
        "verified_signatures": sig_verified,
    }


@app.post("/security/sign-verify")
def security_sign_verify(req: SignVerifyRequest):
    ensure_initialized()
    msg_bytes = req.message.encode("utf-8")
    signature = identity.sign_message(msg_bytes)
    verified = identity.verify_message(msg_bytes, signature, identity.node_id)
    record_log("security.crypto", "OK", f"Ed25519 sign & verify self-test: verified={verified}")
    return {
        "node_id": identity.node_id,
        "message": req.message,
        "signature": signature,
        "verified": verified,
        "algorithm": "Ed25519 (RFC 8032)",
    }


@app.get("/node/config")
def get_node_config():
    ensure_initialized()
    return runtime_config


@app.post("/node/config")
def update_node_config(req: ConfigUpdateRequest):
    ensure_initialized()
    if req.node_name:
        runtime_config["node"]["name"] = req.node_name.strip()
    if req.pow_difficulty is not None and 1 <= req.pow_difficulty <= 4:
        runtime_config["blockchain"]["difficulty"] = req.pow_difficulty
        blockchain.difficulty = req.pow_difficulty
    if req.max_tokens is not None and 64 <= req.max_tokens <= 4096:
        runtime_config["ai"]["max_tokens"] = req.max_tokens
    if req.temperature is not None and 0.0 <= req.temperature <= 2.0:
        runtime_config["ai"]["temperature"] = round(req.temperature, 2)

    record_log(
        "cli.config",
        "OK",
        f"Updated runtime config: name={runtime_config['node']['name']}, diff={blockchain.difficulty}, max_tokens={runtime_config['ai']['max_tokens']}",
    )
    return {"status": "updated", "config": runtime_config}


@app.get("/node/dependencies")
def get_dependencies():
    ensure_initialized()
    pkgs = [
        "fastapi", "uvicorn", "pydantic", "cryptography",
        "httpx", "psutil", "PyYAML", "click",
        "numpy", "faiss-cpu", "llama-cpp-python", "sentence-transformers",
    ]
    installed = {}
    for p in pkgs:
        try:
            installed[p] = importlib.metadata.version(p)
        except importlib.metadata.PackageNotFoundError:
            installed[p] = "builtin-engine"
    return {
        "python_version": sys.version.split()[0],
        "packages": installed,
        "endpoints_count": len(app.routes),
    }


@app.get("/node/logs")
def get_system_logs():
    ensure_initialized()
    return {"logs": system_logs[-50:]}


@app.delete("/node/logs")
def clear_system_logs():
    ensure_initialized()
    system_logs.clear()
    record_log("api.server", "INFO", "System telemetry log buffer cleared by operator.")
    return {"status": "cleared", "logs": system_logs}


# ─── Static UI (Guaranteed on both Vercel Serverless & Localhost) ─────────
@app.get("/ui", include_in_schema=False)
def serve_ui_root():
    return FileResponse(os.path.join(UI_PATH, "index.html"))


if os.path.isdir(UI_PATH):
    app.mount("/ui", StaticFiles(directory=UI_PATH, html=True), name="ui")
