from datetime import datetime, timedelta, timezone
import hashlib
import importlib.metadata
import os
import sys
import time
from typing import Any, Dict, List, Optional

WIB_TZ = timezone(timedelta(hours=7), name="WIB")

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, PlainTextResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from gajahmada_ai.agents.scheduler import AutonomousTaskScheduler
from gajahmada_ai.api.v2_routes import create_v2_router
from gajahmada_ai.blockchain.chain import AIBlockchain, Block, LedgerEventType
from gajahmada_ai.cognition.reasoning import CognitiveReasoningPipeline
from gajahmada_ai.core.engine import AIModelEngine
from gajahmada_ai.knowledge.indonesia_corpus import get_indonesia_knowledge_corpus
from gajahmada_ai.knowledge.open_knowledge import crawl_internet_knowledge, format_references_footer
from gajahmada_ai.knowledge.rag import LocalKnowledge
from gajahmada_ai.learning.experiments import LearningExperimentManager
from gajahmada_ai.memory.manager import MemoryManager
from gajahmada_ai.memory.models import MemoryType, VerificationStatus, VisibilityPolicy
from gajahmada_ai.network.discovery import PeerDiscoveryRegistry
from gajahmada_ai.network.p2p import P2PNetwork
from gajahmada_ai.network.protocol import GIPMessage, GIPMessageType, GIPProtocolValidator
from gajahmada_ai.network.synchronization import KnowledgeSynchronizer
from gajahmada_ai.security.audit import SecurityAuditLogger
from gajahmada_ai.security.auth import AuthManager
from gajahmada_ai.security.crypto import NodeIdentity
from gajahmada_ai.security.policies import GovernancePolicy
from gajahmada_ai.security.rate_limit import SlidingWindowRateLimiter
from gajahmada_ai.security.ssrf import validate_peer_url
from gajahmada_ai.storage.supabase_store import SupabaseKnowledgeStore

try:
    import yaml
except ImportError:
    yaml = None

IS_VERCEL = os.environ.get("VERCEL") == "1"
START_TIME = time.time()

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.normpath(os.path.join(_HERE, "..", ".."))
UI_PATH = os.path.join(_ROOT, "interface", "html")
CONFIG_PATH = os.path.join(_ROOT, "config.yaml")


def get_system_time_info() -> Dict[str, Any]:
    now = datetime.now(WIB_TZ)
    return {
        "iso": now.isoformat(),
        "formatted": now.strftime("%Y-%m-%d %H:%M:%S WIB"),
        "date": now.strftime("%Y-%m-%d"),
        "time": now.strftime("%H:%M:%S"),
        "timezone": "WIB (UTC+07:00)",
        "year": now.year,
        "epoch": round(now.timestamp(), 3),
    }


DEFAULT_KNOWLEDGE_DOCS = [
    "GajahMada System Architecture: Decentralized AI inference & Web3 node terdiri dari 8 modul inti (api.server, core.engine, knowledge.rag, blockchain.chain, network.p2p, security.crypto, cli.main, interface.html).",
    "Module core.engine (AIModelEngine): Menjalankan inferensi LLM lokal (LLaMA GGUF / Neural Synthesis) dan menghasilkan vektor embedding semantik 384-dimensi (MiniLM-L6-v2 / Hash-Projection).",
    "Module knowledge.rag (LocalKnowledge): Menyimpan dokumen dalam indeks vektor 384-dimensi (FAISS IndexFlatL2 & L2-VectorStore) untuk Retrieval-Augmented Generation (RAG) berlatensi rendah.",
    "Module blockchain.chain (AIBlockchain): Mencatat setiap task_hash SHA-256 (prompt + response) ke dalam rantai blok Proof-of-Work (difficulty=2) yang ditandatangani secara kriptografis.",
    "Module security.crypto (NodeIdentity): Menggunakan pasangan kunci asimetris Ed25519 (RFC 8032) untuk identitas node unik (node_id) dan verifikasi tanda tangan digital setiap blok.",
    "Module network.p2p (P2PNetwork): Mengelola topologi mesh peer-to-peer asinkron via httpx, sinkronisasi bootstrap peers (/node/sync), dan siaran gossip bertanda tangan (/node/broadcast).",
] + get_indonesia_knowledge_corpus()

app = FastAPI(
    title="GajahMada 2.0 — Sovereign Hybrid Distributed Intelligence Node",
    description="Sovereign, Independent, Hybrid, Distributed Intelligence Architecture (Local & Serverless Ready)",
    version="2.0.0",
)

_allowed_origins_env = os.environ.get("GAJAHMADA_ALLOWED_ORIGINS", "").strip()
_cors_origins = (
    [o.strip() for o in _allowed_origins_env.split(",") if o.strip()]
    if _allowed_origins_env
    else ["http://localhost:8000", "http://127.0.0.1:8000", "https://gajahmada.vercel.app"]
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins,
    allow_credentials=False,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["Content-Type", "Authorization", "X-API-Key", "X-Node-Id", "X-Signature", "X-Timestamp"],
)

# ─── Global singletons & live telemetry log buffer ────────────────────────
engine: Optional[AIModelEngine] = None
p2p: Optional[P2PNetwork] = None
blockchain: Optional[AIBlockchain] = None
rag: Optional[LocalKnowledge] = None
identity: Optional[NodeIdentity] = None
memory_manager: Optional[MemoryManager] = None
cognitive_pipeline: Optional[CognitiveReasoningPipeline] = None
agent_scheduler: Optional[AutonomousTaskScheduler] = None
audit_logger: Optional[SecurityAuditLogger] = None
auth_manager: Optional[AuthManager] = None
governance_policy: Optional[GovernancePolicy] = None
rate_limiter: Optional[SlidingWindowRateLimiter] = None
gip_validator: Optional[GIPProtocolValidator] = None
peer_discovery: Optional[PeerDiscoveryRegistry] = None
knowledge_sync: Optional[KnowledgeSynchronizer] = None
learning_manager: Optional[LearningExperimentManager] = None
supabase_store: Optional[SupabaseKnowledgeStore] = None

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
    now = datetime.now(WIB_TZ)
    entry = {
        "id": len(system_logs) + 1,
        "timestamp": now.timestamp(),
        "time_str": now.strftime("%Y-%m-%d %H:%M:%S WIB"),
        "iso_time": now.isoformat(),
        "module": module,
        "level": level,
        "message": message,
    }
    system_logs.append(entry)
    if len(system_logs) > 100:
        del system_logs[:-100]


def ensure_initialized():
    global blockchain, identity, engine, rag, p2p, runtime_config
    global memory_manager, cognitive_pipeline, agent_scheduler
    global audit_logger, auth_manager, governance_policy, rate_limiter
    global gip_validator, peer_discovery, knowledge_sync, learning_manager, supabase_store

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

    storage_dir = "/tmp/knowledge_store" if IS_VERCEL else os.path.join(_ROOT, "knowledge_store")
    os.makedirs(storage_dir, exist_ok=True)

    if identity is None:
        key_path = "/tmp/node_key.pem" if IS_VERCEL else os.path.join(_ROOT, "node_key.pem")
        identity = NodeIdentity(key_path=key_path)
        record_log("security.crypto", "OK", f"Ed25519 NodeIdentity loaded: {identity.node_id[:14]}...")

    if audit_logger is None:
        audit_logger = SecurityAuditLogger(storage_path=os.path.join(storage_dir, "security_audit.json"))

    if governance_policy is None:
        governance_policy = GovernancePolicy()

    if auth_manager is None:
        admin_key_path = os.path.join(storage_dir, "admin_api.key")
        auth_manager = AuthManager(
            identity=identity,
            audit_logger=audit_logger,
            admin_key_file=admin_key_path,
        )

    if rate_limiter is None:
        rate_limiter = SlidingWindowRateLimiter(max_requests=240, window_seconds=60.0)

    if blockchain is None:
        chain_path = os.path.join(storage_dir, "chain.json")
        blockchain = AIBlockchain(storage_path=chain_path)
        blockchain.difficulty = int(runtime_config.get("blockchain", {}).get("difficulty", 2))
        record_log(
            "blockchain.chain",
            "OK",
            f"PoW Signed Event Ledger mounted (height={len(blockchain.chain)}, diff={blockchain.difficulty})",
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
        cap = engine.capability_status()
        record_log(
            "core.engine",
            "OK",
            f"AIModelEngine initialized (neural_active={cap['neural_inference_active']}, symbolic_active=True)",
        )

    if rag is None:
        rag = LocalKnowledge(storage_path=storage_dir)
        if engine is not None:
            existing_set = set(rag.doc_store)
            added_default = False
            for doc in DEFAULT_KNOWLEDGE_DOCS:
                if doc not in existing_set:
                    emb = engine.create_embedding(doc)
                    if emb:
                        rag.ingest(doc, emb)
                        added_default = True
            if added_default:
                rag.flush()
        record_log("knowledge.rag", "OK", f"LocalKnowledge mounted ({len(rag.doc_store)} docs indexed)")

    if memory_manager is None:
        sqlite_path = os.path.join(storage_dir, "gajahmada_v2.db")
        backup_dir = os.path.join(storage_dir, "objects")
        memory_manager = MemoryManager(
            embed_fn=engine.create_embedding if engine else None,
            sqlite_path=sqlite_path,
            backup_dir=backup_dir,
        )
        # Seed initial semantic memory from LocalKnowledge if empty
        if memory_manager.storage.health()["active_records"] == 0 and rag is not None:
            for doc in rag.doc_store:
                memory_manager.store(
                    content=doc,
                    memory_type=MemoryType.SEMANTIC,
                    origin_node_id=identity.node_id if identity else "local",
                    source_type="sovereign_core_corpus",
                    author="system_bootstrap",
                    confidence=0.92,
                    verification_status=VerificationStatus.VERIFIED,
                    visibility_policy=VisibilityPolicy.COLLECTIVE,
                )
        record_log("memory.manager", "OK", f"5-Tier MemoryManager mounted ({memory_manager.storage.health()['active_records']} records)")

    if cognitive_pipeline is None:
        cognitive_pipeline = CognitiveReasoningPipeline(
            model_router=engine.router if engine else None,
            symbolic_reasoner=engine.symbolic if engine else None,
            fallback_synthesizer=engine._synthesize_response if engine else None,
        )
        record_log("cognition.reasoning", "OK", "14-Stage Hybrid CognitiveReasoningPipeline active")

    if agent_scheduler is None:
        agent_scheduler = AutonomousTaskScheduler(
            memory_manager=memory_manager,
            cognitive_pipeline=cognitive_pipeline,
            policy=governance_policy,
            audit_logger=audit_logger,
        )
        record_log("agents.scheduler", "OK", f"AutonomousTaskScheduler ready ({len(agent_scheduler.list_agents())} specialist agents)")

    if gip_validator is None:
        gip_validator = GIPProtocolValidator(
            local_identity=identity,
            replay_db_path=os.path.join(storage_dir, "gip_replay.db"),
        )

    if peer_discovery is None:
        peer_discovery = PeerDiscoveryRegistry()

    if knowledge_sync is None:
        knowledge_sync = KnowledgeSynchronizer(memory=memory_manager, identity=identity)

    if learning_manager is None:
        learning_manager = LearningExperimentManager()

    if supabase_store is None:
        supabase_store = SupabaseKnowledgeStore()
        if supabase_store.configured:
            record_log("storage.supabase", "OK", f"Supabase pgvector remote knowledge store configured ({supabase_store.url})")
        else:
            record_log("storage.supabase", "INFO", "Supabase not configured; running in local-first SQLite mode")


def get_runtime_context() -> Dict[str, Any]:
    ensure_initialized()
    return {
        "engine": engine,
        "p2p": p2p,
        "blockchain": blockchain,
        "rag": rag,
        "identity": identity,
        "memory": memory_manager,
        "cognition": cognitive_pipeline,
        "scheduler": agent_scheduler,
        "audit_logger": audit_logger,
        "auth": auth_manager,
        "policy": governance_policy,
        "gip_validator": gip_validator,
        "discovery": peer_discovery,
        "synchronizer": knowledge_sync,
        "learning": learning_manager,
        "supabase": supabase_store,
    }


app.include_router(create_v2_router(get_runtime_context))


# ─── Request models (v1 Backward Compatible with Input Bounds) ────────────
class ChatRequest(BaseModel):
    prompt: str = Field(..., min_length=1, max_length=16384)
    max_tokens: int = Field(default=256, ge=16, le=4096)
    distribute: bool = False


class EmbeddingRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=32768)


class SearchKnowledgeRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=4096)
    k: int = Field(default=3, ge=1, le=50)


class SyncPeersRequest(BaseModel):
    peers: List[str]


class BroadcastRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=4096)


class MineBlockRequest(BaseModel):
    note: str = Field(default="Manual operator audit block", max_length=1024)


class SignVerifyRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=8192)


class ConfigUpdateRequest(BaseModel):
    node_name: Optional[str] = None
    pow_difficulty: Optional[int] = None
    max_tokens: Optional[int] = None
    temperature: Optional[float] = None


# ─── Startup ──────────────────────────────────────────────────────────────
@app.on_event("startup")
async def startup_event():
    ensure_initialized()


# ─── Health, Root & SEO Endpoints ─────────────────────────────────────────
@app.get("/")
def root(request: Request, format: Optional[str] = None):
    ensure_initialized()
    accept = request.headers.get("accept", "")
    if format != "json" and ("text/html" in accept or "*/*" in accept) and os.path.exists(os.path.join(UI_PATH, "index.html")):
        if "application/json" not in accept:
            return FileResponse(
                os.path.join(UI_PATH, "index.html"),
                headers={"Cache-Control": "no-store, no-cache, must-revalidate, max-age=0"},
            )
    cap = engine.capability_status() if engine else {}
    return {
        "status": "online",
        "message": "GajahMada Network Node is running.",
        "mode": "serverless" if IS_VERCEL else "full",
        "ml_available": engine is not None,
        "neural_inference_active": cap.get("neural_inference_active", False),
        "symbolic_reasoning_active": cap.get("symbolic_reasoning_active", True),
        "capabilities": cap,
        "uptime_sec": round(time.time() - START_TIME, 1),
        "ui_url": "/ui",
    }


@app.get("/api/status")
def api_status():
    ensure_initialized()
    cap = engine.capability_status() if engine else {}
    return {
        "status": "online",
        "message": "GajahMada Network Node is running.",
        "mode": "serverless" if IS_VERCEL else "full",
        "ml_available": engine is not None,
        "neural_inference_active": cap.get("neural_inference_active", False),
        "symbolic_reasoning_active": cap.get("symbolic_reasoning_active", True),
        "capabilities": cap,
        "uptime_sec": round(time.time() - START_TIME, 1),
        "system_time": get_system_time_info(),
        "ui_url": "/ui",
    }


@app.get("/google911ef45bbcb1db67.html", include_in_schema=False)
@app.get("/ui/google911ef45bbcb1db67.html", include_in_schema=False)
def google_site_verification():
    return Response(
        content="google-site-verification: google911ef45bbcb1db67.html",
        media_type="text/html; charset=utf-8",
        headers={"Cache-Control": "public, max-age=3600"},
    )


@app.get("/robots.txt", include_in_schema=False)
def robots_txt():
    content = (
        "User-agent: *\n"
        "Allow: /\n"
        "Allow: /ui\n"
        "Sitemap: https://gajahmada.vercel.app/sitemap.xml\n"
    )
    return PlainTextResponse(content)


@app.get("/sitemap.xml", include_in_schema=False)
def sitemap_xml():
    today = datetime.now().astimezone().strftime("%Y-%m-%d")
    xml = f"""<?xml version="1.0" encoding="UTF-8"?>
<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">
  <url>
    <loc>https://gajahmada.vercel.app/</loc>
    <lastmod>{today}</lastmod>
    <changefreq>daily</changefreq>
    <priority>1.0</priority>
  </url>
  <url>
    <loc>https://gajahmada.vercel.app/ui</loc>
    <lastmod>{today}</lastmod>
    <changefreq>daily</changefreq>
    <priority>0.9</priority>
  </url>
</urlset>"""
    return Response(content=xml, media_type="application/xml")


@app.get("/manifest.json", include_in_schema=False)
def web_manifest():
    return {
        "name": "GajahMada — Sovereign Decentralized Node",
        "short_name": "GajahMada",
        "description": "Sovereign Decentralized AI & Web3 Node with 384-d Vector RAG, SHA-256 PoW Blockchain, Ed25519 Identity, and P2P Mesh.",
        "start_url": "/",
        "display": "standalone",
        "background_color": "#040706",
        "theme_color": "#040706",
        "icons": [
            {
                "src": "/ui/logo.png",
                "sizes": "512x512",
                "type": "image/png",
                "purpose": "any maskable",
            }
        ],
    }


@app.get("/health")
def health():
    ensure_initialized()
    cap = engine.capability_status() if engine else {}
    return {
        "ok": True,
        "mode": "serverless" if IS_VERCEL else "full",
        "neural_inference_active": cap.get("neural_inference_active", False),
        "symbolic_reasoning_active": True,
        "supabase": supabase_store.health() if supabase_store else {"configured": False},
        "uptime_sec": round(time.time() - START_TIME, 1),
    }


# ─── AI & Command Interface Endpoints ─────────────────────────────────────
@app.post("/ai/chat")
async def ai_chat(req: ChatRequest):
    ensure_initialized()
    start = time.time()
    context = ""
    matched_docs = 0
    memory_records = []

    active_top_k = int(learning_manager.adapter.active_strategy.get("retrieval_top_k", 3)) if learning_manager else 2

    if rag and engine:
        try:
            query_emb = engine.create_embedding(req.prompt)
            if query_emb:
                results = rag.search(query_emb, k=active_top_k, query_text=req.prompt)
                if results:
                    matched_docs = len(results)
                    context = "\n".join(results)
                    memory_records = [
                        {
                            "id": f"rag-doc-{i}",
                            "content": doc,
                            "source": "sovereign_core_corpus",
                            "verification_status": "VERIFIED",
                        }
                        for i, doc in enumerate(results)
                    ]
                if supabase_store and supabase_store.configured:
                    sb_hits = supabase_store.search_knowledge(query_emb, k=active_top_k)
                    seen_txt = {m["content"] for m in memory_records}
                    for j, hit in enumerate(sb_hits):
                        txt = (hit.get("content") or "").strip()
                        if txt and txt not in seen_txt:
                            seen_txt.add(txt)
                            memory_records.append({
                                "id": f"sb-{hit.get('content_hash', j)[:10]}",
                                "content": txt,
                                "source": hit.get("source_url") or "supabase_pgvector",
                                "verification_status": hit.get("verification_status", "UNVERIFIED"),
                            })
                            matched_docs += 1
        except Exception as e:
            record_log("knowledge.rag", "WARN", f"RAG lookup warning: {e}")

    cfg_temp = float(runtime_config.get("ai", {}).get("temperature", 0.7))
    cfg_tokens = int(req.max_tokens or runtime_config.get("ai", {}).get("max_tokens", 256))

    # Execute through the 14-Stage Cognitive Reasoning Pipeline
    reasoning_res = cognitive_pipeline.execute(
        objective=req.prompt,
        memory_records=memory_records,
        max_tokens=cfg_tokens,
        temperature=cfg_temp,
    )
    response = reasoning_res.conclusion

    # Closed-loop Experience Extraction & Bounded Strategy Adaptation (Level 1 & 2 Learning)
    learning_outcome = (
        learning_manager.record_outcome_and_adapt(
            objective=req.prompt,
            reasoning_res=reasoning_res,
            memory_manager=memory_manager,
            rag_matches=matched_docs,
        )
        if learning_manager
        else {"strategy_version": 1, "retrieval_top_k": active_top_k, "procedural_record_id": None, "adaptation_triggered": False}
    )

    # Persist every question & answer interaction into the GajahMada Brain (RAG Vector Store + 5-Tier Memory)
    brain_record_id = None
    try:
        crawled_info = getattr(engine, "last_crawled_knowledge", None) if engine else None
        if crawled_info and crawled_info.get("extract"):
            knowledge_entry = f"{crawled_info['title']}: {crawled_info['extract']} (Referensi: {crawled_info.get('source_url', '')})"
            src_type = "internet_knowledge_crawler"
        else:
            clean_resp = response.split("\n\n---\n📚")[0].strip()
            knowledge_entry = f"Pertanyaan: {req.prompt.strip()} | Jawaban: {clean_resp}"
            src_type = "cognitive_brain_qa_update"

        # 1. Update 384-d Vector RAG Store (docs.json) if not already present and not a generic fallback
        is_generic_fallback = (
            "Permintaan telah diproses oleh **GajahMada Hybrid Cognitive Engine**" in response
            or "Belum ada data spesifik untuk" in response
        )
        if rag and engine and not is_generic_fallback and knowledge_entry not in rag.doc_store:
            k_emb = engine.create_embedding(knowledge_entry)
            if k_emb:
                rag.ingest(knowledge_entry, k_emb)
                rag.flush()

        # 2. Persist Q&A with P2 Knowledge Quarantine Policy:
        #    Unverified LLM/crawler outputs enter LOCAL Quarantine (UNVERIFIED),
        #    never directly into COLLECTIVE memory until independently verified.
        can_promote_collective = (
            not is_generic_fallback
            and reasoning_res.verification_status == "VERIFIED"
            and float(reasoning_res.confidence) >= 0.85
            and not reasoning_res.contradictions
        )
        if memory_manager:
            target_tier = MemoryType.EPISODIC if is_generic_fallback else MemoryType.SEMANTIC
            rec = memory_manager.store(
                content=knowledge_entry,
                memory_type=target_tier,
                origin_node_id=identity.node_id if identity else "local",
                source_type=src_type,
                author="cognitive_brain",
                confidence=reasoning_res.confidence,
                verification_status=(
                    VerificationStatus.VERIFIED if can_promote_collective else VerificationStatus.UNVERIFIED
                ),
                visibility_policy=(
                    VisibilityPolicy.COLLECTIVE if can_promote_collective else VisibilityPolicy.LOCAL
                ),
            )
            brain_record_id = rec["record_id"]
            memory_manager.working.push({
                "record_id": brain_record_id,
                "question": req.prompt.strip(),
                "answer_preview": response[:240],
                "confidence": reasoning_res.confidence,
                "timestamp": start,
            })
        # 3. Sync ONLY verified COLLECTIVE knowledge to Supabase pgvector (never quarantined/unverified)
        if supabase_store and supabase_store.configured and can_promote_collective:
            k_emb_sb = engine.create_embedding(knowledge_entry) if engine else None
            sig_sb = identity.sign_message(knowledge_entry.encode("utf-8")) if identity else ""
            supabase_store.upsert_knowledge(
                content=knowledge_entry,
                embedding=k_emb_sb,
                source_type=src_type,
                source_url=(crawled_info.get("source_url", "") if crawled_info else ""),
                origin_node_id=identity.node_id if identity else "local",
                signature=sig_sb,
                confidence=reasoning_res.confidence,
                verification_status="VERIFIED",
                visibility="COLLECTIVE",
            )
        record_log("memory.brain", "OK", f"Brain knowledge updated from question '{req.prompt[:32]}' (id={brain_record_id}, docs={len(rag.doc_store) if rag else 0})")
    except Exception as e:
        record_log("memory.brain", "WARN", f"Brain knowledge update warning: {e}")

    task_hash = hashlib.sha256((req.prompt + response).encode()).hexdigest()
    signature = ""
    block_idx = 0

    if blockchain and identity:
        try:
            # P0 Strict Schema v2: ALL new blocks go through record_event() with full-envelope Ed25519 signature
            block = blockchain.record_event(
                event_type=LedgerEventType.TASK_COMPLETED,
                content_reference=f"{reasoning_res.reasoning_id}|{task_hash[:16]}",
                identity=identity,
                metadata={
                    "reasoning_id": reasoning_res.reasoning_id,
                    "task_hash": task_hash,
                    "verification_status": reasoning_res.verification_status,
                },
            )
            block_idx = block.index
            signature = block.signature
            if supabase_store and supabase_store.configured:
                supabase_store.upsert_block(block.to_dict())
        except Exception as e:
            record_log("blockchain.chain", "WARN", f"Block mining error: {e}")

    elapsed = round(time.time() - start, 3)
    record_log(
        "core.engine",
        "INFO",
        f"POST /ai/chat processed in {elapsed}s (rag_matches={matched_docs}, block=#{block_idx}, status={reasoning_res.verification_status})",
    )

    return {
        "response": response,
        "time_taken": elapsed,
        "block_index": block_idx,
        "task_hash": task_hash,
        "signature": signature[:24] if signature else None,
        "rag_matches": matched_docs,
        "brain_record_id": brain_record_id,
        "knowledge_docs": len(rag.doc_store) if rag else 0,
        "reasoning_id": reasoning_res.reasoning_id,
        "verification_status": reasoning_res.verification_status,
        "confidence": reasoning_res.confidence,
        "uncertainty": reasoning_res.uncertainty,
        "capabilities_used": reasoning_res.capabilities_used,
        "strategy_version": learning_outcome.get("strategy_version", 1),
        "procedural_record_id": learning_outcome.get("procedural_record_id"),
    }


@app.post("/ai/ingest")
async def ai_ingest(req: EmbeddingRequest, request: Request):
    ensure_initialized()
    actor = auth_manager.authenticate_request(request, required_capability="memory:write")
    text = req.text.strip()
    if not text:
        raise HTTPException(status_code=400, detail="Text cannot be empty.")
    emb = engine.create_embedding(text)
    if not emb:
        raise HTTPException(status_code=500, detail="Failed to create 384-d embedding.")
    rag.ingest(text, emb)
    rag.flush()

    # Also persist into Layer 4 Semantic Memory with provenance
    mem_rec = memory_manager.store(
        content=text,
        memory_type=MemoryType.SEMANTIC,
        origin_node_id=identity.node_id if identity else "local",
        source_type="operator_ingested",
        author="local_operator",
        confidence=0.80,
        verification_status=VerificationStatus.VERIFIED,
        visibility_policy=VisibilityPolicy.LOCAL,
    )

    supabase_hash = None
    if supabase_store and supabase_store.configured:
        sig_sb = identity.sign_message(text.encode("utf-8")) if identity else ""
        supabase_hash = supabase_store.upsert_knowledge(
            content=text,
            embedding=emb,
            source_type="operator_ingested",
            origin_node_id=identity.node_id if identity else "local",
            signature=sig_sb,
            confidence=0.80,
            verification_status="VERIFIED",
            visibility="COLLECTIVE",
        )

    tokens = len(text.split())
    record_log("knowledge.rag", "OK", f"Assimilated document ({tokens} tokens, total_docs={len(rag.doc_store)})")
    return {
        "status": "assimilated",
        "tokens": tokens,
        "total_docs": len(rag.doc_store),
        "dimension": rag.dimension,
        "record_id": mem_rec["record_id"],
        "integrity_hash": mem_rec["integrity_hash"],
        "supabase_content_hash": supabase_hash,
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
        "memory_tiers": memory_manager.stats()["tiers"] if memory_manager else {},
        "supabase": supabase_store.health() if supabase_store else {"configured": False},
        "docs": [
            {"index": idx, "preview": doc[:120], "tokens": len(doc.split())}
            for idx, doc in enumerate(rag.doc_store if rag else [])
        ],
    }


@app.post("/knowledge/search")
def knowledge_search(req: SearchKnowledgeRequest):
    ensure_initialized()
    emb = engine.create_embedding(req.query)
    results = rag.search(emb, k=req.k, query_text=req.query) if rag else []
    record_log("knowledge.rag", "INFO", f"Semantic search query='{req.query[:30]}' -> {len(results)} match(es)")
    return {"query": req.query, "results": results, "count": len(results)}


class CrawlKnowledgeRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=4096)
    assimilate: bool = True


@app.post("/knowledge/crawl")
@app.post("/api/v2/knowledge/crawl")
def knowledge_crawl(req: CrawlKnowledgeRequest, request: Request):
    ensure_initialized()
    auth_manager.authenticate_request(request, required_capability="memory:write")
    crawled = crawl_internet_knowledge(req.query)
    if not crawled:
        raise HTTPException(status_code=404, detail=f"No internet knowledge found for query: {req.query}")

    refs = crawled.get("references") or [{"title": crawled["title"], "url": crawled.get("source_url", "")}]
    formatted_answer = (
        f"**{crawled['title']}** *(Hasil Crawling Pengetahuan Internet)*:\n\n"
        f"{crawled['extract']}"
        + format_references_footer(refs)
    )
    record_id = None
    if req.assimilate and rag and engine:
        crawl_doc = f"{crawled['title']}: {crawled['extract']} (Referensi: {crawled.get('source_url', '')})"
        emb = engine.create_embedding(crawl_doc)
        if emb:
            rag.ingest(crawl_doc, emb)
            rag.flush()
        if memory_manager:
            # P2 Quarantine Policy: incoming web-crawled data enters LOCAL Quarantine (UNVERIFIED)
            # until independently verified and promoted via /api/v2/knowledge/promote/{record_id}.
            rec = memory_manager.store(
                content=crawl_doc,
                memory_type=MemoryType.SEMANTIC,
                origin_node_id=identity.node_id if identity else "local",
                source_type="internet_knowledge_crawler",
                author="autonomous_crawler",
                confidence=0.55,
                verification_status=VerificationStatus.UNVERIFIED,
                visibility_policy=VisibilityPolicy.LOCAL,
            )
            record_id = rec["record_id"]

    record_log("knowledge.crawler", "OK", f"POST /knowledge/crawl query='{req.query[:30]}' -> title='{crawled['title']}'")
    return {
        "status": "crawled",
        "query": req.query,
        "title": crawled["title"],
        "extract": crawled["extract"],
        "source_url": crawled.get("source_url", ""),
        "references": refs,
        "formatted_response": formatted_answer,
        "record_id": record_id,
        "total_docs": len(rag.doc_store) if rag else 0,
    }


@app.delete("/knowledge/docs/{doc_idx}")
def delete_knowledge_doc(doc_idx: int, request: Request):
    ensure_initialized()
    auth_manager.authenticate_request(request, required_capability="memory:delete")
    if not rag or doc_idx < 0 or doc_idx >= len(rag.doc_store):
        raise HTTPException(status_code=404, detail="Document index not found.")
    removed = rag.remove_at(doc_idx)
    record_log("knowledge.rag", "WARN", f"Removed document #{doc_idx} ({removed[:30]}...)")
    return {"status": "deleted", "total_docs": len(rag.doc_store)}


# ─── Node, Network, Config, Dependencies & Diagnostics Endpoints ──────────
@app.get("/node/info")
def node_info():
    ensure_initialized()
    cap = engine.capability_status() if engine else {}
    return {
        "node_id": identity.node_id if identity else "uninitialized",
        "node_name": runtime_config.get("node", {}).get("name", "gajahmada-alpha"),
        "peers": list(p2p.peers) if p2p else [],
        "blockchain_height": len(blockchain.chain) if blockchain else 0,
        "blockchain_valid": blockchain.is_chain_valid() if blockchain else True,
        "ledger_status": blockchain.status if blockchain else "UNKNOWN",
        "pow_difficulty": blockchain.difficulty if blockchain else 2,
        "knowledge_docs": len(rag.doc_store) if rag else 0,
        "mode": "serverless" if IS_VERCEL else "full",
        "ml_ready": engine is not None,
        "neural_inference_active": cap.get("neural_inference_active", False),
        "symbolic_reasoning_active": True,
        "active_neural_backend": cap.get("active_neural_backend"),
        "active_model_id": cap.get("active_model_id"),
        "supabase_configured": supabase_store.configured if supabase_store else False,
        "memory_tiers": memory_manager.stats()["tiers"] if memory_manager else {},
        "uptime_sec": round(time.time() - START_TIME, 1),
        "system_time": get_system_time_info(),
    }


@app.get("/node/peers")
def get_peers():
    ensure_initialized()
    return {"peers": list(p2p.peers) if p2p else []}


@app.post("/node/sync")
def sync_node(req: SyncPeersRequest, request: Request):
    ensure_initialized()
    auth_manager.authenticate_request(request, required_capability="network:manage")
    added = []
    for peer_url in req.peers:
        clean = peer_url.strip()
        ok, reason_or_clean = validate_peer_url(clean, allow_local=governance_policy.allow_local_peers)
        if not ok:
            audit_logger.record(
                event_type="SSRF_PEER_BLOCKED",
                severity="HIGH",
                actor="api:/node/sync",
                action=f"register_peer:{clean}",
                outcome="BLOCKED",
                details={"reason": reason_or_clean},
            )
            raise HTTPException(status_code=400, detail=f"Invalid or blocked peer URL: {reason_or_clean}")
        p2p.add_peer(reason_or_clean)
        added.append(reason_or_clean)
    record_log("network.p2p", "OK", f"Synced peer(s): {', '.join(added)} (total={len(p2p.peers)})")
    return {"status": "ok", "added": added, "peers_known": list(p2p.peers)}


@app.delete("/node/peers")
def remove_peer(peer: str, request: Request):
    ensure_initialized()
    auth_manager.authenticate_request(request, required_capability="network:manage")
    p2p.remove_peer(peer.strip())
    record_log("network.p2p", "WARN", f"Unlinked peer: {peer}")
    return {"status": "removed", "peers_known": list(p2p.peers)}


@app.post("/node/broadcast")
async def broadcast_to_peers(req: BroadcastRequest, request: Request):
    ensure_initialized()
    auth_manager.authenticate_request(request, required_capability="network:manage")
    gip_msg = GIPMessage(
        message_type=GIPMessageType.GIP_DISCOVERY,
        sender_node_id=identity.node_id,
        recipient_node_id="*",
        payload={
            "message": req.message,
            "node_name": runtime_config.get("node", {}).get("name", "gajahmada-alpha"),
        },
    ).sign(identity)

    results = await p2p.broadcast("/api/v2/network/gip", gip_msg.model_dump())
    reachable = sum(1 for r in results if not isinstance(r, Exception))
    record_log("network.p2p", "INFO", f"GIP/2.0 Broadcast '{req.message[:30]}' to {len(p2p.peers)} peer(s) ({reachable} reachable)")
    return {
        "status": "broadcast_sent",
        "protocol": "GIP/2.0",
        "message_id": gip_msg.message_id,
        "peers_targeted": len(p2p.peers),
        "peers_reachable": reachable,
        "signature": gip_msg.signature[:24],
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
        "neural_inference_active": engine.is_neural_active() if engine else False,
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
def mine_manual_block(req: MineBlockRequest, request: Request):
    ensure_initialized()
    auth_manager.authenticate_request(request, required_capability="system:admin")
    t0 = time.time()
    block = blockchain.record_event(
        event_type=LedgerEventType.AUDIT_BLOCK,
        content_reference=req.note,
        identity=identity,
    )
    elapsed = round(time.time() - t0, 4)
    record_log("blockchain.chain", "OK", f"Mined manual Block #{block.index} (nonce={block.nonce}, {elapsed}s)")
    return {"status": "mined", "block": block.to_dict(), "time_taken": elapsed}


@app.post("/node/chain/verify")
def verify_full_chain():
    ensure_initialized()
    is_valid = blockchain.is_chain_valid(verify_signatures=True)
    sig_verified = 0
    for b in blockchain.chain[1:]:
        if b.signature and blockchain.verify_block_signature(b):
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
    """
    P0 Security Remediation (Signing Oracle Eliminated):
    Verifies Ed25519 (RFC 8032) cryptographic primitives using an isolated ephemeral
    self-test keypair and a domain-separated server nonce. The node's real private key
    is NEVER exposed as an arbitrary signing oracle.
    """
    ensure_initialized()
    import secrets as _secrets
    ephemeral = NodeIdentity(key_path=None)
    msg_digest = hashlib.sha256(req.message.encode("utf-8")).hexdigest()
    challenge = f"GAJAHMADA_SELFTEST_V2|{_secrets.token_hex(16)}|{msg_digest}".encode("utf-8")
    signature = ephemeral.sign_message(challenge)
    verified = ephemeral.verify_message(challenge, signature, ephemeral.node_id)
    record_log("security.crypto", "OK", f"Ed25519 isolated self-test: verified={verified}")
    return {
        "node_id": identity.node_id,
        "selftest_ephemeral_key_id": ephemeral.node_id,
        "message_sha256": msg_digest,
        "signature": signature,
        "verified": verified,
        "algorithm": "Ed25519 (RFC 8032)",
        "oracle_protected": True,
    }


@app.get("/node/config")
def get_node_config():
    ensure_initialized()
    return runtime_config


@app.post("/node/config")
def update_node_config(req: ConfigUpdateRequest, request: Request):
    ensure_initialized()
    auth_manager.authenticate_request(request, required_capability="config:write")
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
        "numpy", "sqlalchemy", "psycopg",
        "faiss-cpu", "llama-cpp-python", "sentence-transformers", "onnxruntime",
    ]
    installed = {}
    for p in pkgs:
        try:
            installed[p] = importlib.metadata.version(p)
        except importlib.metadata.PackageNotFoundError:
            installed[p] = "not-installed (optional)"
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
def clear_system_logs(request: Request):
    ensure_initialized()
    auth_manager.authenticate_request(request, required_capability="logs:clear")
    system_logs.clear()
    record_log("api.server", "INFO", "System telemetry log buffer cleared by operator.")
    return {"status": "cleared", "logs": system_logs}


# ─── Active Rate-Limiting Middleware (P0 Security Remediation) ────────────
@app.middleware("http")
async def enforce_rate_limit(request: Request, call_next):
    path = request.url.path
    if rate_limiter is not None and (
        path.startswith(("/ai/", "/knowledge/", "/node/", "/api/v2/", "/security/"))
    ):
        client_ip = request.client.host if request.client else "unknown"
        bucket_key = f"{client_ip}:{path.split('/')[1]}"
        allowed, _remaining = rate_limiter.check(bucket_key)
        if not allowed:
            return Response(
                content='{"detail":"Rate limit exceeded. Please retry later."}',
                status_code=429,
                media_type="application/json",
                headers={"Retry-After": str(int(rate_limiter.window_seconds))},
            )
    return await call_next(request)


# ─── Static UI (Guaranteed No-Cache on both Vercel Serverless & Localhost) ─
@app.middleware("http")
async def disable_ui_cache(request, call_next):
    response = await call_next(request)
    if request.url.path.startswith("/ui") or request.url.path.endswith((".html", ".css", ".js", ".png")):
        response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
        response.headers["Pragma"] = "no-cache"
        response.headers["Expires"] = "0"
    return response


@app.get("/logo.png", include_in_schema=False)
@app.get("/ui/logo.png", include_in_schema=False)
def serve_logo():
    return FileResponse(
        os.path.join(UI_PATH, "logo.png"),
        headers={"Cache-Control": "public, max-age=3600"},
    )


@app.get("/screenshot.jpg", include_in_schema=False)
@app.get("/ui/screenshot.jpg", include_in_schema=False)
def serve_screenshot():
    return FileResponse(
        os.path.join(UI_PATH, "screenshot.jpg"),
        headers={"Cache-Control": "public, max-age=3600"},
    )


@app.get("/style.css", include_in_schema=False)
def serve_root_css():
    return FileResponse(
        os.path.join(UI_PATH, "style.css"),
        media_type="text/css",
        headers={"Cache-Control": "no-store, max-age=0"},
    )


@app.get("/app.js", include_in_schema=False)
def serve_root_js():
    return FileResponse(
        os.path.join(UI_PATH, "app.js"),
        media_type="application/javascript",
        headers={"Cache-Control": "no-store, max-age=0"},
    )


@app.get("/dashboard", include_in_schema=False)
@app.get("/dashboard/", include_in_schema=False)
@app.get("/ui", include_in_schema=False)
@app.get("/ui/", include_in_schema=False)
def serve_ui_root():
    return FileResponse(
        os.path.join(UI_PATH, "index.html"),
        headers={"Cache-Control": "no-store, no-cache, must-revalidate, max-age=0"},
    )


if os.path.isdir(UI_PATH):
    app.mount("/ui", StaticFiles(directory=UI_PATH, html=True), name="ui")
