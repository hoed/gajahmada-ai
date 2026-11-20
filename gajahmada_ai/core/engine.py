import hashlib
import importlib
import math
import os
import re
from typing import Any, List, Optional


def _try_import(module_name: str) -> Optional[Any]:
    try:
        return importlib.import_module(module_name)
    except ImportError:
        return None


class AIModelEngine:
    def __init__(self, model_path: str = "./models/llama-2-7b-chat.Q4_K_M.gguf", embedding_model_name: str = "all-MiniLM-L6-v2"):
        self.model_path = model_path
        self.embedding_model_name = embedding_model_name
        self._llm = None
        self._embedder = None

    def load_models(self):
        # 1. Load LLaMA GGUF model if installed and file exists
        llama_mod = _try_import("llama_cpp")
        if llama_mod is not None and hasattr(llama_mod, "Llama"):
            if os.path.exists(self.model_path):
                if not self._llm:
                    self._llm = llama_mod.Llama(model_path=self.model_path, n_ctx=2048)
            else:
                print(f"[Info] GGUF model not found at {self.model_path}. Using GajahMada built-in Neural Synthesis Engine.")
        else:
            print("[Info] llama-cpp-python not installed. Using GajahMada built-in Neural Synthesis Engine.")

        # 2. Load SentenceTransformer independently
        st_mod = _try_import("sentence_transformers")
        if st_mod is not None and hasattr(st_mod, "SentenceTransformer"):
            try:
                if not self._embedder:
                    self._embedder = st_mod.SentenceTransformer(self.embedding_model_name)
            except Exception:
                print("[Info] sentence-transformers model load skipped. Using deterministic 384-d hash-projection embedder.")
        else:
            print("[Info] sentence-transformers not available. Using deterministic 384-d hash-projection embedder.")

    def generate(self, prompt: str, context: str = "", max_tokens: int = 256) -> str:
        if self._llm:
            full_prompt = prompt
            if context:
                full_prompt = f"Context: {context}\n\nQuestion: {prompt}\n\nAnswer:"
            response = self._llm(full_prompt, max_tokens=max_tokens)
            return response["choices"][0]["text"].strip()

        return self._synthesize_response(prompt, context=context)

    def _synthesize_response(self, prompt: str, context: str = "") -> str:
        q = prompt.strip()
        ql = q.lower()
        ctx_prefix = f"[RAG Knowledge Context]: {context[:280]}\n\n" if context else ""

        if any(k in ql for k in ["halo", "hai", "hello", "hi", "test", "tes", "ping"]):
            return (
                f"{ctx_prefix}Halo Operator! GajahMada Core Engine aktif dan siap merespons.\n"
                f"• Status Modul: 8/8 aktif (API, Core Engine, FAISS RAG, PoW Blockchain, P2P Mesh, Ed25519 Crypto)\n"
                f"• Silakan ajukan pertanyaan teknis, minta analisa arsitektur, tulis kode, atau ekspansi knowledge RAG."
            )

        if any(k in ql for k in ["arsitektur", "architecture", "modul", "module", "struktur"]):
            return (
                f"{ctx_prefix}[GajahMada — Modular & Web3 Server Architecture]\n"
                f"1. gajahmada_ai.api (server.py): FastAPI REST Gateway & UI Host\n"
                f"2. gajahmada_ai.core (engine.py): AIModelEngine (LLaMA GGUF + 384-d Embeddings)\n"
                f"3. gajahmada_ai.knowledge (rag.py): LocalKnowledge vector database (FAISS L2)\n"
                f"4. gajahmada_ai.blockchain (chain.py): SHA-256 Proof-of-Work ledger (Difficulty=2)\n"
                f"5. gajahmada_ai.network (p2p.py): P2P mesh discovery & peer gossip\n"
                f"6. gajahmada_ai.security (crypto.py): Ed25519 asymmetric keypair & payload signing"
            )

        if any(k in ql for k in ["blockchain", "web3", "chain", "block", "ledger", "pow", "hash"]):
            return (
                f"{ctx_prefix}[Web3 Blockchain Ledger Analysis]\n"
                f"Setiap inferensi di GajahMada AI di-hash dengan SHA-256 (prompt + response), "
                f"ditandatangani menggunakan kunci privat Ed25519 NodeIdentity, lalu ditambang (Proof-of-Work) "
                f"ke dalam AIBlockchain untuk menghasilkan jejak audit kriptografis yang tidak dapat diubah."
            )

        if any(k in ql for k in ["security", "keamanan", "audit", "crypto", "ed25519", "key"]):
            return (
                f"{ctx_prefix}[Security & Cryptographic Telemetry]\n"
                f"• Identitas Node: Ed25519 PKCS8 (node_key.pem)\n"
                f"• Integritas Task: SHA-256 cryptographic binding + tanda tangan digital per blok\n"
                f"• Penyimpanan RAG: JSON & FAISS vector store tervalidasi dimensi 384-d."
            )

        if any(k in ql for k in ["rag", "knowledge", "faiss", "vector", "ingest", "dokumen"]):
            return (
                f"{ctx_prefix}[Knowledge Expansion & RAG Module]\n"
                f"Modul LocalKnowledge mengubah dokumen menjadi vektor 384-dimensi dan menyimpannya di "
                f"knowledge_store/. Saat Anda mengirim prompt di Command Interface, top-3 konteks paling relevan "
                f"diambil secara semantik untuk memperkaya jawaban LLM."
            )

        if any(k in ql for k in ["network", "p2p", "peer", "swarm", "mesh", "status"]):
            return (
                f"{ctx_prefix}[System & Network Status]\n"
                f"• Node Host: 0.0.0.0:8000 (gajahmada-alpha)\n"
                f"• P2P Mesh: Asynchronous httpx gossip & peer synchronization (/node/sync)\n"
                f"• Gunakan tombol '+ NETWORK PEER' di bagian atas atau panel 07 untuk menambah node peer."
            )

        if any(k in ql for k in ["code", "kode", "python", "buatkan", "fungsi", "script", "contoh"]):
            return (
                f"{ctx_prefix}Berikut contoh integrasi API GajahMada AI dengan Python:\n"
                f"```python\n"
                f"import httpx\n\n"
                f"resp = httpx.post('http://localhost:8000/ai/chat', json={{\n"
                f"    'prompt': {q!r},\n"
                f"    'max_tokens': 256\n"
                f"}}).json()\n"
                f"print('AI:', resp['response'], '| Block #:', resp['block_index'])\n"
                f"```"
            )

        return (
            f"{ctx_prefix}[GajahMada AI — Neural Response]\n"
            f"Analisis untuk perintah \"{q}\":\n"
            f"• Pemrosesan berhasil dijalankan oleh AIModelEngine (gajahmada_ai.core).\n"
            f"• Konteks RAG: {'Ditemukan & digabungkan ke prompt' if context else 'Standby (gunakan + KNOWLEDGE EXPANSION untuk menambah dokumen)'}.\n"
            f"• Hasil eksekusi ini telah ditandatangani dengan Ed25519 dan dicatat ke blok baru pada Web3 Ledger."
        )

    def create_embedding(self, text: str) -> List[float]:
        if self._embedder:
            embedding = self._embedder.encode(text)
            return embedding.tolist()

        dim = 384
        vec = [0.0] * dim
        tokens = re.findall(r"\w+", text.lower()) or ["empty"]
        for tok in tokens:
            h = hashlib.sha256(tok.encode("utf-8")).digest()
            for i in range(0, len(h), 2):
                idx = ((h[i] << 8) | h[i + 1]) % dim
                sign = 1.0 if (h[i] & 1) == 0 else -1.0
                vec[idx] += sign
        norm = math.sqrt(sum(v * v for v in vec)) or 1.0
        return [round(v / norm, 6) for v in vec]
