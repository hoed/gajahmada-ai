from datetime import datetime, timedelta, timezone
import hashlib
import importlib
import math
import os
import re
from typing import Any, Dict, List, Optional

from gajahmada_ai.knowledge.open_knowledge import (
    crawl_internet_knowledge,
    format_references_footer,
    resolve_open_domain_wikipedia,
    resolve_quran_query,
)
from gajahmada_ai.neural.llama_backend import deterministic_hash_embed, wants_detailed_answer
from gajahmada_ai.neural.router import ModelRouter
from gajahmada_ai.reasoning.symbolic import SymbolicReasoner

WIB_TZ = timezone(timedelta(hours=7), name="WIB")

_STOPWORDS = {
    "yang", "dan", "di", "ke", "dari", "ini", "itu", "dengan", "untuk", "pada",
    "adalah", "sebagai", "dalam", "atau", "juga", "oleh", "karena", "tentang",
    "apa", "siapa", "kapan", "dimana", "bagaimana", "mengapa", "kenapa", "berapa",
    "jelaskan", "tolong", "buatkan", "sebutkan", "ceritakan", "kamu", "anda", "saya",
    "the", "is", "are", "was", "were", "in", "on", "at", "to", "for", "of", "and",
    "or", "what", "who", "when", "where", "how", "why", "can", "you", "me",
}


def get_wib_now() -> datetime:
    return datetime.now(WIB_TZ)


def extract_keywords(text: str) -> List[str]:
    tokens = re.findall(r"[a-z0-9]+", text.lower())
    return [t for t in tokens if len(t) > 2 and t not in _STOPWORDS]


def _safe_eval_math(expr: str) -> Optional[float]:
    reasoner = SymbolicReasoner()
    _, val = reasoner.evaluate_math(expr)
    return val


def _try_import(module_name: str) -> Optional[Any]:
    try:
        return importlib.import_module(module_name)
    except ImportError:
        return None


def _has_word(text_lower: str, words: List[str]) -> bool:
    for w in words:
        if " " in w:
            if w in text_lower:
                return True
        else:
            if re.search(rf"\b{re.escape(w)}\b", text_lower):
                return True
    return False


class AIModelEngine:
    """
    GajahMada Core Engine (v1 + v2 Bridge).
    Delegates neural model loading and inference to `ModelRouter` (`gajahmada_ai.neural.router`),
    symbolic math evaluation to `SymbolicReasoner` (`gajahmada_ai.reasoning.symbolic`),
    and open-domain internet crawling to `crawl_internet_knowledge` (`gajahmada_ai.knowledge.open_knowledge`).
    Always provides verifiable source references at the end of knowledge answers.
    """

    def __init__(
        self,
        model_path: str = "./models/llama-2-7b-chat.Q4_K_M.gguf",
        embedding_model_name: str = "all-MiniLM-L6-v2",
    ):
        self.model_path = model_path
        self.embedding_model_name = embedding_model_name
        self.router = ModelRouter(gguf_model_path=model_path)
        self.symbolic = SymbolicReasoner()
        self._llm = None
        self._embedder = None
        self.last_crawled_knowledge: Optional[Dict[str, Any]] = None

    def load_models(self) -> Dict[str, bool]:
        load_res = self.router.load_available_models()
        self._llm = self.router.llama_backend._llm if self.router.llama_backend.metadata.is_loaded else None

        st_mod = _try_import("sentence_transformers")
        if st_mod is not None and hasattr(st_mod, "SentenceTransformer"):
            try:
                if not self._embedder:
                    self._embedder = st_mod.SentenceTransformer(self.embedding_model_name)
                    self.router.llama_backend._embedder = self._embedder
            except Exception:
                pass
        return load_res

    def is_neural_active(self) -> bool:
        return self.router.get_active_neural_backend() is not None

    def capability_status(self) -> Dict[str, Any]:
        return self.router.capability_report()

    def generate(
        self,
        prompt: str,
        context: str = "",
        max_tokens: int = 256,
        temperature: float = 0.7,
    ) -> str:
        gen_res = self.router.route_generation(
            prompt=prompt,
            context=context,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        if gen_res.neural_used and gen_res.text:
            return gen_res.text

        return self._synthesize_response(prompt, context=context)

    def _synthesize_response(self, prompt: str, context: str = "") -> str:
        q = prompt.strip()
        ql = q.lower()
        now = get_wib_now()
        now_str = now.strftime("%Y-%m-%d %H:%M:%S WIB (UTC+07:00)")
        current_year = now.year
        years_merdeka = max(1, current_year - 1945)
        detailed = wants_detailed_answer(q)
        self.last_crawled_knowledge = None

        # 1. Identity / "Siapa kamu" / "Apa itu GajahMada"
        if _has_word(ql, [
            "siapa kamu", "kamu siapa", "siapa anda", "anda siapa", "siapa nama kamu",
            "nama kamu siapa", "who are you", "what are you", "perkenalkan diri",
            "apa itu gajahmada", "tentang kamu", "identitas kamu",
        ]):
            if not detailed:
                return (
                    "Saya **GajahMada**, mesin kecerdasan hibrida berdaulat (*Sovereign Hybrid AI & Web3 Node*) "
                    "yang menggabungkan model neural lokal (GGUF), penalaran simbolik, memori 5-tier, dan crawling internet otomatis."
                    + format_references_footer([
                        {"title": "GajahMada AI Repository", "url": "https://github.com/hoed/gajahmada-ai"},
                    ])
                )
            return (
                "Saya adalah **GajahMada**, arsitektur kecerdasan hibrida berdaulat dan terdistribusi "
                "(*Sovereign Hybrid Distributed Intelligence & Web3 Node*).\n\n"
                "• **Asal Nama:** Terinspirasi dari **Mahapatih Gajah Mada**, tokoh pemersatu Nusantara melalui *Sumpah Palapa* (1336 M).\n"
                "• **Kemampuan Utama:**\n"
                "  1. Penalaran hibrida (**Symbolic + Evidence + Local Neural GGUF + Internet Knowledge Crawler**).\n"
                "  2. **5-Tier Persistent Memory** & pencarian semantik **384-d Vector RAG** (`/search`, `/crawl`, `/ingest`).\n"
                "  3. Audit **Web3 Signed Event Ledger SHA-256 PoW** (`/mine`, `/verify`) & kriptografi **Ed25519** + **GIP v2.0 P2P Mesh** (`/status`, `/peers`)."
                + format_references_footer([
                    {"title": "GajahMada AI Repository", "url": "https://github.com/hoed/gajahmada-ai"},
                ])
            )

        # 2. Greetings & Small Talk
        if _has_word(ql, ["halo", "hai", "hello", "hi", "hey", "selamat pagi", "selamat siang", "selamat sore", "selamat malam", "assalamualaikum", "apa kabar", "ping", "tes", "test"]):
            if not _has_word(ql, ["indonesia", "nusantara", "sejarah", "gajah mada", "majapahit", "presiden", "arsitektur", "blockchain"]):
                return (
                    f"Halo! Saya **GajahMada** (`{now_str}`). Ada yang bisa saya bantu? "
                    f"Ketik pertanyaan apa saja atau `/help` untuk daftar perintah."
                )

        # 3. Thanks / Appreciation
        if _has_word(ql, ["terima kasih", "makasih", "thanks", "thank you", "mantap", "keren", "bagus"]):
            return "Sama-sama! Silakan tanyakan hal lain jika diperlukan."

        # 4. Help / Capabilities
        if _has_word(ql, ["help", "bantuan", "fitur", "bisa apa", "apa yang bisa", "daftar perintah"]):
            if not detailed:
                return (
                    "**Perintah Utama GajahMada:** `/crawl <topik/URL>`, `/search <kueri>`, `/ingest <teks>`, "
                    "`/status`, `/time`, `/mine <catatan>`, `/verify`, `/peers`. Atau tanyakan langsung topik apa saja."
                )
            return (
                "**Fitur & Perintah Command Center GajahMada:**\n"
                "• `/crawl <topik/URL>` — Crawling pengetahuan dari internet & simpan ke memori.\n"
                "• `/search <kueri>` — Pencarian semantik 384-d di basis pengetahuan.\n"
                "• `/ingest <teks>` — Simpan pengetahuan baru ke Vector Store.\n"
                "• `/status` · `/time` · `/mine <catatan>` · `/verify` · `/peers`"
            )

        # 5. System Time / Date Queries
        if _has_word(ql, ["jam berapa", "tanggal berapa", "hari apa", "waktu sekarang", "jam sekarang", "tahun berapa", "what time", "current time", "waktu sistem"]):
            if not detailed:
                return f"Waktu sistem sekarang: **`{now_str}`**."
            return (
                f"**Waktu Sistem Sekarang:** `{now_str}`\n"
                f"• **ISO-8601:** `{now.isoformat()}`\n"
                f"• **Tahun:** `{current_year}` ({years_merdeka} tahun Kemerdekaan RI)."
            )

        # 6. Math / Arithmetic & Symbolic Calculation
        sym_res = self.symbolic.reason(q)
        if sym_res.applicable and sym_res.conclusion:
            if sym_res.math_value is not None:
                formatted_num = int(sym_res.math_value) if sym_res.math_value.is_integer() else round(sym_res.math_value, 6)
                return f"Hasilnya adalah **`{formatted_num}`**."
            return f"**{sym_res.conclusion}**"

        # 6B. Al-Qur'an Surah & Ayah Resolver (114 Surahs + Arabic & Kemenag RI Translation + References)
        quran_ans = resolve_quran_query(q)
        if quran_ans:
            return quran_ans

        # 7. Specific Nusantara & Indonesia Topics (concise by default, detailed when requested)
        if _has_word(ql, ["gajah mada", "sumpah palapa", "amukti palapa", "majapahit", "hayam wuruk", "raden wijaya", "nagarakretagama", "sutasoma", "bhinneka tunggal ika", "trowulan"]):
            refs = [{"title": "Wikipedia: Gajah Mada & Sumpah Palapa", "url": "https://id.wikipedia.org/wiki/Gajah_Mada"}]
            if not detailed:
                return (
                    "**Gajah Mada** adalah Mahapatih Kemaharajaan Majapahit (1336–1364 M) yang mengucapkan **Sumpah Palapa** "
                    "untuk menyatukan seluruh Nusantara di bawah Raja Hayam Wuruk. Dari era Majapahit pula lahir semboyan "
                    "**\"Bhinneka Tunggal Ika\"** (*Kakawin Sutasoma*, Mpu Tantular)."
                    + format_references_footer(refs)
                )
            return (
                "**Mahapatih Gajah Mada, Sumpah Palapa & Kemaharajaan Majapahit:**\n"
                "• **Kemaharajaan Majapahit (1293 – Abad ke-16):** Didirikan oleh **Raden Wijaya** di Trowulan (Mojokerto, Jawa Timur) pada tahun 1293, mencapai puncak kejayaan pada masa **Raja Hayam Wuruk (1350–1389)**.\n"
                "• **Sumpah Palapa (1336 M):** Diucapkan Gajah Mada saat diangkat menjadi *Mahapatih Amangkubhumi* — berpantang menikmati *palapa* sebelum menyatukan Nusantara (Gurun, Seran, Tanjung Pura, Haru, Pahang, Dompo, Bali, Sunda, Palembang, Tumasik).\n"
                "• **Warisan:** Kitab *Nagarakretagama* (1365 M, Mpu Prapanca) mencatat 98 wilayah Nusantara, dan *Kakawin Sutasoma* (Mpu Tantular) melahirkan semboyan **\"Bhinneka Tunggal Ika\"**."
                + format_references_footer(refs)
            )

        if _has_word(ql, ["sriwijaya", "mataram kuno", "borobudur", "prambanan", "syailendra", "sanjaya", "kutai", "mulawarman", "tarumanagara", "purnawarman", "kalingga", "singasari", "kertanagara", "ken arok", "kediri", "airlangga", "pamalayu"]):
            refs = [{"title": "Wikipedia: Sriwijaya & Kerajaan Klasik Nusantara", "url": "https://id.wikipedia.org/wiki/Sriwijaya"}]
            if not detailed:
                return (
                    "Kerajaan klasik awal Nusantara dimulai dari **Kutai** dan **Tarumanagara** (abad ke-4–5 M), "
                    "dilanjutkan kemaharajaan maritim **Sriwijaya** (abad ke-7–13 M), **Mataram Kuno** (pembangun Borobudur & Prambanan), "
                    "hingga **Singasari** di bawah Raja Kertanagara."
                    + format_references_footer(refs)
                )
            return (
                "**Kerajaan Klasik & Kemaharajaan Maritim Awal Nusantara:**\n"
                "• **Kutai & Tarumanagara (Abad ke-4 – ke-5 M):** Kerajaan tertua di Kalimantan Timur (Raja Mulawarman, Prasasti Yupa) dan Jawa Barat (Raja Purnawarman).\n"
                "• **Sriwijaya (Abad ke-7 – ke-13 M):** Berpusat di Palembang, menguasai Selat Malaka & Selat Sunda serta pusat Buddha internasional.\n"
                "• **Mataram Kuno (Abad ke-8 – ke-10 M):** Dinasti Syailendra & Sanjaya membangun **Candi Borobudur** dan **Candi Prambanan**.\n"
                "• **Singasari (1222–1292 M):** Didirikan Ken Arok; Raja Kertanagara melancarkan **Ekspedisi Pamalayu (1275)**."
                + format_references_footer(refs)
            )

        if _has_word(ql, ["kesultanan", "samudera pasai", "wali songo", "walisongo", "demak", "mataram islam", "sultan agung", "banten", "gowa", "tallo", "hasanuddin", "ternate", "tidore", "baabullah", "nuku", "iskandar muda", "jalur rempah"]):
            refs = [{"title": "Jalur Rempah Kemdikbud RI", "url": "https://jalurrempah.kemdikbud.go.id"}]
            if not detailed:
                return (
                    "Era Kesultanan Nusantara (abad ke-13–18) mencakup **Samudera Pasai** & **Aceh** (Sumatera), "
                    "**Demak**, **Banten**, & **Mataram Islam** (Jawa), **Gowa-Tallo** (Makassar), serta **Ternate & Tidore** "
                    "di Maluku sebagai pusat jalur rempah dunia."
                    + format_references_footer(refs)
                )
            return (
                "**Era Kesultanan Islam & Jaringan Niaga Rempah Nusantara (Abad ke-13 – ke-18):**\n"
                "• **Sumatera:** Samudera Pasai (Sultan Malik al-Saleh) & Aceh Darussalam (Sultan Iskandar Muda).\n"
                "• **Jawa:** Demak (Raden Patah & Wali Songo), Cirebon, Banten (Sultan Ageng Tirtayasa), & Mataram Islam (Sultan Agung).\n"
                "• **Sulawesi & Maluku:** Gowa-Tallo (Sultan Hasanuddin), Ternate (Sultan Baabullah), & Tidore (Sultan Nuku)."
                + format_references_footer(refs)
            )

        if _has_word(ql, ["pahlawan", "kolonial", "voc", "belanda", "diponegoro", "pattimura", "imam bonjol", "cut nyak dhien", "teuku umar", "antasari", "sisingamangaraja", "kartini", "budi utomo", "kebangkitan nasional", "sumpah pemuda", "28 oktober 1928"]):
            refs = [{"title": "Wikipedia: Sumpah Pemuda & Pergerakan Nasional", "url": "https://id.wikipedia.org/wiki/Sumpah_Pemuda"}]
            if not detailed:
                return (
                    "Perjuangan kemerdekaan dimulai dari perlawanan pahlawan daerah (Diponegoro, Pattimura, Imam Bonjol, Cut Nyak Dhien), "
                    "berlanjut ke **Kebangkitan Nasional** (**Budi Utomo**, 20 Mei 1908), dan ikrar **Sumpah Pemuda** (28 Oktober 1928: "
                    "Satu Tanah Air, Satu Bangsa, Satu Bahasa Indonesia)."
                    + format_references_footer(refs)
                )
            return (
                "**Perlawanan Kolonial, Kebangkitan Nasional (1908) & Sumpah Pemuda (1928):**\n"
                "• **Perlawanan Bersenjata:** Pattimura (1817), Imam Bonjol (1803–1838), Diponegoro (1825–1830), Antasari, Sisingamangaraja XII, Teuku Umar, & Cut Nyak Dhien.\n"
                "• **Kebangkitan Nasional (20 Mei 1908):** R.A. Kartini, Budi Utomo, Sarekat Islam, Muhammadiyah (1912), NU (1926), & PNI (1927).\n"
                "• **Sumpah Pemuda (28 Oktober 1928):** Ikrar Satu Tanah Air, Satu Bangsa, dan Bahasa Persatuan Indonesia."
                + format_references_footer(refs)
            )

        if _has_word(ql, ["proklamasi", "kemerdekaan", "17 agustus", "pancasila", "bpupki", "ppki", "piagam jakarta", "uud 1945", "djuanda", "deklarasi djuanda", "kaa", "konferensi asia afrika", "sudirman"]):
            refs = [{"title": "Wikipedia: Proklamasi Kemerdekaan Indonesia", "url": "https://id.wikipedia.org/wiki/Proklamasi_Kemerdekaan_Indonesia"}]
            if not detailed:
                return (
                    "**Proklamasi Kemerdekaan RI** dibacakan oleh **Ir. Soekarno dan Drs. Mohammad Hatta** pada **17 Agustus 1945** "
                    f"di Jakarta (usia {years_merdeka} tahun pada {current_year}), dengan dasar negara **Pancasila** (1 Juni 1945) "
                    "dan konstitusi **UUD 1945** (18 Agustus 1945)."
                    + format_references_footer(refs)
                )
            return (
                f"**Pancasila, Proklamasi Kemerdekaan RI (17 Agustus 1945) & Kedaulatan NKRI:**\n"
                f"• **Pancasila (1 Juni 1945):** Dirumuskan dalam sidang BPUPKI oleh Ir. Soekarno, diikuti Piagam Jakarta (22 Juni 1945).\n"
                f"• **Proklamasi (17 Agustus 1945):** Dibacakan Soekarno-Hatta di Jl. Pegangsaan Timur No. 56 Jakarta; UUD 1945 disahkan PPKI pada 18 Agustus 1945.\n"
                f"• **Deklarasi Djuanda (13 Desember 1957):** Menyatukan laut antarpulau sebagai wilayah kedaulatan NKRI (*Wawasan Nusantara*)."
                + format_references_footer(refs)
            )

        if _has_word(ql, ["presiden", "wakil presiden", "soekarno", "soeharto", "habibie", "gus dur", "abdurrahman wahid", "megawati", "sby", "yudhoyono", "jokowi", "joko widodo", "prabowo", "gibran", "orde baru", "reformasi"]):
            refs = [{"title": "Wikipedia: Daftar Presiden Indonesia", "url": "https://id.wikipedia.org/wiki/Daftar_Presiden_Indonesia"}]
            if not detailed and not _has_word(ql, ["daftar", "semua", "urutan"]):
                return (
                    f"Presiden Republik Indonesia saat ini ({current_year}) adalah **Prabowo Subianto** (Presiden ke-8, menjabat sejak 20 Oktober 2024), "
                    "didampingi Wakil Presiden **Gibran Rakabuming Raka**."
                    + format_references_footer(refs)
                )
            return (
                f"**Daftar Presiden Republik Indonesia (1945 – {current_year}):**\n"
                "1. **Ir. Soekarno** (1945–1967)\n"
                "2. **Soeharto** (1967–1998)\n"
                "3. **B.J. Habibie** (1998–1999)\n"
                "4. **K.H. Abdurrahman Wahid / Gus Dur** (1999–2001)\n"
                "5. **Megawati Soekarnoputri** (2001–2004)\n"
                "6. **Susilo Bambang Yudhoyono** (2004–2014)\n"
                "7. **Joko Widodo** (2014–2024)\n"
                f"8. **Prabowo Subianto** (2024–sekarang, {current_year}; Wapres: Gibran Rakabuming Raka)"
                + format_references_footer(refs)
            )

        if _has_word(ql, ["provinsi", "38 provinsi", "ikn", "ibu kota nusantara", "geografi", "jumlah pulau", "penduduk", "populasi", "suku", "bahasa daerah", "unesco", "ekonomi", "g20", "hilirisasi", "indonesia emas"]):
            refs = [{"title": "Wikipedia: Provinsi di Indonesia & IKN", "url": "https://id.wikipedia.org/wiki/Provinsi_di_Indonesia"}]
            if not detailed:
                return (
                    f"Indonesia ({current_year}) memiliki **38 Provinsi**, **17.504+ pulau**, penduduk sekitar **282+ juta jiwa**, "
                    "anggota **G20**, dan sedang membangun **Ibu Kota Nusantara (IKN)** di Kalimantan Timur."
                    + format_references_footer(refs)
                )
            return (
                f"**Profil Indonesia Modern ({current_year}):**\n"
                "• **Geografi & Demografi:** 17.504+ pulau, daratan 1,91 juta km², penduduk ~282+ juta jiwa, 1.340+ suku, 718 bahasa daerah.\n"
                "• **Wilayah:** **38 Provinsi** (termasuk 4 provinsi baru di Papua) dan **Ibu Kota Nusantara (IKN)** di Kalimantan Timur.\n"
                "• **Ekonomi & UNESCO:** Anggota G20 (PDB > USD 1,4 triliun), warisan UNESCO antara lain Borobudur, Prambanan, Komodo, Subak, Batik, Wayang, Angklung, & Gamelan."
                + format_references_footer(refs)
            )

        if _has_word(ql, ["indonesia", "nusantara", "sejarah"]):
            refs = [{"title": "Wikipedia: Sejarah Indonesia", "url": "https://id.wikipedia.org/wiki/Sejarah_Indonesia"}]
            if not detailed:
                return (
                    "Sejarah Indonesia berkembang dari kerajaan klasik (**Kutai, Sriwijaya, Majapahit**), era **Kesultanan Rempah**, "
                    "perjuangan nasional (**Budi Utomo 1908, Sumpah Pemuda 1928**), hingga **Proklamasi 17 Agustus 1945**. "
                    f"Kini ({current_year}) Indonesia memiliki **38 Provinsi** dipimpin Presiden **Prabowo Subianto** menuju **Indonesia Emas 2045**."
                    + format_references_footer(refs)
                )
            return (
                f"**Ringkasan Sejarah Nusantara hingga Indonesia ({current_year}):**\n"
                "1. **Kerajaan Awal (Abad ke-4–13 M):** Kutai, Tarumanagara, Sriwijaya, Mataram Kuno, & Singasari.\n"
                "2. **Majapahit (1293–Abad ke-16):** Hayam Wuruk & Mahapatih Gajah Mada (*Sumpah Palapa* 1336 M).\n"
                "3. **Kesultanan & Perlawanan Kolonial (Abad ke-13–19):** Aceh, Demak, Banten, Mataram Islam, Gowa-Tallo, Ternate-Tidore.\n"
                "4. **Pergerakan & Kemerdekaan (1908–1945):** Budi Utomo (1908), Sumpah Pemuda (1928), Pancasila & Proklamasi (17 Agustus 1945).\n"
                f"5. **Indonesia Modern ({current_year}):** 38 Provinsi, ~282+ juta jiwa, anggota G20, pembangunan IKN."
                + format_references_footer(refs)
            )

        if _has_word(ql, ["arsitektur", "architecture", "modul", "module", "struktur"]):
            refs = [{"title": "GajahMada 2.0 Architecture", "url": "https://github.com/hoed/gajahmada-ai"}]
            if not detailed:
                return (
                    "**GajahMada 2.0** terdiri atas 8 lapisan: *Sovereign Foundation* (Ed25519/PoW Ledger), *GIP v2.0 P2P Mesh*, "
                    "*Perception & Web Crawler*, *5-Tier Memory*, *Hybrid Reasoning* (Local GGUF + Symbolic), *Autonomous Agents*, "
                    "*Collective Intelligence*, dan *Governance*."
                    + format_references_footer(refs)
                )
            return (
                "**Arsitektur 8 Lapisan GajahMada 2.0:**\n"
                "1. **Sovereign Foundation:** Ed25519 Identity, Signed Event Ledger, SQLite/PostgreSQL.\n"
                "2. **Distributed Infrastructure:** GIP v2.0 P2P, Replay Protection, SSRF Guard.\n"
                "3. **Perception & Knowledge:** Normalisasi input & Internet Knowledge Crawler.\n"
                "4. **5-Tier Memory Engine:** Working, Semantic, Episodic, Procedural, & Collective Memory.\n"
                "5. **Hybrid Reasoning:** Local Neural (`llama.cpp` GGUF) + Symbolic + Evidence + Critic.\n"
                "6. **Autonomous Agent Runtime:** Planner, Researcher, Analyst, Verifier & Sandbox.\n"
                "7. **Collective Intelligence:** Delegasi & verifikasi multi-node.\n"
                "8. **Intelligence Governance:** RBAC, kebijakan eksekusi, & pembelajaran terkontrol."
                + format_references_footer(refs)
            )

        if _has_word(ql, ["blockchain", "web3", "chain", "block", "ledger", "pow", "hash"]):
            return (
                "Setiap event penting di GajahMada di-hash dengan **SHA-256**, ditandatangani **Ed25519**, "
                "dan ditambang dengan **Proof-of-Work (`difficulty=2`)** ke dalam `chain.json`."
                + format_references_footer([
                    {"title": "GajahMada Signed Event PoW Ledger", "url": "gajahmada_ai/blockchain/chain.py"},
                ])
            )

        if _has_word(ql, ["security", "keamanan", "audit", "crypto", "ed25519"]):
            return (
                "GajahMada menggunakan kriptografi **Ed25519 (RFC 8032)** untuk identitas node (`node_key.pem`, izin `0600`), "
                "tanda tangan blok ledger, serta proteksi *replay* pada pesan P2P GIP v2.0."
                + format_references_footer([
                    {"title": "RFC 8032 — EdDSA", "url": "https://datatracker.ietf.org/doc/html/rfc8032"},
                ])
            )

        if _has_word(ql, ["code", "kode", "python", "buatkan", "fungsi", "script", "contoh"]):
            return (
                f"Contoh pemanggilan API **GajahMada** (Python):\n"
                f"```python\n"
                f"import httpx\n"
                f"resp = httpx.post('http://localhost:8000/ai/chat', json={{'prompt': {q!r}}}).json()\n"
                f"print(resp['response'])\n"
                f"```"
            )

        # 8. Autonomous Multi-Source Internet Knowledge Crawler (Wikipedia ID/EN + DuckDuckGo + Weather + Direct URL)
        crawled = crawl_internet_knowledge(q)
        if crawled:
            self.last_crawled_knowledge = crawled
            refs = crawled.get("references") or [{"title": crawled["title"], "url": crawled.get("source_url", "")}]
            return (
                f"{crawled['extract']}"
                + format_references_footer(refs, max_refs=3 if detailed else 1)
            )

        if context:
            clean_ctx = context.strip()
            if not detailed and len(clean_ctx) > 360:
                clean_ctx = clean_ctx[:357].rsplit(" ", 1)[0] + "..."
            return (
                f"{clean_ctx}"
                + format_references_footer([
                    {"title": "GajahMada Provenance Memory", "url": "gajahmada_ai/memory/manager.py"},
                ], max_refs=1)
            )

        return (
            f"Belum ada data spesifik untuk **\"{q}\"** di memori lokal. "
            f"Coba gunakan `/crawl {q}` atau `/ingest <informasi>` untuk menambahkannya ke Brain."
        )

    def create_embedding(self, text: str) -> List[float]:
        if self._embedder is not None:
            embedding = self._embedder.encode(text)
            return embedding.tolist()
        return self.router.embed(text)
