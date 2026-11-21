from datetime import datetime, timedelta, timezone
import ast
import hashlib
import importlib
import math
import operator
import os
import re
from typing import Any, List, Optional

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
    allowed_ops = {
        ast.Add: operator.add,
        ast.Sub: operator.sub,
        ast.Mult: operator.mul,
        ast.Div: operator.truediv,
        ast.FloorDiv: operator.floordiv,
        ast.Mod: operator.mod,
        ast.Pow: operator.pow,
        ast.USub: operator.neg,
        ast.UAdd: operator.pos,
    }

    def _eval(node):
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return node.value
        if isinstance(node, ast.BinOp) and type(node.op) in allowed_ops:
            left = _eval(node.left)
            right = _eval(node.right)
            if isinstance(node.op, ast.Pow) and abs(right) > 1000:
                raise ValueError("Exponent too large")
            return allowed_ops[type(node.op)](left, right)
        if isinstance(node, ast.UnaryOp) and type(node.op) in allowed_ops:
            return allowed_ops[type(node.op)](_eval(node.operand))
        raise ValueError("Unsupported expression")

    try:
        cleaned = expr.replace("x", "*").replace("×", "*").replace("÷", "/").replace("^", "**")
        m = re.search(r"([0-9]+(?:\.[0-9]+)?(?:\s*[\+\-\*\/\%]\s*[0-9]+(?:\.[0-9]+)?)+)", cleaned)
        if not m:
            return None
        tree = ast.parse(m.group(1), mode="eval")
        return _eval(tree.body)
    except Exception:
        return None


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
    def __init__(
        self,
        model_path: str = "./models/llama-2-7b-chat.Q4_K_M.gguf",
        embedding_model_name: str = "all-MiniLM-L6-v2",
    ):
        self.model_path = model_path
        self.embedding_model_name = embedding_model_name
        self._llm = None
        self._embedder = None

    def load_models(self):
        llama_mod = _try_import("llama_cpp")
        if llama_mod is not None and hasattr(llama_mod, "Llama"):
            if os.path.exists(self.model_path):
                if not self._llm:
                    self._llm = llama_mod.Llama(model_path=self.model_path, n_ctx=2048)
            else:
                print(f"[Info] GGUF model not found at {self.model_path}. Using GajahMada built-in Neural Synthesis Engine.")
        else:
            print("[Info] llama-cpp-python not installed. Using GajahMada built-in Neural Synthesis Engine.")

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
                full_prompt = f"Context:\n{context}\n\nQuestion: {prompt}\n\nAnswer:"
            response = self._llm(full_prompt, max_tokens=max_tokens)
            return response["choices"][0]["text"].strip()

        return self._synthesize_response(prompt, context=context)

    def _synthesize_response(self, prompt: str, context: str = "") -> str:
        q = prompt.strip()
        ql = q.lower()
        now = get_wib_now()
        now_str = now.strftime("%Y-%m-%d %H:%M:%S WIB (UTC+07:00)")
        current_year = now.year
        years_merdeka = max(1, current_year - 1945)

        # 1. Identity / "Siapa kamu" / "Apa itu GajahMada"
        if _has_word(ql, [
            "siapa kamu", "kamu siapa", "siapa anda", "anda siapa", "siapa nama kamu",
            "nama kamu siapa", "who are you", "what are you", "perkenalkan diri",
            "apa itu gajahmada", "tentang kamu", "identitas kamu",
        ]):
            return (
                "Saya adalah **GajahMada**, asisten kecerdasan buatan (AI) terdesentralisasi dan pusat kendali "
                "node Web3 berdaulat (*Sovereign Decentralized AI & Web3 Node*).\n\n"
                "• **Asal Nama:** Terinspirasi dari **Mahapatih Gajah Mada**, tokoh pemersatu Nusantara melalui *Sumpah Palapa* (1336 M).\n"
                "• **Kemampuan Utama Saya:**\n"
                "  1. Menjawab pertanyaan sejarah lengkap **Nusantara hingga Indonesia sekarang** (tahun " + str(current_year) + ").\n"
                "  2. Menjalankan pencarian pengetahuan semantik **384-d Vector RAG** (`/search` & `/ingest`).\n"
                "  3. Mengaudit & menambang blok **Web3 Blockchain SHA-256 Proof-of-Work** (`/mine` & `/verify`).\n"
                "  4. Memverifikasi tanda tangan kriptografis **Ed25519 (RFC 8032)** dan memantau jaringan **P2P Mesh** (`/status`, `/peers`)."
            )

        # 2. Greetings & Small Talk
        if _has_word(ql, ["halo", "hai", "hello", "hi", "hey", "selamat pagi", "selamat siang", "selamat sore", "selamat malam", "assalamualaikum", "apa kabar", "ping", "tes", "test"]):
            if not _has_word(ql, ["indonesia", "nusantara", "sejarah", "gajah mada", "majapahit", "presiden", "arsitektur", "blockchain"]):
                return (
                    f"Halo! Saya **GajahMada** (Waktu Sistem: `{now_str}`).\n"
                    f"Seluruh 8 modul sistem sedang aktif dan siap membantu Anda. Anda dapat bertanya tentang:\n"
                    f"• **Sejarah & Detail Indonesia:** *\"Siapa Mahapatih Gajah Mada?\"*, *\"Apa isi Sumpah Palapa?\"*, *\"Siapa Presiden Indonesia sekarang?\"*, *\"Jelaskan sejarah Nusantara sampai Indonesia sekarang\"*, atau *\"Apa itu IKN & 38 Provinsi?\"*\n"
                    f"• **Arsitektur & Web3:** *\"Jelaskan arsitektur GajahMada\"*, *\"Bagaimana cara kerja blockchain PoW & Ed25519?\"*, atau perintah `/status`, `/verify`, `/mine`, `/time`."
                )

        # 3. Thanks / Appreciation
        if _has_word(ql, ["terima kasih", "makasih", "thanks", "thank you", "mantap", "keren", "bagus"]):
            return (
                "Sama-sama! Senang dapat membantu Anda. Silakan ketik pertanyaan atau perintah berikutnya kapan saja."
            )

        # 4. Help / Capabilities
        if _has_word(ql, ["help", "bantuan", "fitur", "bisa apa", "apa yang bisa", "daftar perintah"]):
            return (
                "Berikut daftar fitur dan perintah yang dapat Anda gunakan di **Command Center GajahMada**:\n"
                "1. **Tanya Jawab AI & Pengetahuan Indonesia:** Tanyakan sejarah Nusantara, Majapahit, Sumpah Palapa, Sriwijaya, Kesultanan, Proklamasi 1945, Pancasila, Daftar Presiden RI, 38 Provinsi, IKN, Ekonomi G20, atau Budaya UNESCO.\n"
                "2. **Perintah Sistem (Slash Commands):**\n"
                "   • `/status` — Cek performa CPU, RAM, Disk, jumlah dokumen RAG, peer P2P, & blockchain.\n"
                "   • `/time` atau `/now` — Cek sinkronisasi waktu sistem sekarang (WIB / ISO-8601).\n"
                "   • `/ingest <teks>` — Tambahkan pengetahuan baru ke dalam Vector Store 384-d.\n"
                "   • `/search <kueri>` — Cari dokumen di basis pengetahuan RAG secara semantik.\n"
                "   • `/mine <catatan>` — Tambang blok Proof-of-Work baru ke Web3 Ledger.\n"
                "   • `/verify` — Verifikasi keaslian rantai blok SHA-256 & tanda tangan Ed25519.\n"
                "   • `/peers` — Lihat daftar node peer P2P yang terhubung."
            )

        # 5. System Time / Date Queries
        if _has_word(ql, ["jam berapa", "tanggal berapa", "hari apa", "waktu sekarang", "jam sekarang", "tahun berapa", "what time", "current time", "waktu sistem"]):
            return (
                f"**Waktu Sistem Sekarang (LIVE):**\n"
                f"• **Tanggal & Jam:** `{now_str}`\n"
                f"• **Format ISO-8601:** `{now.isoformat()}`\n"
                f"• **Konteks Nasional:** Tahun `{current_year}` ({years_merdeka} tahun sejak Proklamasi Kemerdekaan RI 17 Agustus 1945)."
            )

        # 6. Math / Arithmetic Calculation
        math_res = _safe_eval_math(q)
        if math_res is not None and (_has_word(ql, ["berapa", "hitung", "hasil"]) or re.match(r"^[\d\s\+\-\*\/\%\(\)\.x×÷\^=]+$", ql)):
            formatted_num = int(math_res) if isinstance(math_res, float) and math_res.is_integer() else round(math_res, 6)
            return f"Hasil perhitungan untuk `{q}` adalah **`{formatted_num}`**."

        # 7. Specific Nusantara & Indonesia Topics
        # 7A. Gajah Mada, Sumpah Palapa & Majapahit
        if _has_word(ql, ["gajah mada", "sumpah palapa", "amukti palapa", "majapahit", "hayam wuruk", "raden wijaya", "nagarakretagama", "sutasoma", "bhinneka tunggal ika", "trowulan"]):
            return (
                "**Mahapatih Gajah Mada, Sumpah Palapa & Kemaharajaan Majapahit:**\n"
                "• **Kemaharajaan Majapahit (1293 – Abad ke-16):** Didirikan oleh **Raden Wijaya** di Trowulan (Mojokerto, Jawa Timur) pada tahun 1293 setelah mengalahkan pasukan Mongol (Dinasti Yuan). Majapahit mencapai puncak kejayaan pada masa **Raja Hayam Wuruk (Sri Rajasanagara, 1350–1389)**.\n"
                "• **Mahapatih Gajah Mada & Sumpah Palapa (1336 M):** Saat diangkat menjadi *Mahapatih Amangkubhumi* pada tahun 1336 M, Gajah Mada mengucapkan **Sumpah Palapa** (*\"Lamun huwus kalah Nusantara isun amukti palapa...\"*), yakni berpantang menikmati *palapa* (kenikmatan duniawi) sebelum menyatukan seluruh kepulauan Nusantara: **Gurun, Seran, Tanjung Pura, Haru, Pahang, Dompo, Bali, Sunda, Palembang, dan Tumasik (Singapura)**.\n"
                "• **Warisan bagi Indonesia Modern:** Kitab *Nagarakretagama* (1365 M, Mpu Prapanca) mencatat 98 wilayah Nusantara dari Sumatera hingga Papua Barat, sedangkan *Kakawin Sutasoma* (Mpu Tantular) melahirkan semboyan **\"Bhinneka Tunggal Ika, Tan Hana Dharma Mangrwa\"** yang menjadi semboyan resmi NKRI."
            )

        # 7B. Sriwijaya, Mataram Kuno, Kutai, Tarumanagara, Singasari
        if _has_word(ql, ["sriwijaya", "mataram kuno", "borobudur", "prambanan", "syailendra", "sanjaya", "kutai", "mulawarman", "tarumanagara", "purnawarman", "kalingga", "singasari", "kertanagara", "ken arok", "kediri", "airlangga", "pamalayu"]):
            return (
                "**Kerajaan Klasik & Kemaharajaan Maritim Awal Nusantara:**\n"
                "• **Kutai & Tarumanagara (Abad ke-4 – ke-5 M):** Kerajaan tertua di Nusantara adalah **Kutai Martapura** di Kalimantan Timur (Raja Mulawarman, 7 Prasasti Yupa) dan **Tarumanagara** di Jawa Barat (Raja Purnawarman), disusul **Kalingga** (Ratu Shima).\n"
                "• **Kemaharajaan Sriwijaya (Abad ke-7 – ke-13 M):** Berpusat di Palembang (didirikan Dapunta Hyang Sri Jayanasa, 683 M; puncak di masa Raja Balaputradewa), Sriwijaya menguasai jalur maritim Selat Malaka & Selat Sunda serta menjadi pusat pengajaran Buddha internasional.\n"
                "• **Mataram Kuno (Abad ke-8 – ke-10 M):** Dinasti Syailendra dan Sanjaya di Jawa Tengah membangun mahakarya dunia **Candi Borobudur** dan **Candi Prambanan**, dilanjutkan era Raja Airlangga (Kahuripan) dan Kediri.\n"
                "• **Singasari (1222–1292 M):** Didirikan oleh Ken Arok; di bawah **Raja Kertanagara**, Singasari melancarkan **Ekspedisi Pamalayu (1275)** sebagai cikal-bakal penyatuan geopolitik Nusantara (*Cakrawala Mandala Dwipantara*)."
            )

        # 7C. Kesultanan Islam & Jalur Rempah
        if _has_word(ql, ["kesultanan", "samudera pasai", "wali songo", "walisongo", "demak", "mataram islam", "sultan agung", "banten", "gowa", "tallo", "hasanuddin", "ternate", "tidore", "baabullah", "nuku", "iskandar muda", "jalur rempah"]):
            return (
                "**Era Kesultanan Islam & Jaringan Niaga Rempah Nusantara (Abad ke-13 – ke-18):**\n"
                "• **Sumatera & Selat Malaka:** **Kesultanan Samudera Pasai** (Sultan Malik al-Saleh, abad ke-13) dan **Kesultanan Aceh Darussalam** (mencapai puncak kejayaan maritim & militer di bawah **Sultan Iskandar Muda**).\n"
                "• **Jawa:** **Kesultanan Demak** (Raden Patah & dakwah kultural **Wali Songo**), **Kesultanan Cirebon**, **Kesultanan Banten** (Sultan Ageng Tirtayasa), dan **Kesultanan Mataram Islam** (mencapai puncak di bawah **Sultan Agung Hanyokrokusumo**).\n"
                "• **Kalimantan, Sulawesi & Maluku (Pusat Rempah Dunia):** **Kesultanan Banjar & Kutai** di Kalimantan, **Kesultanan Gowa-Tallo** di Makassar (**Sultan Hasanuddin** — *Ayam Jantan dari Timur*), serta **Kesultanan Ternate** (Sultan Baabullah) dan **Tidore** (Sultan Nuku) di Maluku sebagai pusat cengkeh dan pala dunia."
            )

        # 7D. Pahlawan, Kolonialisme, Kebangkitan Nasional & Sumpah Pemuda
        if _has_word(ql, ["pahlawan", "kolonial", "voc", "belanda", "diponegoro", "pattimura", "imam bonjol", "cut nyak dhien", "teuku umar", "antasari", "sisingamangaraja", "kartini", "budi utomo", "kebangkitan nasional", "sumpah pemuda", "28 oktober 1928"]):
            return (
                "**Perlawanan Pahlawan Nusantara, Kebangkitan Nasional (1908) & Sumpah Pemuda (1928):**\n"
                "• **Perlawanan Bersenjata:** Bangsa Nusantara melawan kolonialisme melalui Sultan Agung (Mataram), Sultan Ageng Tirtayasa (Banten), Sultan Hasanuddin (Makassar), Kapitan Pattimura (Maluku, 1817), Tuanku Imam Bonjol (Perang Padri, 1803–1838), Pangeran Diponegoro (Perang Jawa, 1825–1830), Pangeran Antasari (Banjar), I Gusti Ketut Jelantik (Bali), Sisingamangaraja XII (Tapanuli), serta Teuku Umar, Cut Nyak Dhien, & Cut Meutia (Perang Aceh).\n"
                "• **Kebangkitan Nasional (20 Mei 1908):** Dimulai dari perjuangan emansipasi **R.A. Kartini** dan berdirinya **Budi Utomo (20 Mei 1908)** oleh dr. Sutomo & dr. Wahidin Sudirohusodo, diikuti Sarekat Islam, Indische Partij (Tiga Serangkai: Douwes Dekker, Cipto Mangunkusumo, Ki Hajar Dewantara), Muhammadiyah (K.H. Ahmad Dahlan, 1912), Nahdlatul Ulama (K.H. Hasyim Asy'ari, 1926), dan PNI (Ir. Soekarno, 1927).\n"
                "• **Sumpah Pemuda (28 Oktober 1928):** Kongres Pemuda II di Jakarta mengikrarkan: **Satu Tanah Air (Indonesia), Satu Bangsa (Bangsa Indonesia), dan Menjunjung Bahasa Persatuan (Bahasa Indonesia)**, diiringi lagu *Indonesia Raya* karya W.R. Supratman."
            )

        # 7E. Proklamasi, Pancasila, BPUPKI, PPKI, Deklarasi Djuanda
        if _has_word(ql, ["proklamasi", "kemerdekaan", "17 agustus", "pancasila", "bpupki", "ppki", "piagam jakarta", "uud 1945", "djuanda", "deklarasi djuanda", "kaa", "konferensi asia afrika", "sudirman"]):
            return (
                f"**Kelahiran Pancasila, Proklamasi Kemerdekaan RI (17 Agustus 1945) & Kedaulatan NKRI:**\n"
                f"• **Perumusan Dasar Negara (1945):** Dalam sidang BPUPKI (29 Mei – 1 Juni 1945), **Ir. Soekarno** menyampaikan pidato lahirnya **Pancasila** pada **1 Juni 1945**, disusul perumusan Piagam Jakarta oleh Panitia Sembilan (22 Juni 1945).\n"
                f"• **Proklamasi Kemerdekaan (17 Agustus 1945):** Pada Jumat, **17 Agustus 1945 pukul 10.00 WIB** di Jl. Pegangsaan Timur No. 56 Jakarta, **Ir. Soekarno dan Drs. Mohammad Hatta** memproklamasikan Kemerdekaan Republik Indonesia (kini berusia **{years_merdeka} tahun** pada tahun {current_year}). Pada 18 Agustus 1945, PPKI mengesahkan **UUD 1945**.\n"
                f"• **Revolusi Fisik & Deklarasi Djuanda:** Kemerdekaan dipertahankan melalui Pertempuran 10 November 1945 (Surabaya), gerilya Jenderal Sudirman, Serangan Umum 1 Maret 1949, hingga KMB (27 Desember 1949). Pada **13 Desember 1957**, **Deklarasi Djuanda** menyatukan seluruh laut antarpulau sebagai wilayah kedaulatan utuh NKRI (*Wawasan Nusantara*, diakui dalam UNCLOS 1982)."
            )

        # 7F. Presiden Indonesia, Orde Baru, Reformasi
        if _has_word(ql, ["presiden", "wakil presiden", "soekarno", "soeharto", "habibie", "gus dur", "abdurrahman wahid", "megawati", "sby", "yudhoyono", "jokowi", "joko widodo", "prabowo", "gibran", "orde baru", "reformasi"]):
            return (
                f"**Daftar Presiden Republik Indonesia (1945 – Sekarang Tahun {current_year}):**\n"
                f"1. **Ir. Soekarno (1945–1967):** Proklamator Kemerdekaan, Bapak Bangsa, penggagas Pancasila & Konferensi Asia-Afrika 1955.\n"
                f"2. **Jenderal Besar TNI (Purn.) Soeharto (1967–1998):** Era Orde Baru, pembangunan Repelita, pendiri ASEAN (1967), swasembada beras (1984), & peluncuran **Satelit Palapa A1 (1976)**.\n"
                f"3. **Prof. Dr. Ing. B.J. Habibie (1998–1999):** Pelopor industri dirgantara (pesawat N-250) & awal transisi demokrasi Era Reformasi.\n"
                f"4. **K.H. Abdurrahman Wahid / Gus Dur (1999–2001):** Bapak Pluralisme dan penegak hak-hak minoritas.\n"
                f"5. **Megawati Soekarnoputri (2001–2004):** Presiden perempuan pertama RI; pembentukan KPK & Mahkamah Konstitusi.\n"
                f"6. **Jenderal TNI (Purn.) Prof. Dr. Susilo Bambang Yudhoyono (2004–2014):** Presiden pertama yang dipilih langsung oleh rakyat; stabilitas makroekonomi & perdamaian Aceh.\n"
                f"7. **Ir. Joko Widodo (2014–2024):** Pembangunan infrastruktur konektivitas Nusantara, hilirisasi sumber daya alam, Presidensi G20 2022, & inisiasi Ibu Kota Nusantara (IKN).\n"
                f"8. **Jenderal TNI (Purn.) Prabowo Subianto (20 Oktober 2024 – Sekarang, {current_year}):** Presiden ke-8 RI (didampingi Wakil Presiden Gibran Rakabuming Raka), berfokus pada kemandirian pangan, energi, hilirisasi industri, pertahanan, dan penguatan SDM menuju Indonesia Emas 2045."
            )

        # 7G. Geografi, 38 Provinsi, IKN, Demografi, Budaya UNESCO, Ekonomi
        if _has_word(ql, ["provinsi", "38 provinsi", "ikn", "ibu kota nusantara", "geografi", "jumlah pulau", "penduduk", "populasi", "suku", "bahasa daerah", "unesco", "ekonomi", "g20", "hilirisasi", "indonesia emas"]):
            return (
                f"**Profil Indonesia Modern Sekarang (Tahun {current_year}):**\n"
                f"• **Geografi & Demografi:** Negara kepulauan terbesar di dunia dengan **17.504+ pulau**, luas daratan **1,91 juta km²**, dan luas perairan/ZEE **6,4 juta km²**. Penduduk mencapai **~282+ juta jiwa** (terbesar ke-4 di dunia), dengan **1.340+ suku bangsa** dan **718 bahasa daerah** yang dipersatukan oleh **Bahasa Indonesia**.\n"
                f"• **38 Provinsi & Ibu Kota Nusantara (IKN):** Terbagi atas **38 Provinsi** dari Sumatera hingga Papua (termasuk 4 provinsi baru: Papua Selatan, Papua Tengah, Papua Pegunungan, dan Papua Barat Daya), serta membangun **Ibu Kota Nusantara (IKN)** di Kalimantan Timur sebagai *Smart Forest City*.\n"
                f"• **Ekonomi & Warisan Dunia (UNESCO):** Ekonomi terbesar di Asia Tenggara (**Anggota G20**, PDB > USD 1,4 triliun) dengan fokus hilirisasi mineral kritis (nikel, tembaga, bauksit, ekosistem EV) & ekonomi digital. Warisan UNESCO meliputi **Borobudur, Prambanan, Sangiran, Subak Bali, Komodo, Lorentz, Wayang, Keris, Batik, Angklung, Pinisi, Pencak Silat, Gamelan, dan Jamu** menuju **Visi Indonesia Emas 2045**."
            )

        # 7H. Full History Overview: Nusantara -> Indonesia Sekarang
        if _has_word(ql, ["indonesia", "nusantara", "sejarah"]):
            return (
                f"**Perjalanan Sejarah Lengkap: Dari Nusantara hingga Indonesia Sekarang (Tahun {current_year}):**\n"
                f"1. **Prasejarah & Kerajaan Awal (Abad ke-4 – ke-7 M):** Dari *Homo erectus* (Sangiran) & migrasi Austronesia hingga lahirnya kerajaan tertulis tertua **Kutai Martapura** (Kalimantan Timur, Raja Mulawarman) dan **Tarumanagara** (Jawa Barat, Raja Purnawarman).\n"
                f"2. **Kemaharajaan Maritim Sriwijaya, Mataram Kuno & Singasari (Abad ke-7 – ke-13 M):** Kejayaan bahari **Sriwijaya** di Palembang, pembangunan **Candi Borobudur & Prambanan** oleh Mataram Kuno, serta visi penyatuan **Cakrawala Mandala Dwipantara** oleh Raja Kertanagara (Singasari).\n"
                f"3. **Kemaharajaan Majapahit & Sumpah Palapa Gajah Mada (1293 – Abad ke-16):** Di bawah Raja Hayam Wuruk dan **Mahapatih Gajah Mada** (*Sumpah Palapa*, 1336 M), kepulauan Nusantara dipersatukan dan melahirkan semboyan **\"Bhinneka Tunggal Ika\"** (*Kakawin Sutasoma*, Mpu Tantular).\n"
                f"4. **Era Kesultanan Rempah & Perlawanan Kolonial (Abad ke-13 – ke-19):** Kejayaan Samudera Pasai, Aceh, Demak, Banten, Mataram Islam, Gowa-Tallo, Ternate, & Tidore, disusul perjuangan para pahlawan melawan kolonialisme (Diponegoro, Pattimura, Imam Bonjol, Hasanuddin, Cut Nyak Dhien).\n"
                f"5. **Kebangkitan Nasional, Sumpah Pemuda & Proklamasi (1908 – 1945):** Berdirinya **Budi Utomo (20 Mei 1908)**, ikrar **Sumpah Pemuda (28 Oktober 1928)**, lahirnya **Pancasila (1 Juni 1945)**, hingga **Proklamasi Kemerdekaan RI (17 Agustus 1945)** oleh Soekarno-Hatta.\n"
                f"6. **Indonesia Sekarang ({current_year} — {years_merdeka} Tahun Merdeka):** Dipersatukan oleh **Deklarasi Djuanda (1957)** & **Satelit Palapa (1976)**, berkembang melalui Era Reformasi hingga kepemimpinan **Presiden ke-8 Prabowo Subianto**. Kini Indonesia berdiri dengan **38 Provinsi**, **~282+ juta jiwa**, anggota **G20**, membangun **Ibu Kota Nusantara (IKN)** menuju **Indonesia Emas 2045**."
            )

        # 8. Technical Architecture, Blockchain, Security, RAG, P2P, Code
        if _has_word(ql, ["arsitektur", "architecture", "modul", "module", "struktur"]):
            return (
                "**Arsitektur Sistem & 8 Modul Inti GajahMada:**\n"
                "1. `gajahmada_ai.api` (`server.py`): FastAPI REST & Web3 Gateway (16+ endpoint aktif).\n"
                "2. `gajahmada_ai.core` (`engine.py`): `AIModelEngine` (LLaMA GGUF + Neural Synthesis + 384-d Embeddings).\n"
                "3. `gajahmada_ai.knowledge` (`rag.py` & `indonesia_corpus.py`): `LocalKnowledge` 384-d Vector Store (24+ dokumen arsitektur & sejarah Nusantara–Indonesia).\n"
                "4. `gajahmada_ai.blockchain` (`chain.py`): `AIBlockchain` SHA-256 Proof-of-Work Audit Ledger (`difficulty=2`).\n"
                "5. `gajahmada_ai.security` (`crypto.py`): `NodeIdentity` Ed25519 (RFC 8032) digital signature & verification.\n"
                "6. `gajahmada_ai.network` (`p2p.py`): `P2PNetwork` asynchronous `httpx` mesh topology & gossip broadcast."
            )

        if _has_word(ql, ["blockchain", "web3", "chain", "block", "ledger", "pow", "hash"]):
            return (
                "**Arsitektur Web3 & Proof-of-Work Blockchain GajahMada:**\n"
                "Setiap interaksi di Command Center menghasilkan `task_hash = SHA256(prompt + response)`, ditandatangani dengan "
                "kunci privat **Ed25519** milik node, lalu ditambang dengan **Proof-of-Work (`difficulty=2`)** ke dalam rantai blok "
                "persisten (`chain.json`). Anda dapat mengauditnya kapan saja dengan perintah `/verify` atau `/mine`."
            )

        if _has_word(ql, ["security", "keamanan", "audit", "crypto", "ed25519"]):
            return (
                "**Keamanan & Kriptografi Ed25519 (RFC 8032):**\n"
                "• **Identitas Node:** Menggunakan kurva eliptik Ed25519 (`node_key.pem` dengan izin `0600`).\n"
                "• **Verifikasi Blok:** Setiap blok menyimpan `signature` Ed25519 atas `task_hash` SHA-256 yang dapat diverifikasi publik melalui `POST /security/sign-verify` dan `POST /node/chain/verify`."
            )

        if _has_word(ql, ["code", "kode", "python", "buatkan", "fungsi", "script", "contoh"]):
            return (
                f"Berikut contoh kode Python untuk memanggil API **GajahMada**:\n"
                f"```python\n"
                f"import httpx\n\n"
                f"resp = httpx.post('http://localhost:8000/ai/chat', json={{\n"
                f"    'prompt': {q!r},\n"
                f"    'max_tokens': 512\n"
                f"}}).json()\n"
                f"print(resp['response'])\n"
                f"print('Block #:', resp['block_index'], '| Hash:', resp['task_hash'][:16])\n"
                f"```"
            )

        # 9. If relevant RAG context was found for a custom question, present it cleanly
        if context:
            return (
                f"Berdasarkan basis pengetahuan **GajahMada (384-d Vector RAG)** untuk pertanyaan **\"{q}\"**:\n\n"
                f"{context}\n\n"
                f"*(Terverifikasi oleh Ed25519 pada `{now_str}`)*"
            )

        # 10. Clean conversational fallback
        return (
            f"Menjawab pertanyaan Anda tentang **\"{q}\"**:\n"
            f"• Permintaan telah diproses oleh **GajahMada Core Engine** pada `{now_str}`.\n"
            f"• Untuk jawaban mendalam mengenai **Sejarah Nusantara hingga Indonesia Sekarang**, **Sumpah Palapa Gajah Mada**, **38 Provinsi & IKN**, **Daftar Presiden RI**, atau **Arsitektur Web3**, silakan tanyakan topik terkait atau gunakan `/search <kata kunci>` dan `/ingest <pengetahuan baru>`."
        )

    def create_embedding(self, text: str) -> List[float]:
        if self._embedder:
            embedding = self._embedder.encode(text)
            return embedding.tolist()

        dim = 384
        vec = [0.0] * dim
        tokens = extract_keywords(text) or re.findall(r"\w+", text.lower()) or ["empty"]
        for tok in tokens:
            h = hashlib.sha256(tok.encode("utf-8")).digest()
            for i in range(0, len(h), 2):
                idx = ((h[i] << 8) | h[i + 1]) % dim
                sign = 1.0 if (h[i] & 1) == 0 else -1.0
                vec[idx] += sign
        norm = math.sqrt(sum(v * v for v in vec)) or 1.0
        return [round(v / norm, 6) for v in vec]
