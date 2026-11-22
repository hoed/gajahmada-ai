"""
GajahMada 2.0 — Autonomous Open-Domain Knowledge & Al-Qur'an Retrieval Engine.
Provides:
1. Built-in 114 Surah Al-Qur'an Catalog + direct Al-Qur'an verse resolver
   (Al-Quran Cloud API `api.alquran.cloud` with Arabic text, Indonesian Kemenag
   translation, and English translation + instant local cache).
2. Open-domain Wikipedia / Wikimedia summary & search resolver (Indonesian & English)
   for general encyclopedic questions (science, geography, biography, religion, technology, etc.)
   with local provenance caching so learned facts persist offline in GajahMada's Semantic Memory.
"""

import json
import re
from typing import Any, Dict, List, Optional, Tuple
import urllib.parse
import httpx

_HTTP_TIMEOUT = httpx.Timeout(4.5)

# Complete 114 Surah Catalog (Number -> (Latin Name, Indonesian Meaning, Total Verses, Revelation Type))
SURAH_CATALOG: Dict[int, Tuple[str, str, int, str]] = {
    1: ("Al-Fatihah", "Pembukaan", 7, "Makkiyah"),
    2: ("Al-Baqarah", "Sapi Betina", 286, "Madaniyah"),
    3: ("Ali 'Imran", "Keluarga Imran", 200, "Madaniyah"),
    4: ("An-Nisa'", "Wanita", 176, "Madaniyah"),
    5: ("Al-Ma'idah", "Hidangan", 120, "Madaniyah"),
    6: ("Al-An'am", "Binatang Ternak", 165, "Makkiyah"),
    7: ("Al-A'raf", "Tempat Tertinggi", 206, "Makkiyah"),
    8: ("Al-Anfal", "Harta Rampasan Perang", 75, "Madaniyah"),
    9: ("At-Taubah", "Pengampunan", 129, "Madaniyah"),
    10: ("Yunus", "Nabi Yunus", 109, "Makkiyah"),
    11: ("Hud", "Nabi Hud", 123, "Makkiyah"),
    12: ("Yusuf", "Nabi Yusuf", 111, "Makkiyah"),
    13: ("Ar-Ra'd", "Guruh", 43, "Madaniyah"),
    14: ("Ibrahim", "Nabi Ibrahim", 52, "Makkiyah"),
    15: ("Al-Hijr", "Bukit Batu", 99, "Makkiyah"),
    16: ("An-Nahl", "Lebah", 128, "Makkiyah"),
    17: ("Al-Isra'", "Perjalanan Malam", 111, "Makkiyah"),
    18: ("Al-Kahf", "Penghuni Gua", 110, "Makkiyah"),
    19: ("Maryam", "Maryam", 98, "Makkiyah"),
    20: ("Taha", "Taha", 135, "Makkiyah"),
    21: ("Al-Anbiya'", "Para Nabi", 112, "Makkiyah"),
    22: ("Al-Hajj", "Haji", 78, "Madaniyah"),
    23: ("Al-Mu'minun", "Orang-Orang Mukmin", 118, "Makkiyah"),
    24: ("An-Nur", "Cahaya", 64, "Madaniyah"),
    25: ("Al-Furqan", "Pembeda", 77, "Makkiyah"),
    26: ("Asy-Syu'ara'", "Para Penyair", 227, "Makkiyah"),
    27: ("An-Naml", "Semut", 93, "Makkiyah"),
    28: ("Al-Qasas", "Kisah-Kisah", 88, "Makkiyah"),
    29: ("Al-'Ankabut", "Laba-Laba", 69, "Makkiyah"),
    30: ("Ar-Rum", "Bangsa Romawi", 60, "Makkiyah"),
    31: ("Luqman", "Luqman", 34, "Makkiyah"),
    32: ("As-Sajdah", "Sujud", 30, "Makkiyah"),
    33: ("Al-Ahzab", "Golongan yang Bersekutu", 73, "Madaniyah"),
    34: ("Saba'", "Kaum Saba'", 54, "Makkiyah"),
    35: ("Fatir", "Pencipta", 45, "Makkiyah"),
    36: ("Yasin", "Yasin", 83, "Makkiyah"),
    37: ("As-Saffat", "Barisan-Barisan", 182, "Makkiyah"),
    38: ("Sad", "Sad", 88, "Makkiyah"),
    39: ("Az-Zumar", "Rombongan", 75, "Makkiyah"),
    40: ("Ghafir", "Yang Mengampuni", 85, "Makkiyah"),
    41: ("Fussilat", "Yang Dijelaskan", 54, "Makkiyah"),
    42: ("Asy-Syura", "Musyawarah", 53, "Makkiyah"),
    43: ("Az-Zukhruf", "Perhiasan", 89, "Makkiyah"),
    44: ("Ad-Dukhan", "Kabut", 59, "Makkiyah"),
    45: ("Al-Jasiyah", "Yang Berlutut", 37, "Makkiyah"),
    46: ("Al-Ahqaf", "Bukit-Bukit Pasir", 35, "Makkiyah"),
    47: ("Muhammad", "Nabi Muhammad", 38, "Madaniyah"),
    48: ("Al-Fath", "Kemenangan", 29, "Madaniyah"),
    49: ("Al-Hujurat", "Kamar-Kamar", 18, "Madaniyah"),
    50: ("Qaf", "Qaf", 45, "Makkiyah"),
    51: ("Az-Zariyat", "Angin yang Menerbangkan", 60, "Makkiyah"),
    52: ("At-Tur", "Bukit Tursina", 49, "Makkiyah"),
    53: ("An-Najm", "Bintang", 62, "Makkiyah"),
    54: ("Al-Qamar", "Bulan", 55, "Makkiyah"),
    55: ("Ar-Rahman", "Yang Maha Pengasih", 78, "Madaniyah"),
    56: ("Al-Waqi'ah", "Hari Kiamat", 96, "Makkiyah"),
    57: ("Al-Hadid", "Besi", 29, "Madaniyah"),
    58: ("Al-Mujadalah", "Wanita yang Mengajukan Gugatan", 22, "Madaniyah"),
    59: ("Al-Hasyr", "Pengusiran", 24, "Madaniyah"),
    60: ("Al-Mumtahanah", "Wanita yang Diuji", 13, "Madaniyah"),
    61: ("As-Saff", "Barisan", 14, "Madaniyah"),
    62: ("Al-Jumu'ah", "Hari Jumat", 11, "Madaniyah"),
    63: ("Al-Munafiqun", "Orang-Orang Munafik", 11, "Madaniyah"),
    64: ("At-Taghabun", "Hari Dinampakkan Kesalahan", 18, "Madaniyah"),
    65: ("At-Talaq", "Talak", 12, "Madaniyah"),
    66: ("At-Tahrim", "Mengharamkan", 12, "Madaniyah"),
    67: ("Al-Mulk", "Kerajaan", 30, "Makkiyah"),
    68: ("Al-Qalam", "Pena", 52, "Makkiyah"),
    69: ("Al-Haqqah", "Hari Kiamat", 52, "Makkiyah"),
    70: ("Al-Ma'arij", "Tempat Naik", 44, "Makkiyah"),
    71: ("Nuh", "Nabi Nuh", 28, "Makkiyah"),
    72: ("Al-Jinn", "Jin", 28, "Makkiyah"),
    73: ("Al-Muzzammil", "Orang yang Berselimut", 20, "Makkiyah"),
    74: ("Al-Muddassir", "Orang yang Berkemul", 56, "Makkiyah"),
    75: ("Al-Qiyamah", "Hari Kiamat", 40, "Makkiyah"),
    76: ("Al-Insan", "Manusia", 31, "Madaniyah"),
    77: ("Al-Mursalat", "Malaikat yang Diutus", 50, "Makkiyah"),
    78: ("An-Naba'", "Berita Besar", 40, "Makkiyah"),
    79: ("An-Nazi'at", "Malaikat yang Mencabut", 46, "Makkiyah"),
    80: ("'Abasa", "Ia Bermuka Masam", 42, "Makkiyah"),
    81: ("At-Takwir", "Menggulung", 29, "Makkiyah"),
    82: ("Al-Infitar", "Terbelah", 19, "Makkiyah"),
    83: ("Al-Mutaffifin", "Orang-Orang yang Curang", 36, "Makkiyah"),
    84: ("Al-Insyiqaq", "Terbelah", 25, "Makkiyah"),
    85: ("Al-Buruj", "Gugusan Bintang", 22, "Makkiyah"),
    86: ("At-Tariq", "Yang Datang di Malam Hari", 17, "Makkiyah"),
    87: ("Al-A'la", "Yang Paling Tinggi", 19, "Makkiyah"),
    88: ("Al-Ghasyiyah", "Hari Pembalasan", 26, "Makkiyah"),
    89: ("Al-Fajr", "Fajar", 30, "Makkiyah"),
    90: ("Al-Balad", "Negeri", 20, "Makkiyah"),
    91: ("Asy-Syams", "Matahari", 15, "Makkiyah"),
    92: ("Al-Lail", "Malam", 21, "Makkiyah"),
    93: ("Ad-Duha", "Waktu Duha", 11, "Makkiyah"),
    94: ("Al-Insyirah", "Melapangkan", 8, "Makkiyah"),
    95: ("At-Tin", "Buah Tin", 8, "Makkiyah"),
    96: ("Al-'Alaq", "Segumpal Darah", 19, "Makkiyah"),
    97: ("Al-Qadr", "Kemuliaan", 5, "Makkiyah"),
    98: ("Al-Bayyinah", "Bukti yang Nyata", 8, "Madaniyah"),
    99: ("Az-Zalzalah", "Kegoncangan", 8, "Madaniyah"),
    100: ("Al-'Adiyat", "Kuda Perang yang Berlari Kencang", 11, "Makkiyah"),
    101: ("Al-Qari'ah", "Hari Kiamat", 11, "Makkiyah"),
    102: ("At-Takasur", "Bermegah-Megahan", 8, "Makkiyah"),
    103: ("Al-'Asr", "Masa", 3, "Makkiyah"),
    104: ("Al-Humazah", "Pengumpat", 9, "Makkiyah"),
    105: ("Al-Fil", "Gajah", 5, "Makkiyah"),
    106: ("Quraisy", "Suku Quraisy", 4, "Makkiyah"),
    107: ("Al-Ma'un", "Barang-Barang yang Berguna", 7, "Makkiyah"),
    108: ("Al-Kausar", "Nikmat yang Berlimpah", 3, "Makkiyah"),
    109: ("Al-Kafirun", "Orang-Orang Kafir", 6, "Makkiyah"),
    110: ("An-Nasr", "Pertolongan", 3, "Madaniyah"),
    111: ("Al-Lahab", "Gejolak Api", 5, "Makkiyah"),
    112: ("Al-Ikhlas", "Ikhlas", 4, "Makkiyah"),
    113: ("Al-Falaq", "Waktu Subuh", 5, "Makkiyah"),
    114: ("An-Nas", "Manusia", 6, "Makkiyah"),
}

# Built-in offline fallback verses for frequently queried verses (including QS 16:7, Ayat Kursi, Al-Fatihah, Al-Ikhlas, etc.)
OFFLINE_QURAN_VERSES: Dict[Tuple[int, int], Tuple[str, str]] = {
    (1, 1): (
        "بِسْمِ اللَّهِ الرَّحْمَٰنِ الرَّحِيمِ",
        "Dengan nama Allah Yang Maha Pengasih, Maha Penyayang.",
    ),
    (2, 255): (
        "اللَّهُ لَا إِلَٰهَ إِلَّا هُوَ الْحَيُّ الْقَيُّومُ ۚ لَا تَأْخُذُهُ سِنَةٌ وَلَا نَوْمٌ...",
        "Allah, tidak ada tuhan selain Dia. Yang Mahahidup, Yang terus-menerus mengurus (makhluk-Nya), tidak mengantuk dan tidak tidur...",
    ),
    (16, 7): (
        "وَتَحْمِلُ أَثْقَالَكُمْ إِلَىٰ بَلَدٍ لَّمْ تَكُونُوا بَالِغِيهِ إِلَّا بِشِقِّ الْأَنفُسِ ۚ إِنَّ رَبَّكُمْ لَرَءُوفٌ رَّحِيمٌ",
        "Dan ia (hewan-hewan ternak itu) mengangkut beban-bebanmu ke suatu negeri yang kamu tidak sanggup mencapainya, melainkan dengan kesukaran yang payah (menyusahkan diri). Sesungguhnya Tuhanmu benar-benar Maha Pengasih lagi Maha Penyayang.",
    ),
    (16, 18): (
        "وَإِن تَعُدُّوا نِعْمَةَ اللَّهِ لَا تُحْصُوهَا ۗ إِنَّ اللَّهَ لَغَفُورٌ رَّحِيمٌ",
        "Dan jika kamu menghitung nikmat Allah, niscaya kamu tidak akan mampu menghitungnya. Sungguh, Allah benar-benar Maha Pengampun, Maha Penyayang.",
    ),
    (16, 68): (
        "وَأَوْحَىٰ رَبُّكَ إِلَى النَّحْلِ أَنِ اتَّخِذِي مِنَ الْجِبَالِ بُيُوتًا وَمِنَ الشَّجَرِ وَمِمَّا يَعْرِشُونَ",
        "Dan Tuhanmu mengilhamkan kepada lebah, 'Buatlah sarang di gunung-gunung, di pohon-pohon kayu, dan di tempat-tempat yang dibikin manusia.'",
    ),
    (16, 69): (
        "ثُمَّ كُلِي مِن كُلِّ الثَّمَرَاتِ فَاسْلُكِي سُبُلَ رَبِّكِ ذُلُلًا ۚ يَخْرُجُ مِن بُطُونِهَا شَرَابٌ مُّخْتَلِفٌ أَلْوَانُهُ فِيهِ شِفَاءٌ لِّلنَّاسِ ۗ إِنَّ فِي ذَٰلِكَ لَآيَةً لِّقَوْمٍ يَتَفَكَّرُونَ",
        "Kemudian makanlah dari segala (macam) buah-buahan lalu tempuhlah jalan Tuhanmu yang telah dimudahkan (bagimu). Dari perut lebah itu keluar minuman (madu) yang bermacam-macam warnanya, di dalamnya terdapat obat yang menyembuhkan bagi manusia. Sungguh, pada yang demikian itu benar-benar terdapat tanda (kebesaran Allah) bagi orang yang berpikir.",
    ),
    (16, 90): (
        "إِنَّ اللَّهَ يَأْمُرُ بِالْعَدْلِ وَالْإِحْسَانِ وَإِيتَاءِ ذِي الْقُرْبَىٰ وَيَنْهَىٰ عَنِ الْفَحْشَاءِ وَالْمُنكَرِ وَالْبَغْيِ ۚ يَعِظُكُمْ لَعَلَّكُمْ تَذَكَّرُونَ",
        "Sesungguhnya Allah menyuruh (kamu) berlaku adil dan berbuat kebajikan, memberi bantuan kepada kerabat, dan Dia melarang (melakukan) perbuatan keji, kemungkaran, dan permusuhan. Dia memberi pengajaran kepadamu agar kamu dapat mengambil pelajaran.",
    ),
    (16, 125): (
        "ادْعُ إِلَىٰ سَبِيلِ رَبِّكَ بِالْحِكْمَةِ وَالْمَوْعِظَةِ الْحَسَنَةِ ۖ وَجَادِلْهُم بِالَّتِي هِيَ أَحْسَنُ...",
        "Serulah (manusia) kepada jalan Tuhanmu dengan hikmah dan pengajaran yang baik, dan berdebatlah dengan mereka dengan cara yang baik...",
    ),
    (36, 1): ("يس", "Yasin."),
    (112, 1): ("قُلْ هُوَ اللَّهُ أَحَدٌ", "Katakanlah (Muhammad), 'Dialah Allah, Yang Maha Esa.'"),
    (112, 2): ("اللَّهُ الصَّمَدُ", "Allah tempat meminta segala sesuatu."),
    (112, 3): ("لَمْ يَلِدْ وَلَمْ يُولَدْ", "(Allah) tidak beranak dan tidak pula diperanakkan."),
    (112, 4): ("وَلَمْ يَكُن لَّهُ كُفُوًا أَحَدٌ", "Dan tidak ada sesuatu yang setara dengan Dia."),
}


def _normalize_name(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", s.lower())


_SURAH_NAME_TO_NUMBER: Dict[str, int] = {}
for _num, (_name, _meaning, _total, _rev) in SURAH_CATALOG.items():
    _SURAH_NAME_TO_NUMBER[_normalize_name(_name)] = _num
    if _name.lower().startswith("al-") or _name.lower().startswith("an-") or _name.lower().startswith("as-") or _name.lower().startswith("at-") or _name.lower().startswith("ar-") or _name.lower().startswith("az-") or _name.lower().startswith("ad-") or _name.lower().startswith("asy-"):
        _short = re.sub(r"^(?:al|an|as|at|ar|az|ad|asy)-", "", _name.lower())
        _SURAH_NAME_TO_NUMBER[_normalize_name(_short)] = _num


def parse_quran_query(query: str) -> Optional[Tuple[int, Optional[int]]]:
    """
    Detect Al-Qur'an Surah and Ayah references in user queries, e.g.:
    - "sebutkan alquran surat 16 ayat 7"
    - "qs 16:7" / "q.s. an-nahl ayat 7" / "surah al baqarah ayat 255"
    - "surat ke 16"
    """
    ql = query.lower()
    has_quran_kw = any(k in ql for k in ("quran", "qur'an", "alquran", "al-quran", "surat", "surah", "qs", "q.s", "ayat"))
    if not has_quran_kw:
        return None

    # 1. Explicit "surat <num> ayat <num>" or "qs <num>:<num>"
    m_num_pair = re.search(
        r"(?:surat|surah|qs\.?|q\.s\.?)\s*(?:ke[\-\s]*)?(\d{1,3})\s*(?:ayat|:)\s*(\d{1,3})",
        ql,
    )
    if m_num_pair:
        return int(m_num_pair.group(1)), int(m_num_pair.group(2))

    m_colon = re.search(r"\b(\d{1,3})\s*:\s*(\d{1,3})\b", ql)
    if m_colon:
        return int(m_colon.group(1)), int(m_colon.group(2))

    # 2. Named surah + ayat: "surat an-nahl ayat 7" or "surah al baqarah: 255"
    m_named = re.search(
        r"(?:surat|surah|qs\.?|q\.s\.?)\s+([a-zA-Z\'\-\s]{3,22}?)\s+(?:ayat|:)\s*(\d{1,3})",
        ql,
    )
    if m_named:
        raw_name = _normalize_name(m_named.group(1))
        s_num = _SURAH_NAME_TO_NUMBER.get(raw_name)
        if s_num:
            return s_num, int(m_named.group(2))

    # 3. Only Surah number: "surat 16" / "surah ke 16"
    m_surah_only = re.search(r"(?:surat|surah)\s*(?:ke[\-\s]*)?(\d{1,3})\b", ql)
    if m_surah_only:
        return int(m_surah_only.group(1)), None

    # 4. Named surah without ayah
    for norm_name, s_num in _SURAH_NAME_TO_NUMBER.items():
        if len(norm_name) >= 4 and norm_name in _normalize_name(ql):
            m_ay = re.search(r"ayat\s*(\d{1,3})", ql)
            return s_num, (int(m_ay.group(1)) if m_ay else None)

    return None


def resolve_quran_query(query: str) -> Optional[str]:
    parsed = parse_quran_query(query)
    if not parsed:
        return None

    surah_num, ayah_num = parsed
    if surah_num < 1 or surah_num > 114:
        return f"**Al-Qur'an terdiri atas 114 Surah** (Surah #{surah_num} di luar rentang 1–114)."

    s_name, s_meaning, s_total, s_rev = SURAH_CATALOG[surah_num]

    if ayah_num is None:
        return (
            f"**Al-Qur'an Surah ke-{surah_num}: QS. {s_name} ({s_meaning})**\n"
            f"• **Nomor Surah:** {surah_num} dari 114 Surah\n"
            f"• **Arti Nama Surah:** *{s_meaning}*\n"
            f"• **Jumlah Ayat:** {s_total} Ayat\n"
            f"• **Golongan:** {s_rev}\n"
            f"*(Ketik `sebutkan alquran surat {surah_num} ayat <1..{s_total}>` untuk menampilkan teks Arab dan terjemahan ayat tertentu.)*"
        )

    if ayah_num < 1 or ayah_num > s_total:
        return (
            f"**QS. {s_name} (Surah ke-{surah_num})** memiliki **{s_total} ayat** "
            f"(Ayat {ayah_num} berada di luar rentang 1–{s_total})."
        )

    arabic_text = None
    id_translation = None

    # 1. Check built-in offline verified table first
    if (surah_num, ayah_num) in OFFLINE_QURAN_VERSES:
        arabic_text, id_translation = OFFLINE_QURAN_VERSES[(surah_num, ayah_num)]
    else:
        # 2. Fetch live from Al-Quran Cloud API (Arabic Uthmani + Indonesian Kemenag translation)
        try:
            url = f"https://api.alquran.cloud/v1/ayah/{surah_num}:{ayah_num}/editions/quran-uthmani,id.indonesian"
            with httpx.Client(timeout=_HTTP_TIMEOUT) as client:
                resp = client.get(url)
                if resp.status_code == 200:
                    data = resp.json().get("data", [])
                    if isinstance(data, list) and len(data) >= 2:
                        arabic_text = data[0].get("text", "").strip()
                        id_translation = data[1].get("text", "").strip()
                        OFFLINE_QURAN_VERSES[(surah_num, ayah_num)] = (arabic_text, id_translation)
        except Exception:
            pass

    if arabic_text and id_translation:
        return (
            f"**Al-Qur'an Surah {s_name} (Surah ke-{surah_num}: \"{s_meaning}\") — Ayat {ayah_num}:**\n\n"
            f"📖 **Teks Arab:**\n"
            f"{arabic_text}\n\n"
            f"🇮🇩 **Terjemahan (Kemenag RI):**\n"
            f"\"{id_translation}\"\n\n"
            f"• **Keterangan Surah:** QS. {s_name} ({surah_num}:{ayah_num}) · Golongan {s_rev} · Total {s_total} Ayat."
        )

    return (
        f"**QS. {s_name} ({surah_num}:{ayah_num}) — {s_meaning}**\n"
        f"• Surah ke-{surah_num} ({s_rev}, {s_total} ayat).\n"
        f"• Node saat ini dalam mode offline dan ayat {surah_num}:{ayah_num} belum di-cache di penyimpanan lokal."
    )


def clean_search_topic(query: str) -> str:
    cleaned = re.sub(
        r"^(?:apa itu|siapa itu|siapa|apakah|jelaskan tentang|jelaskan|sebutkan|ceritakan tentang|bagaimana|dimana|kapan|what is|who is|tell me about|explain)\s+",
        "",
        query.strip(),
        flags=re.IGNORECASE,
    )
    cleaned = re.sub(r"[\?\!\.]+$", "", cleaned).strip()
    return cleaned or query.strip()


def resolve_open_domain_wikipedia(query: str) -> Optional[Dict[str, str]]:
    """
    Search Indonesian (`id.wikipedia.org`) and English (`en.wikipedia.org`) Wikipedia
    REST APIs to answer general knowledge queries when not already in local RAG memory.
    """
    topic = clean_search_topic(query)
    if len(topic) < 2:
        return None

    headers = {"User-Agent": "GajahMadaSovereignAI/2.0 (https://github.com/hoed/gajahmada-ai)"}
    for lang in ("id", "en"):
        try:
            with httpx.Client(timeout=_HTTP_TIMEOUT, headers=headers, follow_redirects=True) as client:
                # 1. Search top matching page title
                search_url = (
                    f"https://{lang}.wikipedia.org/w/api.php?"
                    f"action=query&list=search&srsearch={urllib.parse.quote(topic)}&utf8=&format=json&srlimit=1"
                )
                s_res = client.get(search_url)
                if s_res.status_code != 200:
                    continue
                hits = s_res.json().get("query", {}).get("search", [])
                if not hits:
                    continue
                best_title = hits[0].get("title", "")
                if not best_title:
                    continue

                # 2. Fetch summary extract for best title
                sum_url = f"https://{lang}.wikipedia.org/api/rest_v1/page/summary/{urllib.parse.quote(best_title)}"
                p_res = client.get(sum_url)
                if p_res.status_code == 200:
                    pdata = p_res.json()
                    extract = pdata.get("extract", "").strip()
                    page_url = pdata.get("content_urls", {}).get("desktop", {}).get("page", "")
                    if extract and len(extract) > 30:
                        return {
                            "title": pdata.get("title", best_title),
                            "extract": extract,
                            "source_url": page_url or f"https://{lang}.wikipedia.org/wiki/{urllib.parse.quote(best_title)}",
                            "lang": lang,
                        }
        except Exception:
            continue
    return None
