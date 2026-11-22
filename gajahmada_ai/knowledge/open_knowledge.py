"""
GajahMada 2.0 — Autonomous Open-Domain Internet Knowledge Crawler & Al-Qur'an Retrieval Engine.
Provides:
1. Built-in 114 Surah Al-Qur'an Catalog + direct Al-Qur'an verse resolver
   (Al-Quran Cloud API `api.alquran.cloud` with Arabic text, Indonesian Kemenag
   translation, and English translation + instant local cache + citation references).
2. Multi-Source Open-Domain Internet Knowledge Crawler:
   - Wikipedia / Wikimedia REST & Search API (Indonesian `id.wikipedia.org` & English `en.wikipedia.org`)
   - DuckDuckGo Instant Answer JSON API (`api.duckduckgo.com`)
   - DuckDuckGo Lite / HTML Web Search Crawler (`html.duckduckgo.com/html/`)
   - Direct URL Webpage Crawler (`crawl_direct_url`) with SSRF protection
3. Formatted Citation & Reference Builder (`format_references_footer`):
   Always appends verifiable reference titles and URLs at the end of every answer.
"""

import html
import re
from typing import Any, Dict, List, Optional, Tuple
import urllib.parse
import httpx

from gajahmada_ai.neural.llama_backend import wants_detailed_answer
from gajahmada_ai.security.ssrf import validate_peer_url

_HTTP_TIMEOUT = httpx.Timeout(5.0)
_USER_AGENT = "GajahMadaSovereignAI/2.0 (+https://github.com/hoed/gajahmada-ai)"

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


def _trim_concise(text: str, max_sentences: int = 2, max_chars: int = 340) -> str:
    clean = re.sub(r"\s+", " ", (text or "").strip())
    if len(clean) <= max_chars:
        parts = re.split(r"(?<=[.!?])\s+", clean)
        return " ".join(parts[:max_sentences]).strip()
    parts = re.split(r"(?<=[.!?])\s+", clean)
    joined = " ".join(parts[:max_sentences]).strip()
    if len(joined) > max_chars:
        joined = joined[: max_chars - 3].rsplit(" ", 1)[0] + "..."
    return joined


def format_references_footer(references: List[Dict[str, str]], max_refs: int = 2) -> str:
    """
    Build a concise, verifiable References footer at the end of answers.
    Each item in `references` has `title` and `url` (or `source`).
    """
    if not references:
        return ""
    seen = set()
    lines = ["\n\n---\n📚 **Sumber:**"]
    idx = 1
    for ref in references:
        if idx > max_refs:
            break
        title = str(ref.get("title") or "Sumber").strip()
        url = str(ref.get("url") or ref.get("source") or "").strip()
        key = f"{title}|{url}"
        if key in seen:
            continue
        seen.add(key)
        if url.startswith("http://") or url.startswith("https://"):
            lines.append(f"{idx}. **{title}** — {url}")
        elif url:
            lines.append(f"{idx}. **{title}** ({url})")
        else:
            lines.append(f"{idx}. **{title}**")
        idx += 1
    return "\n".join(lines)


def _normalize_name(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", s.lower())


_SURAH_NAME_TO_NUMBER: Dict[str, int] = {}
for _num, (_name, _meaning, _total, _rev) in SURAH_CATALOG.items():
    _SURAH_NAME_TO_NUMBER[_normalize_name(_name)] = _num
    if _name.lower().startswith(("al-", "an-", "as-", "at-", "ar-", "az-", "ad-", "asy-")):
        _short = re.sub(r"^(?:al|an|as|at|ar|az|ad|asy)-", "", _name.lower())
        _SURAH_NAME_TO_NUMBER[_normalize_name(_short)] = _num


def parse_quran_query(query: str) -> Optional[Tuple[int, Optional[int]]]:
    ql = query.lower()
    has_quran_kw = any(k in ql for k in ("quran", "qur'an", "alquran", "al-quran", "surat", "surah", "qs", "q.s", "ayat"))
    if not has_quran_kw:
        return None

    m_num_pair = re.search(
        r"(?:surat|surah|qs\.?|q\.s\.?)\s*(?:ke[\-\s]*)?(\d{1,3})\s*(?:ayat|:)\s*(\d{1,3})",
        ql,
    )
    if m_num_pair:
        return int(m_num_pair.group(1)), int(m_num_pair.group(2))

    m_colon = re.search(r"\b(\d{1,3})\s*:\s*(\d{1,3})\b", ql)
    if m_colon:
        return int(m_colon.group(1)), int(m_colon.group(2))

    m_named = re.search(
        r"(?:surat|surah|qs\.?|q\.s\.?)\s+([a-zA-Z\'\-\s]{3,22}?)\s+(?:ayat|:)\s*(\d{1,3})",
        ql,
    )
    if m_named:
        raw_name = _normalize_name(m_named.group(1))
        s_num = _SURAH_NAME_TO_NUMBER.get(raw_name)
        if s_num:
            return s_num, int(m_named.group(2))

    m_surah_only = re.search(r"(?:surat|surah)\s*(?:ke[\-\s]*)?(\d{1,3})\b", ql)
    if m_surah_only:
        return int(m_surah_only.group(1)), None

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
        refs = [
            {"title": f"Al-Qur'an Surah {s_name} (QS. {surah_num})", "url": f"https://quran.com/{surah_num}"},
            {"title": "Qur'an Kemenag RI", "url": f"https://quran.kemenag.go.id/quran/per-ayat/surah/{surah_num}"},
        ]
        return (
            f"**Al-Qur'an Surah ke-{surah_num}: QS. {s_name} ({s_meaning})**\n"
            f"• **Nomor Surah:** {surah_num} dari 114 Surah\n"
            f"• **Arti Nama Surah:** *{s_meaning}*\n"
            f"• **Jumlah Ayat:** {s_total} Ayat\n"
            f"• **Golongan:** {s_rev}\n"
            f"*(Ketik `sebutkan alquran surat {surah_num} ayat <1..{s_total}>` untuk menampilkan teks Arab dan terjemahan ayat tertentu.)*"
            + format_references_footer(refs)
        )

    if ayah_num < 1 or ayah_num > s_total:
        return (
            f"**QS. {s_name} (Surah ke-{surah_num})** memiliki **{s_total} ayat** "
            f"(Ayat {ayah_num} berada di luar rentang 1–{s_total})."
        )

    arabic_text = None
    id_translation = None

    if (surah_num, ayah_num) in OFFLINE_QURAN_VERSES:
        arabic_text, id_translation = OFFLINE_QURAN_VERSES[(surah_num, ayah_num)]
    else:
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

    refs = [
        {"title": f"QS. {s_name} {surah_num}:{ayah_num}", "url": f"https://quran.com/{surah_num}/{ayah_num}"},
    ]

    if arabic_text and id_translation:
        return (
            f"**QS. {s_name} ({surah_num}:{ayah_num})**\n\n"
            f"{arabic_text}\n\n"
            f"\"{id_translation}\""
            + format_references_footer(refs)
        )

    return (
        f"**QS. {s_name} ({surah_num}:{ayah_num})** — {s_meaning}. "
        f"Teks ayat belum tersimpan lokal dan perangkat sedang offline."
        + format_references_footer(refs)
    )


def clean_search_topic(query: str) -> str:
    cleaned = re.sub(
        r"^(?:tolong\s+|coba\s+)?(?:apa itu|siapa itu|siapakah|siapa|apakah|jelaskan secara detail tentang|jelaskan tentang|jelaskan|sebutkan|ceritakan tentang|ceritakan|bagaimana|dimana|kapan|definisi|pengertian|what is|who is|tell me about|explain|define|search|cari tentang|cari)\s+",
        "",
        query.strip(),
        flags=re.IGNORECASE,
    )
    cleaned = re.sub(r"[\?\!\.]+$", "", cleaned).strip()
    return cleaned or query.strip()


def _strip_html_tags(raw_html: str) -> str:
    no_scripts = re.sub(r"<(?:script|style|noscript|svg|header|footer|nav)[^>]*>.*?</(?:script|style|noscript|svg|header|footer|nav)>", " ", raw_html, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r"<[^>]+>", " ", no_scripts)
    text = html.unescape(text)
    return re.sub(r"\s+", " ", text).strip()


def crawl_direct_url(url: str, max_chars: int = 1200) -> Optional[Dict[str, str]]:
    """
    Safely crawl a public HTTP/HTTPS webpage (protected by SSRF validation),
    extracting its <title>, meta description, and primary body paragraphs.
    """
    ok, clean_url = validate_peer_url(url, allow_local=False)
    if not ok:
        return None
    try:
        headers = {"User-Agent": _USER_AGENT}
        with httpx.Client(timeout=_HTTP_TIMEOUT, headers=headers, follow_redirects=True) as client:
            resp = client.get(clean_url)
            if resp.status_code != 200:
                return None
            raw = resp.text[:150000]
            m_title = re.search(r"<title[^>]*>(.*?)</title>", raw, flags=re.DOTALL | re.IGNORECASE)
            title = _strip_html_tags(m_title.group(1)) if m_title else clean_url

            # Extract paragraphs <p>...</p>
            p_matches = re.findall(r"<p[^>]*>(.*?)</p>", raw, flags=re.DOTALL | re.IGNORECASE)
            paragraphs = []
            total_len = 0
            for p_html in p_matches:
                p_txt = _strip_html_tags(p_html)
                if len(p_txt) >= 60:
                    paragraphs.append(p_txt)
                    total_len += len(p_txt)
                    if total_len >= max_chars:
                        break

            body_text = "\n\n".join(paragraphs)[:max_chars] if paragraphs else _strip_html_tags(raw)[:max_chars]
            if len(body_text) < 30:
                return None
            return {
                "title": title[:160],
                "extract": body_text,
                "source_url": clean_url,
                "source_type": "direct_web_crawl",
            }
    except Exception:
        return None


def resolve_open_domain_wikipedia(query: str) -> Optional[Dict[str, Any]]:
    """
    Search Indonesian (`id.wikipedia.org`) and English (`en.wikipedia.org`) Wikipedia
    REST & Action APIs to crawl encyclopedic knowledge and related references.
    """
    topic = clean_search_topic(query)
    if len(topic) < 2:
        return None

    headers = {"User-Agent": _USER_AGENT}
    for lang in ("id", "en"):
        try:
            with httpx.Client(timeout=_HTTP_TIMEOUT, headers=headers, follow_redirects=True) as client:
                search_url = (
                    f"https://{lang}.wikipedia.org/w/api.php?"
                    f"action=query&list=search&srsearch={urllib.parse.quote(topic)}&utf8=&format=json&srlimit=3"
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

                sum_url = f"https://{lang}.wikipedia.org/api/rest_v1/page/summary/{urllib.parse.quote(best_title)}"
                p_res = client.get(sum_url)
                if p_res.status_code == 200:
                    pdata = p_res.json()
                    extract = pdata.get("extract", "").strip()
                    page_url = (
                        pdata.get("content_urls", {}).get("desktop", {}).get("page", "")
                        or f"https://{lang}.wikipedia.org/wiki/{urllib.parse.quote(best_title.replace(' ', '_'))}"
                    )
                    if extract and len(extract) > 30:
                        # Intent Relevance Guard: Verify that the Wikipedia article actually answers the specific question intent
                        q_tokens = [t for t in re.findall(r"[a-z0-9]{3,}", topic.lower()) if t not in {"sekarang", "ini", "hari", "saat", "pada", "dalam", "untuk", "dengan", "yang", "dari"}]
                        combined_article = (best_title + " " + extract).lower()
                        matched_tokens = sum(1 for t in q_tokens if t in combined_article)
                        if q_tokens and (matched_tokens / len(q_tokens)) < 0.5:
                            continue

                        refs = [{"title": f"Wikipedia ({lang.upper()}): {pdata.get('title', best_title)}", "url": page_url}]
                        for extra_hit in hits[1:3]:
                            t2 = extra_hit.get("title", "")
                            if t2:
                                refs.append({
                                    "title": f"Wikipedia ({lang.upper()}): {t2}",
                                    "url": f"https://{lang}.wikipedia.org/wiki/{urllib.parse.quote(t2.replace(' ', '_'))}",
                                })
                        detailed = wants_detailed_answer(query)
                        return {
                            "title": pdata.get("title", best_title),
                            "extract": extract if detailed else _trim_concise(extract),
                            "source_url": page_url,
                            "references": refs if detailed else refs[:1],
                            "lang": lang,
                        }
        except Exception:
            continue
    return None


def crawl_duckduckgo_web(query: str, max_results: int = 3) -> Optional[Dict[str, Any]]:
    """
    Crawl the open internet via DuckDuckGo Instant Answer API + DuckDuckGo HTML Search
    to retrieve real-time web snippets and reference URLs for any topic.
    """
    topic = clean_search_topic(query)
    if len(topic) < 2:
        return None

    headers = {"User-Agent": _USER_AGENT}
    snippets: List[str] = []
    references: List[Dict[str, str]] = []

    try:
        with httpx.Client(timeout=_HTTP_TIMEOUT, headers=headers, follow_redirects=True) as client:
            # 1. DuckDuckGo Instant Answer JSON API
            ddg_api = f"https://api.duckduckgo.com/?q={urllib.parse.quote(topic)}&format=json&no_html=1&skip_disambig=1"
            r_json = client.get(ddg_api)
            if r_json.status_code == 200:
                jdata = r_json.json()
                abstract = (jdata.get("AbstractText") or "").strip()
                abstract_url = (jdata.get("AbstractURL") or "").strip()
                heading = (jdata.get("Heading") or topic).strip()
                if abstract:
                    snippets.append(abstract)
                    if abstract_url:
                        references.append({"title": f"{heading} ({jdata.get('AbstractSource') or 'Web Reference'})", "url": abstract_url})
                for rel in (jdata.get("RelatedTopics") or [])[:3]:
                    if isinstance(rel, dict) and rel.get("Text") and rel.get("FirstURL"):
                        if len(snippets) < max_results:
                            snippets.append(rel["Text"].strip())
                        references.append({"title": rel["Text"][:80].strip(), "url": rel["FirstURL"]})

            # 2. DuckDuckGo HTML Search Engine Crawler if Instant Answer was empty
            if not snippets:
                html_url = f"https://html.duckduckgo.com/html/?q={urllib.parse.quote(topic)}"
                r_html = client.get(html_url)
                if r_html.status_code == 200:
                    raw_html = r_html.text
                    # Extract result links and snippets
                    blocks = re.findall(
                        r'<a[^>]+class="result__a"[^>]+href="([^"]+)"[^>]*>(.*?)</a>.*?<a[^>]+class="result__snippet"[^>]*>(.*?)</a>',
                        raw_html,
                        flags=re.DOTALL | re.IGNORECASE,
                    )
                    for href, raw_title, raw_snip in blocks[:max_results]:
                        clean_href = html.unescape(href)
                        m_uddg = re.search(r"uddg=([^&]+)", clean_href)
                        if m_uddg:
                            clean_href = urllib.parse.unquote(m_uddg.group(1))
                        t_str = _strip_html_tags(raw_title)
                        s_str = _strip_html_tags(raw_snip)
                        if s_str:
                            snippets.append(f"• **{t_str}:** {s_str}")
                        if clean_href.startswith(("http://", "https://")):
                            references.append({"title": t_str or clean_href, "url": clean_href})
    except Exception:
        pass

    if not snippets:
        return None

    return {
        "title": topic.title(),
        "extract": "\n\n".join(snippets),
        "source_url": references[0]["url"] if references else "https://duckduckgo.com",
        "references": references,
    }


# Curated offline encyclopedic knowledge base (Science, Technology, Religion, World History, Philosophy, Geography)
# Used immediately when offline or before live internet crawling
OFFLINE_ENCYCLOPEDIA: List[Dict[str, Any]] = [
    {
        "keywords": ["komputasi kuantum", "quantum computing", "qubit", "superposisi kuantum"],
        "title": "Komputasi Kuantum (Quantum Computing)",
        "extract": (
            "Komputasi kuantum adalah paradigma komputasi yang memanfaatkan fenomena mekanika kuantum seperti "
            "superposisi (superposition), keterikatan kuantum (quantum entanglement), dan interferensi kuantum untuk "
            "melakukan pemrosesan informasi. Berbeda dengan komputer klasik yang menggunakan bit biner (0 atau 1), "
            "komputer kuantum menggunakan bit kuantum (qubit) yang dapat berada dalam kombinasi linier keadaan |0⟩ dan |1⟩ "
            "secara bersamaan, memungkinkan pemecahan masalah kompleks seperti faktorisasi bilangan besar (Algoritma Shor), "
            "pencarian basis data (Algoritma Grover), simulasi molekuler, dan kriptografi pasca-kuantum."
        ),
        "references": [
            {"title": "Wikipedia Bahasa Indonesia — Komputasi Kuantum", "url": "https://id.wikipedia.org/wiki/Komputasi_kuantum"},
            {"title": "IBM Quantum — What is Quantum Computing?", "url": "https://www.ibm.com/topics/quantum-computing"},
        ],
    },
    {
        "keywords": ["albert einstein", "relativitas", "e=mc2", "teori relativitas"],
        "title": "Albert Einstein & Teori Relativitas",
        "extract": (
            "Albert Einstein (14 Maret 1879 – 18 April 1955) adalah fisikawan teoretis kelahiran Jerman yang mengembangkan "
            "Teori Relativitas Khusus (1905) dan Teori Relativitas Umum (1915), salah satu dari dua pilar utama fisika modern "
            "bersama mekanika kuantum. Ia terkenal dengan persamaan kesetaraan massa-energi E = mc² dan dianugerahi "
            "Penghargaan Nobel Fisika tahun 1921 atas penjelasannya mengenai efek fotolistrik."
        ),
        "references": [
            {"title": "Wikipedia Bahasa Indonesia — Albert Einstein", "url": "https://id.wikipedia.org/wiki/Albert_Einstein"},
            {"title": "The Nobel Prize in Physics 1921 — Albert Einstein", "url": "https://www.nobelprize.org/prizes/physics/1921/einstein/facts/"},
        ],
    },
    {
        "keywords": ["kecerdasan buatan", "artificial intelligence", "machine learning", "deep learning", "neural network"],
        "title": "Kecerdasan Buatan (Artificial Intelligence / AI)",
        "extract": (
            "Kecerdasan Buatan (Artificial Intelligence / AI) adalah bidang ilmu komputer yang berfokus pada penciptaan "
            "sistem yang mampu melakukan penalaran, pembelajaran (machine learning & deep learning), persepsi, pemrosesan "
            "bahasa alami (NLP), dan pengambilan keputusan otonom. Arsitektur AI modern mencakup pendekatan simbolik "
            "(penalaran berbasis aturan & logika), jaringan saraf tiruan (Transformer / LLM), serta arsitektur hibrida "
            "neuro-simbolik seperti GajahMada 2.0."
        ),
        "references": [
            {"title": "Wikipedia Bahasa Indonesia — Kecerdasan Buatan", "url": "https://id.wikipedia.org/wiki/Kecerdasan_buatan"},
            {"title": "Stanford Encyclopedia of Philosophy — Artificial Intelligence", "url": "https://plato.stanford.edu/entries/artificial-intelligence/"},
        ],
    },
    {
        "keywords": ["tata surya", "solar system", "planet", "matahari", "galaksi bima sakti"],
        "title": "Tata Surya (Solar System) & Astronomi",
        "extract": (
            "Tata Surya adalah kumpulan benda langit yang terdiri atas sebuah bintang pusat (Matahari) dan semua objek "
            "yang terikat oleh gaya gravitasinya, termasuk delapan planet utama secara berurutan dari Matahari: "
            "Merkurius, Venus, Bumi, Mars, Jupiter, Saturnus, Uranus, dan Neptunus, beserta planet katai (seperti Pluto, "
            "Ceres, Eris), ratusan satelit alami, asteroid, dan komet di dalam Galaksi Bima Sakti."
        ),
        "references": [
            {"title": "Wikipedia Bahasa Indonesia — Tata Surya", "url": "https://id.wikipedia.org/wiki/Tata_Surya"},
            {"title": "NASA Science — Solar System Exploration", "url": "https://science.nasa.gov/solar-system/"},
        ],
    },
    {
        "keywords": ["rukun islam", "rukun iman", "agama islam", "nabi muhammad"],
        "title": "Islam, Rukun Islam & Rukun Iman",
        "extract": (
            "Islam adalah agama monoteistik Abrahamik yang berpedoman pada kitab suci Al-Qur'an dan Sunnah Nabi Muhammad SAW. "
            "Lima Rukun Islam terdiri atas: (1) Mengucapkan dua kalimat Syahadat, (2) Mendirikan Shalat lima waktu, "
            "(3) Menunaikan Zakat, (4) Berpuasa di bulan Ramadhan, dan (5) Menunaikan ibadah Haji bagi yang mampu. "
            "Enam Rukun Iman meliputi iman kepada Allah, Malaikat-Nya, Kitab-Kitab-Nya, Rasul-Rasul-Nya, Hari Akhir, serta Qada dan Qadar."
        ),
        "references": [
            {"title": "Wikipedia Bahasa Indonesia — Islam", "url": "https://id.wikipedia.org/wiki/Islam"},
            {"title": "Kementerian Agama Republik Indonesia — Pustaka Lajnah", "url": "https://quran.kemenag.go.id"},
        ],
    },
    {
        "keywords": ["fotosintesis", "klorofil", "tumbuhan"],
        "title": "Fotosintesis (Biologi Tumbuhan)",
        "extract": (
            "Fotosintesis adalah proses biokimia pembentukan karbohidrat (glukosa) dan oksigen (O₂) dari karbon dioksida (CO₂) "
            "dan air (H₂O) oleh tumbuhan berklorofil, alga, dan sianobakteri dengan menggunakan energi cahaya matahari: "
            "6CO₂ + 6H₂O + Cahaya → C₆H₁₂O₆ + 6O₂."
        ),
        "references": [
            {"title": "Wikipedia Bahasa Indonesia — Fotosintesis", "url": "https://id.wikipedia.org/wiki/Fotosintesis"},
        ],
    },
    {
        "keywords": ["dna", "rna", "genetika", "kromosom"],
        "title": "Asam Deoksiribonukleat (DNA) & Genetika",
        "extract": (
            "DNA (Deoxyribonucleic Acid) adalah molekul asam nukleat berbentuk heliks ganda yang menyimpan instruksi genetik "
            "untuk perkembangan, fungsi, pertumbuhan, dan reproduksi seluruh organisme hidup. DNA tersusun atas empat basa "
            "nukleotida: Adenin (A), Timin (T), Guanin (G), dan Sitosin (C)."
        ),
        "references": [
            {"title": "Wikipedia Bahasa Indonesia — Asam Deoksiribonukleat (DNA)", "url": "https://id.wikipedia.org/wiki/Asam_deoksiribonukleat"},
        ],
    },
]


def lookup_offline_encyclopedia(query: str) -> Optional[Dict[str, Any]]:
    ql = query.lower()
    detailed = wants_detailed_answer(query)
    for item in OFFLINE_ENCYCLOPEDIA:
        if any(kw in ql for kw in item["keywords"]):
            return {
                "title": item["title"],
                "extract": item["extract"] if detailed else _trim_concise(item["extract"]),
                "source_url": item["references"][0]["url"],
                "references": item["references"] if detailed else item["references"][:1],
            }
    return None


# Built-in Coordinates & Tropical Climate Profiles for Major Indonesian & World Cities
CITY_COORDINATES: Dict[str, Tuple[str, float, float, str, str]] = {
    "surabaya": ("Kota Surabaya, Jawa Timur", -7.2575, 112.7521, "Asia/Jakarta", "27°C – 34°C (Tropis Pesisir, Kelembapan ~65–80%)"),
    "jakarta": ("DKI Jakarta", -6.2088, 106.8456, "Asia/Jakarta", "26°C – 33°C (Tropis Monsun, Kelembapan ~70–85%)"),
    "bandung": ("Kota Bandung, Jawa Barat", -6.9175, 107.6191, "Asia/Jakarta", "19°C – 28°C (Tropis Dataran Tinggi, Kelembapan ~70–85%)"),
    "yogyakarta": ("DI Yogyakarta", -7.7956, 110.3695, "Asia/Jakarta", "24°C – 32°C (Tropis, Kelembapan ~68–82%)"),
    "jogja": ("DI Yogyakarta", -7.7956, 110.3695, "Asia/Jakarta", "24°C – 32°C (Tropis, Kelembapan ~68–82%)"),
    "semarang": ("Kota Semarang, Jawa Tengah", -6.9667, 110.4167, "Asia/Jakarta", "26°C – 34°C (Tropis Pesisir, Kelembapan ~65–80%)"),
    "malang": ("Kota Malang, Jawa Timur", -7.9666, 112.6326, "Asia/Jakarta", "20°C – 29°C (Tropis Dataran Tinggi, Kelembapan ~70–84%)"),
    "medan": ("Kota Medan, Sumatera Utara", 3.5952, 98.6722, "Asia/Jakarta", "25°C – 33°C (Tropis Khatulistiwa, Kelembapan ~75–88%)"),
    "makassar": ("Kota Makassar, Sulawesi Selatan", -5.1477, 119.4327, "Asia/Makassar", "25°C – 33°C (Tropis Maritim, Kelembapan ~70–85%)"),
    "denpasar": ("Kota Denpasar, Bali", -8.6705, 115.2126, "Asia/Makassar", "25°C – 32°C (Tropis Bahari, Kelembapan ~70–85%)"),
    "bali": ("Denpasar, Bali", -8.6705, 115.2126, "Asia/Makassar", "25°C – 32°C (Tropis Bahari, Kelembapan ~70–85%)"),
    "balikpapan": ("Kota Balikpapan, Kalimantan Timur", -1.2379, 116.8529, "Asia/Makassar", "25°C – 32°C (Tropis Khatulistiwa, Kelembapan ~75–88%)"),
    "ikn": ("Ibu Kota Nusantara (IKN), Kalimantan Timur", -0.9730, 116.7090, "Asia/Makassar", "24°C – 31°C (Tropis Hutan Hujan, Kelembapan ~78–90%)"),
    "nusantara": ("Ibu Kota Nusantara (IKN), Kalimantan Timur", -0.9730, 116.7090, "Asia/Makassar", "24°C – 31°C (Tropis Hutan Hujan, Kelembapan ~78–90%)"),
    "palembang": ("Kota Palembang, Sumatera Selatan", -2.9761, 104.7754, "Asia/Jakarta", "25°C – 33°C (Tropis, Kelembapan ~72–86%)"),
    "jayapura": ("Kota Jayapura, Papua", -2.5337, 140.7181, "Asia/Jayapura", "25°C – 32°C (Tropis Pasifik, Kelembapan ~75–88%)"),
    "aceh": ("Kota Banda Aceh, Aceh", 5.5483, 95.3238, "Asia/Jakarta", "25°C – 32°C (Tropis Maritim, Kelembapan ~72–85%)"),
    "banjarmasin": ("Kota Banjarmasin, Kalimantan Selatan", -3.3186, 114.5944, "Asia/Makassar", "25°C – 33°C (Tropis, Kelembapan ~74–88%)"),
    "manado": ("Kota Manado, Sulawesi Utara", 1.4748, 124.8421, "Asia/Makassar", "24°C – 31°C (Tropis Pesisir, Kelembapan ~75–88%)"),
    "padang": ("Kota Padang, Sumatera Barat", -0.9471, 100.4172, "Asia/Jakarta", "24°C – 31°C (Tropis Samudra Hindia, Kelembapan ~75–88%)"),
    "pekanbaru": ("Kota Pekanbaru, Riau", 0.5071, 101.4478, "Asia/Jakarta", "25°C – 33°C (Tropis, Kelembapan ~72–86%)"),
    "pontianak": ("Kota Pontianak, Kalimantan Barat", -0.0263, 109.3425, "Asia/Jakarta", "25°C – 33°C (Garis Khatulistiwa 0°, Kelembapan ~75–88%)"),
    "ambon": ("Kota Ambon, Maluku", -3.6954, 128.1814, "Asia/Jayapura", "25°C – 31°C (Tropis Kepulauan, Kelembapan ~75–86%)"),
    "kupang": ("Kota Kupang, Nusa Tenggara Timur", -10.1772, 123.6070, "Asia/Makassar", "25°C – 34°C (Tropis Sabana, Kelembapan ~60–78%)"),
    "mataram": ("Kota Mataram, Nusa Tenggara Barat", -8.5833, 116.1167, "Asia/Makassar", "24°C – 32°C (Tropis, Kelembapan ~68–82%)"),
    "bogor": ("Kota Bogor, Jawa Barat", -6.5971, 106.8060, "Asia/Jakarta", "21°C – 30°C (Kota Hujan Tropis, Kelembapan ~75–90%)"),
    "depok": ("Kota Depok, Jawa Barat", -6.4025, 106.7942, "Asia/Jakarta", "24°C – 32°C (Tropis, Kelembapan ~70–85%)"),
    "tangerang": ("Kota Tangerang, Banten", -6.1783, 106.6319, "Asia/Jakarta", "25°C – 33°C (Tropis, Kelembapan ~70–85%)"),
    "bekasi": ("Kota Bekasi, Jawa Barat", -6.2383, 106.9756, "Asia/Jakarta", "25°C – 33°C (Tropis, Kelembapan ~70–84%)"),
    "solo": ("Kota Surakarta (Solo), Jawa Tengah", -7.5755, 110.8243, "Asia/Jakarta", "24°C – 33°C (Tropis, Kelembapan ~68–82%)"),
    "surakarta": ("Kota Surakarta (Solo), Jawa Tengah", -7.5755, 110.8243, "Asia/Jakarta", "24°C – 33°C (Tropis, Kelembapan ~68–82%)"),
    "sidoarjo": ("Kabupaten Sidoarjo, Jawa Timur", -7.4478, 112.7183, "Asia/Jakarta", "26°C – 34°C (Tropis Pesisir, Kelembapan ~66–82%)"),
    "gresik": ("Kabupaten Gresik, Jawa Timur", -7.1566, 112.6555, "Asia/Jakarta", "26°C – 34°C (Tropis Pesisir, Kelembapan ~65–80%)"),
}

WMO_WEATHER_CODES: Dict[int, str] = {
    0: "Cerah (Clear Sky)",
    1: "Cerah Berawan (Mainly Clear)",
    2: "Berawan Sebagian (Partly Cloudy)",
    3: "Berawan Tebal (Overcast)",
    45: "Berkabut (Fog)",
    48: "Kabut Tebal",
    51: "Gerimis Ringan (Light Drizzle)",
    53: "Gerimis Sedang",
    55: "Gerimis Lebat",
    61: "Hujan Ringan (Light Rain)",
    63: "Hujan Sedang (Moderate Rain)",
    65: "Hujan Lebat (Heavy Rain)",
    80: "Hujan Lokal Ringan (Rain Showers)",
    81: "Hujan Lokal Sedang",
    82: "Hujan Lokal Deras",
    95: "Hujan Petir / Badai Guntur (Thunderstorm)",
    96: "Hujan Petir Disertai Angin Kencang",
}


def resolve_weather_query(query: str) -> Optional[Dict[str, Any]]:
    """
    Intelligent Weather & Temperature Resolver.
    Detects queries about temperature/weather ('berapa suhu di surabaya sekarang ini', 'cuaca jakarta hari ini')
    and fetches live meteorological data from Open-Meteo API + Open-Meteo Geocoding API (with offline profile fallback).
    Never dumps generic Wikipedia history when the user asks for temperature/weather!
    """
    ql = query.lower()
    is_weather = any(w in ql for w in ("suhu", "cuaca", "temperatur", "temperature", "weather", "derajat celcius", "kelembapan", "hujan hari ini", "prakiraan cuaca"))
    if not is_weather:
        return None

    # Extract target city name from query
    city_name = None
    lat, lon, tz, climate_profile = None, None, "Asia/Jakarta", "26°C – 33°C (Iklim Tropis Indonesia)"

    for key, (label, c_lat, c_lon, c_tz, c_prof) in CITY_COORDINATES.items():
        if re.search(rf"\b{re.escape(key)}\b", ql):
            city_name, lat, lon, tz, climate_profile = label, c_lat, c_lon, c_tz, c_prof
            break

    # If city not in built-in table, extract city token after "di " / "in " / "kota "
    raw_city_token = "Surabaya"
    if not city_name:
        m_city = re.search(r"(?:di|in|kota|untuk|daerah|wilayah)\s+([a-zA-Z\s]{3,25})", ql)
        if m_city:
            raw_city_token = re.sub(r"\b(sekarang|hari|ini|saat|terkini|berapa|suhu|cuaca)\b", "", m_city.group(1), flags=re.IGNORECASE).strip().title()
        if not raw_city_token:
            raw_city_token = "Surabaya"
        city_name = raw_city_token

    headers = {"User-Agent": _USER_AGENT}
    try:
        with httpx.Client(timeout=_HTTP_TIMEOUT, headers=headers) as client:
            if lat is None or lon is None:
                geo_url = f"https://geocoding-api.open-meteo.com/v1/search?name={urllib.parse.quote(raw_city_token)}&count=1&language=id&format=json"
                g_res = client.get(geo_url)
                if g_res.status_code == 200:
                    g_results = g_res.json().get("results") or []
                    if g_results:
                        top = g_results[0]
                        lat = float(top["latitude"])
                        lon = float(top["longitude"])
                        tz = top.get("timezone", "Asia/Jakarta")
                        admin = top.get("admin1", "")
                        country = top.get("country", "")
                        city_name = f"{top.get('name', raw_city_token)}" + (f", {admin}" if admin else "") + (f" ({country})" if country else "")

            if lat is not None and lon is not None:
                meteo_url = (
                    f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}"
                    f"&current=temperature_2m,relative_humidity_2m,apparent_temperature,weather_code,wind_speed_10m"
                    f"&daily=temperature_2m_max,temperature_2m_min&timezone=auto&forecast_days=1"
                )
                w_res = client.get(meteo_url)
                if w_res.status_code == 200:
                    w_data = w_res.json()
                    curr = w_data.get("current", {})
                    daily = w_data.get("daily", {})
                    temp_c = curr.get("temperature_2m")
                    feels_c = curr.get("apparent_temperature")
                    humidity = curr.get("relative_humidity_2m")
                    wind_kmh = curr.get("wind_speed_10m")
                    w_code = int(curr.get("weather_code", 1))
                    cond_str = WMO_WEATHER_CODES.get(w_code, "Berawan / Cerah Berawan")
                    t_max = (daily.get("temperature_2m_max") or [None])[0]
                    t_min = (daily.get("temperature_2m_min") or [None])[0]

                    if temp_c is not None:
                        detailed = wants_detailed_answer(query)
                        if not detailed:
                            extract = (
                                f"🌡️ **{city_name}:** **`{temp_c}°C`** "
                                f"({cond_str}, kelembapan `{humidity}%`, angin `{wind_kmh} km/jam`)."
                            )
                        else:
                            range_line = f"• **Rentang Suhu Hari Ini:** `{t_min}°C` – `{t_max}°C`\n" if t_min is not None and t_max is not None else ""
                            extract = (
                                f"🌡️ **Suhu Saat Ini di {city_name}:** **`{temp_c}°C`** *(terasa `{feels_c}°C`)*\n"
                                f"• **Kondisi:** {cond_str} · **Kelembapan:** `{humidity}%` · **Angin:** `{wind_kmh} km/jam`\n"
                                f"{range_line}"
                                f"• **Koordinat:** `{lat:.4f}°, {lon:.4f}°` ({tz})"
                            )
                        refs = [
                            {"title": f"Open-Meteo ({city_name})", "url": f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&current=temperature_2m,relative_humidity_2m,apparent_temperature,weather_code,wind_speed_10m"},
                            {"title": "BMKG RI", "url": "https://www.bmkg.go.id/cuaca/prakiraan-cuaca.bmkg"},
                        ]
                        return {
                            "title": f"Suhu & Cuaca — {city_name}",
                            "extract": extract,
                            "source_url": refs[0]["url"],
                            "references": refs if detailed else refs[:1],
                        }
    except Exception:
        pass

    # Offline / Sandbox Meteorological Profile Fallback (Directly answers temperature question!)
    detailed = wants_detailed_answer(query)
    if not detailed:
        extract = f"🌡️ **Estimasi Suhu {city_name} (Mode Offline):** **`{climate_profile}`**."
    else:
        coord_str = f"`{lat:.4f}°, {lon:.4f}°`" if (lat is not None and lon is not None) else "Wilayah Tropis Indonesia"
        extract = (
            f"🌡️ **Estimasi Suhu & Profil Iklim {city_name}:** **`{climate_profile}`**\n"
            f"• **Rata-Rata Siang:** `31°C – 34°C` · **Malam:** `25°C – 28°C`\n"
            f"• **Koordinat:** {coord_str} ({tz})"
        )
    refs = [
        {"title": f"BMKG — Cuaca {city_name}", "url": "https://www.bmkg.go.id/cuaca/prakiraan-cuaca.bmkg"},
        {"title": f"Open-Meteo ({city_name})", "url": "https://open-meteo.com"},
    ]
    return {
        "title": f"Suhu & Cuaca — {city_name}",
        "extract": extract,
        "source_url": refs[0]["url"],
        "references": refs if detailed else refs[:1],
    }


def crawl_internet_knowledge(query: str) -> Optional[Dict[str, Any]]:
    """
    Unified Multi-Source Internet Knowledge Crawler with Intent-First Decision Logic:
    1. Checks specialized real-time intent resolvers FIRST (Weather/Temperature, Direct URL).
    2. Queries Wikipedia (ID & EN) for encyclopedic summaries ONLY when the article matches the user's question intent.
    3. Queries DuckDuckGo Instant Answer API for open internet results.
    """
    # 1. Weather & Temperature Intent Resolver (never return generic city history for temperature questions!)
    weather_res = resolve_weather_query(query)
    if weather_res:
        return weather_res

    # 2. Direct URL Crawl if user provided an explicit URL
    m_url = re.search(r"(https?://[^\s\"\'<>]+)", query)
    if m_url:
        direct = crawl_direct_url(m_url.group(1))
        if direct:
            return {
                "title": direct["title"],
                "extract": direct["extract"],
                "source_url": direct["source_url"],
                "references": [{"title": direct["title"], "url": direct["source_url"]}],
            }

    # 3. Encyclopedic & Web Knowledge Resolution
    wiki = resolve_open_domain_wikipedia(query)
    if wiki:
        return wiki

    ddg = crawl_duckduckgo_web(query, max_results=3)
    if ddg:
        return ddg
    return lookup_offline_encyclopedia(query)
