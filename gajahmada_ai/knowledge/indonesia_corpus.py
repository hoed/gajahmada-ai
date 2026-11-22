"""
GajahMada — Korpus Pengetahuan Nusantara & Indonesia Modern
Deterministic per calendar year to prevent duplicate timestamp pollution in vector stores.
"""

from datetime import datetime


def get_indonesia_knowledge_corpus():
    now = datetime.now().astimezone()
    current_year = now.year
    years_independent = max(1, current_year - 1945)

    return [
        (
            "Sejarah Prasejarah & Peradaban Awal Nusantara: Kepulauan Nusantara merupakan salah satu pusat "
            "evolusi manusia tertua di dunia dengan temuan Homo erectus (Pithecanthropus erectus) di Sangiran dan Trinil "
            "(Jawa) serta Homo floresiensis di Liang Bua (Flores). Gelombang migrasi penutur Austronesia sekitar 2000 SM "
            "membawa teknologi pelayaran bercadik, pertanian padi, tradisi Megalitikum (punden berundak, menhir, sarkofagus), "
            "dan kebudayaan perunggu Dongson yang menjadikan Nusantara jaringan maritim awal antara Samudra Hindia dan Pasifik."
        ),
        (
            "Kerajaan Hindu-Buddha Awal Nusantara (Abad ke-4 – ke-7 M): Sejarah tertulis Nusantara dimulai pada abad ke-4 M "
            "melalui tujuh Prasasti Yupa beraksara Pallawa berbahasa Sanskerta di Kerajaan Kutai Martapura (Kalimantan Timur) "
            "pada masa Raja Mulawarman. Di Jawa Barat berkembang Kerajaan Tarumanagara (abad ke-5 M) di bawah Raja Purnawarman "
            "(Prasasti Ciaruteun, Tugu) yang membangun kanal irigasi Sungai Gomati, disusul Kerajaan Kalingga di Jawa Tengah "
            "yang dipimpin secara adil oleh Ratu Shima pada abad ke-7 M."
        ),
        (
            "Kemaharajaan Bahari Sriwijaya & Mataram Kuno (Abad ke-7 – ke-11 M): Kerajaan Sriwijaya yang berpusat di Palembang "
            "(Sumatera Selatan), didirikan oleh Dapunta Hyang Sri Jayanasa (Prasasti Kedukan Bukit, 683 M) dan mencapai puncak "
            "di bawah Raja Balaputradewa, menguasai jalur perdagangan Selat Malaka dan Selat Sunda serta menjadi pusat studi "
            "agama Buddha bertaraf internasional (dikunjungi biksu I-Tsing). Sementara di Jawa Tengah, Dinasti Syailendra dan "
            "Sanjaya (Mataram Kuno) membangun mahakarya arsitektur dunia Candi Borobudur (abad ke-8/9 M) dan Candi Prambanan, "
            "sebelum pusat kekuasaan bergeser ke Jawa Timur pada masa Mpu Sindok, Raja Dharmawangsa, Raja Airlangga (Kahuripan), "
            "dan Kerajaan Kediri (Raja Jayabaya)."
        ),
        (
            "Kerajaan Singasari & Visi Cakrawala Mandala Dwipantara (1222 – 1292 M): Didirikan oleh Ken Arok (Sri Rajasa) di "
            "Malang, Jawa Timur, setelah mengalahkan Kediri di Ganter (1222). Singasari mencapai puncak visi geopolitik pada masa "
            "Raja Kertanagara (1268–1292) yang menggagas penyatuan kerajaan-kerajaan Nusantara melalui Ekspedisi Pamalayu (1275) "
            "ke Melayu/Dharmasraya untuk membendung ekspansi Kekaisaran Mongol (Dinasti Yuan di bawah Kubilai Khan)."
        ),
        (
            "Kemaharajaan Majapahit & Mahapatih Gajah Mada (1293 – Abad ke-16): Didirikan oleh Raden Wijaya (Kertarajasa "
            "Jayawardhana) pada tahun 1293 di Trowulan, Mojokerto, setelah memukul mundur armada Mongol. Majapahit mencapai "
            "puncak kejayaan pada masa Raja Hayam Wuruk (Sri Rajasanagara, 1350–1389) bersama Mahapatih Gajah Mada (menjabat "
            "Mahapatih Amangkubhumi 1336–1364). Pada tahun 1336 M, Gajah Mada mengucapkan Sumpah Palapa (Amukti Palapa): "
            "berpantang menikmati palapa (kenikmatan duniawi) sebelum berhasil menyatukan Nusantara—mencakup Gurun, Seran, "
            "Tanjung Pura, Haru, Pahang, Dompo, Bali, Sunda, Palembang, dan Tumasik (Singapura)."
        ),
        (
            "Warisan Konstitusi, Sastra & Semboyan Bhinneka Tunggal Ika dari Majapahit: Pada masa Majapahit lahir kitab hukum "
            "Kutara Manawa Dharmasastra, sistem administrasi pemerintahan terpadu, mata uang Gobog, serta armada laut di bawah "
            "Laksamana Mpu Nala. Kitab Nagarakretagama (Desawarnana, 1365 M) karya Mpu Prapanca mencatat 98 wilayah cakupan "
            "Nusantara dari Sumatera, Semenanjung Malaya, Kalimantan, Sulawesi, Nusa Tenggara, Maluku, hingga Papua Barat (Wanin). "
            "Sementara Kakawin Sutasoma karya Mpu Tantular melahirkan semboyan 'Bhinneka Tunggal Ika, Tan Hana Dharma Mangrwa' "
            "(Berbeda-beda tetapi tetap satu jua, tidak ada kebenaran yang mendua) yang kini menjadi semboyan resmi Negara "
            "Kesatuan Republik Indonesia."
        ),
        (
            "Era Kesultanan Islam & Jalur Rempah Nusantara (Abad ke-13 – ke-18): Masuknya Islam melalui jaringan perdagangan "
            "maritim melahirkan kesultanan-kesultanan besar: Kesultanan Samudera Pasai (Sultan Malik al-Saleh, abad ke-13), "
            "Kesultanan Aceh Darussalam (puncak di bawah Sultan Iskandar Muda), Kesultanan Demak di Jawa bersama dakwah kultural "
            "Wali Songo, Kesultanan Cirebon dan Banten (Sultan Ageng Tirtayasa), Kesultanan Mataram Islam (Sultan Agung "
            "Hanyokrokusumo), Kesultanan Banjar dan Kutai di Kalimantan, Kesultanan Gowa-Tallo di Sulawesi Selatan (Sultan "
            "Hasanuddin, 'Ayam Jantan dari Timur'), serta Kesultanan Ternate (Sultan Baabullah) dan Tidore (Sultan Nuku) di "
            "Maluku sebagai pusat perdagangan cengkeh dan pala dunia."
        ),
        (
            "Perlawanan Pahlawan Nusantara Terhadap Kolonialisme (Abad ke-16 – Awal Abad ke-20): Kedatangan bangsa Eropa "
            "(Portugis 1511, Spanyol, VOC Belanda 1602, dan pemerintah kolonial Hindia Belanda) memicu perlawanan heroik di "
            "seluruh kepulauan: serangan Sultan Agung ke Batavia (1628–1629), perlawanan Sultan Baabullah & Kapitan Pattimura "
            "Thomas Matulessy di Maluku (1817), Perang Padri di Sumatera Barat (Tuanku Imam Bonjol, 1803–1838), Perang Jawa "
            "yang dipimpin Pangeran Diponegoro (1825–1830), Perang Banjar (Pangeran Antasari), Perang Puputan di Bali (I Gusti "
            "Ketut Jelantik), perlawanan Sisingamangaraja XII di Tapanuli, serta Perang Aceh (1873–1904) yang dipimpin Sultan "
            "Muhammad Daud Syah, Teuku Umar, Panglima Polem, Cut Nyak Dhien, dan Cut Meutia."
        ),
        (
            "Kebangkitan Nasional & Sumpah Pemuda (1908 – 1942): Pada awal abad ke-20, perjuangan beralih ke organisasi modern "
            "dan pendidikan: emansipasi R.A. Kartini, berdirinya Budi Utomo (20 Mei 1908 — Hari Kebangkitan Nasional) oleh "
            "dr. Sutomo dan dr. Wahidin Sudirohusodo, Sarekat Islam (H.O.S. Tjokroaminoto), Indische Partij (Tiga Serangkai: "
            "E.F.E. Douwes Dekker, dr. Cipto Mangunkusumo, dan Ki Hajar Dewantara pendiri Taman Siswa), Muhammadiyah (K.H. Ahmad "
            "Dahlan, 1912), Nahdlatul Ulama (K.H. Hasyim Asy'ari, 1926), Perhimpunan Indonesia (Mohammad Hatta, Sutan Sjahrir), "
            "dan PNI (Ir. Soekarno, 1927). Puncaknya adalah Kongres Pemuda II di Jakarta pada 28 Oktober 1928 yang melahirkan "
            "Sumpah Pemuda: bertumpah darah satu Tanah Air Indonesia, berbangsa satu Bangsa Indonesia, dan menjunjung bahasa "
            "persatuan Bahasa Indonesia, serta dikumandangkannya lagu Indonesia Raya oleh W.R. Supratman."
        ),
        (
            "BPUPKI, Kelahiran Pancasila, PPKI & Proklamasi Kemerdekaan RI (1942 – 17 Agustus 1945): Pada masa pendudukan Jepang "
            "(1942–1945), dibentuk BPUPKI (Dokuritsu Junbi Cosakai) yang bersidang merumuskan dasar negara (29 Mei – 1 Juni 1945: "
            "Moh. Yamin, Prof. Soepomo, dan pidato Ir. Soekarno pada 1 Juni 1945 tentang Pancasila), dilanjutkan Panitia Sembilan "
            "(Piagam Jakarta, 22 Juni 1945) dan PPKI. Pada Jumat, 17 Agustus 1945 pukul 10.00 WIB di Jl. Pegangsaan Timur No. 56 "
            "Jakarta, Ir. Soekarno dan Drs. Mohammad Hatta atas nama bangsa Indonesia memproklamasikan Kemerdekaan Republik "
            "Indonesia. Sehari kemudian (18 Agustus 1945), PPKI mengesahkan UUD 1945 serta menetapkan Ir. Soekarno sebagai "
            "Presiden pertama dan Drs. Mohammad Hatta sebagai Wakil Presiden pertama RI."
        ),
        (
            "Revolusi Kemerdekaan, Deklarasi Djuanda & Konferensi Asia-Afrika (1945 – 1965): Bangsa Indonesia mempertahankan "
            "kemerdekaan melalui perjuangan bersenjata (Pertempuran Surabaya 10 November 1945, Bandung Lautan Api, Palagan "
            "Ambarawa, Puputan Margarana I Gusti Ngurah Rai, perang gerilya Panglima Besar Jenderal Sudirman, dan Serangan Umum "
            "1 Maret 1949 di Yogyakarta) serta diplomasi (Perjanjian Linggarjati, Renville, Roem-Roijen, hingga pengakuan "
            "kedaulatan di Konferensi Meja Bundar 27 Desember 1949). Pada 18–24 April 1955 Indonesia menyelenggarakan Konferensi "
            "Asia-Afrika (KAA) di Bandung (Dasasila Bandung). Pada 13 Desember 1957, PM Djuanda Kartawidjaja mengumumkan "
            "Deklarasi Djuanda yang menyatukan seluruh perairan antarpulau sebagai wilayah kedaulatan penuh NKRI (konsep negara "
            "kepulauan / Archipelagic State yang kemudian diakui dunia dalam UNCLOS 1982)."
        ),
        (
            "Era Orde Baru, Satelit Palapa & Pembangunan Nasional (1966 – 1998): Pada masa pemerintahan Presiden Soeharto "
            "(Presiden ke-2 RI), Indonesia melaksanakan pembangunan bertahap (Repelita), menjadi salah satu dari 5 negara "
            "pendiri ASEAN (Deklarasi Bangkok, 8 Agustus 1967 oleh Adam Malik), meluncurkan sistem komunikasi Satelit Palapa A1 "
            "pada 8 Juli 1976 (mengambil nama dari Sumpah Palapa Gajah Mada untuk menyatukan jaringan telekomunikasi Nusantara), "
            "mencapai swasembada beras (penghargaan FAO 1984), serta mengembangkan industri dirgantara nasional (IPTN / pesawat "
            "N-250 yang dipimpin Prof. Dr. Ing. B.J. Habibie), sebelum berakhir akibat krisis moneter Asia dan Gerakan Reformasi "
            "Mei 1998."
        ),
        (
            f"Era Reformasi & Daftar Presiden Republik Indonesia (1998 – Sekarang Tahun {current_year}): Era Reformasi (sejak 1998) "
            f"membawa demokratisasi, amandemen UUD 1945 (1999–2002), kebebasan pers, otonomi daerah, pembentukan Mahkamah "
            f"Konstitusi & KPK, serta pemilihan presiden secara langsung sejak 2004. Urutan Presiden Republik Indonesia hingga "
            f"tahun {current_year} ({years_independent} tahun Indonesia merdeka): (1) Ir. Soekarno (1945–1967), (2) Soeharto "
            f"(1967–1998), (3) Prof. Dr. Ing. B.J. Habibie (1998–1999), (4) K.H. Abdurrahman Wahid / Gus Dur (1999–2001), "
            f"(5) Megawati Soekarnoputri (2001–2004), (6) Jenderal TNI (Purn.) Prof. Dr. Susilo Bambang Yudhoyono (2004–2014), "
            f"(7) Ir. Joko Widodo (2014–2024), dan (8) Jenderal TNI (Purn.) Prabowo Subianto (dilantik 20 Oktober 2024 – sekarang "
            f"di tahun {current_year}, didampingi Wakil Presiden Gibran Rakabuming Raka)."
        ),
        (
            f"Geografi, Demografi, 38 Provinsi & Ibu Kota Nusantara (IKN) Tahun {current_year}: Indonesia sekarang "
            f"adalah negara kepulauan terbesar di dunia yang terletak di garis khatulistiwa (6° LU – 11° LS "
            f"dan 95° BT – 141° BT), memiliki lebih dari 17.504 pulau (5 pulau besar: Sumatera, Jawa, Kalimantan, Sulawesi, dan "
            f"Papua), luas daratan sekitar 1,91 juta km², dan luas perairan/ZEE sekitar 6,4 juta km². Jumlah penduduk Indonesia "
            f"mencapai sekitar 282+ juta jiwa (terbesar ke-4 di dunia), terdiri atas lebih dari 1.340 suku bangsa dan 718 bahasa "
            f"daerah yang dipersatukan oleh Bahasa Indonesia. Secara administratif, NKRI terbagi menjadi 38 Provinsi (termasuk "
            f"Papua Selatan, Papua Tengah, Papua Pegunungan, dan Papua Barat Daya) serta membangun Ibu Kota Nusantara (IKN) di "
            f"Kalimantan Timur sebagai Smart Forest City yang hijau dan berkelanjutan."
        ),
        (
            f"Ekonomi Modern, Hilirisasi, Budaya UNESCO & Visi Indonesia Emas 2045 (Status Tahun {current_year}): Indonesia adalah "
            f"ekonomi terbesar di Asia Tenggara (PDB > USD 1,4 triliun) dan anggota G20. Strategi ekonomi Indonesia tahun "
            f"{current_year} berfokus pada hilirisasi sumber daya alam strategis (nikel, tembaga, bauksit, ekosistem baterai & "
            f"kendaraan listrik), ketahanan pangan & energi, ekonomi digital, serta kedaulatan AI & teknologi. Warisan Indonesia "
            f"yang diakui UNESCO meliputi Candi Borobudur, Candi Prambanan, Situs Sangiran, Subak Bali, Tambang Ombilin Sawahlunto, "
            f"Sumbu Filosofi Yogyakarta, Taman Nasional Komodo, Lorentz, Hutan Hujan Tropis Sumatera, Ujung Kulon, serta Warisan "
            f"Budaya Takbenda (Wayang, Keris, Batik, Angklung, Tari Saman, Noken, Pinisi, Pencak Silat, Pantun, Gamelan, Jamu) "
            f"menuju Visi Indonesia Emas 2045 (100 tahun kemerdekaan RI)."
        ),
    ]


INDONESIA_KNOWLEDGE_CORPUS = get_indonesia_knowledge_corpus()
