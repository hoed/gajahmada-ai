"""
Curated world & space fact base (offline, bundled, no external AI dependency).

Scope & honesty: this is a *curated* set of stable, well-established facts (not "all facts in the
world", which no finite corpus can be). Anything outside it is handled by the live knowledge
crawler (when network is available) or an explicit "not found" answer — never a fabricated one.

Rules for entries: only stable facts; approximate values are phrased with "sekitar"; contested or
fast-changing facts (office holders, populations, records) are deliberately excluded.
Each sentence is self-contained so it can be retrieved and quoted on its own.
"""
from __future__ import annotations

from typing import List

# (country, capital note) — capitals only where the answer is uncontested or explicitly qualified.
_CAPITALS = [
    ("Indonesia", "Jakarta (saat ini); Ibu Kota Nusantara (IKN) di Kalimantan Timur sedang dibangun sebagai calon ibu kota baru berdasarkan UU No. 3 Tahun 2022"),
    ("Afghanistan", "Kabul"), ("Aljazair", "Aljir"), ("Amerika Serikat", "Washington, D.C."),
    ("Arab Saudi", "Riyadh"), ("Argentina", "Buenos Aires"), ("Australia", "Canberra"),
    ("Austria", "Wina"), ("Bangladesh", "Dhaka"), ("Belanda", "Amsterdam (ibu kota konstitusional; pusat pemerintahan di Den Haag)"),
    ("Belarus", "Minsk"), ("Belgia", "Brussel"), ("Bhutan", "Thimphu"), ("Brasil", "Brasília"),
    ("Brunei Darussalam", "Bandar Seri Begawan"), ("Bulgaria", "Sofia"), ("Chili", "Santiago"),
    ("Denmark", "Kopenhagen"), ("Ekuador", "Quito"), ("Estonia", "Tallinn"), ("Ethiopia", "Addis Ababa"),
    ("Fiji", "Suva"), ("Filipina", "Manila"), ("Finlandia", "Helsinki"), ("Ghana", "Accra"),
    ("Hongaria", "Budapest"), ("India", "New Delhi"), ("Inggris (Britania Raya)", "London"),
    ("Irak", "Baghdad"), ("Iran", "Teheran"), ("Irlandia", "Dublin"), ("Islandia", "Reykjavik"),
    ("Italia", "Roma"), ("Jepang", "Tokyo"), ("Jerman", "Berlin"), ("Kamboja", "Phnom Penh"),
    ("Kanada", "Ottawa"), ("Kazakhstan", "Astana"), ("Kenya", "Nairobi"), ("Kolombia", "Bogotá"),
    ("Korea Selatan", "Seoul"), ("Korea Utara", "Pyongyang"), ("Kroasia", "Zagreb"), ("Kuba", "Havana"),
    ("Kuwait", "Kota Kuwait"), ("Laos", "Vientiane"), ("Latvia", "Riga"), ("Libya", "Tripoli"),
    ("Lituania", "Vilnius"), ("Malaysia", "Kuala Lumpur"), ("Maladewa", "Malé"), ("Maroko", "Rabat"),
    ("Meksiko", "Kota Meksiko"), ("Mesir", "Kairo"), ("Mongolia", "Ulaanbaatar"), ("Myanmar", "Naypyidaw"),
    ("Nepal", "Kathmandu"), ("Nigeria", "Abuja"), ("Norwegia", "Oslo"), ("Pakistan", "Islamabad"),
    ("Papua Nugini", "Port Moresby"), ("Paraguay", "Asunción"), ("Peru", "Lima"), ("Polandia", "Warsawa"),
    ("Portugal", "Lisabon"), ("Prancis", "Paris"), ("Qatar", "Doha"), ("Republik Ceko", "Praha"),
    ("Rumania", "Bukares"), ("Rusia", "Moskow"), ("Selandia Baru", "Wellington"), ("Senegal", "Dakar"),
    ("Serbia", "Beograd"), ("Singapura", "Singapura (negara kota)"), ("Slovakia", "Bratislava"),
    ("Spanyol", "Madrid"), ("Sri Lanka", "Sri Jayawardenapura Kotte (Kolombo adalah kota terbesar)"),
    ("Sudan", "Khartoum"), ("Swedia", "Stockholm"), ("Swiss", "Bern"), ("Tanzania", "Dodoma"),
    ("Thailand", "Bangkok"), ("Timor Leste", "Dili"), ("Tiongkok (Cina)", "Beijing"), ("Tunisia", "Tunis"),
    ("Turki", "Ankara"), ("Uganda", "Kampala"), ("Ukraina", "Kyiv"), ("Uni Emirat Arab", "Abu Dhabi"),
    ("Uruguay", "Montevideo"), ("Uzbekistan", "Tashkent"), ("Venezuela", "Caracas"), ("Vietnam", "Hanoi"),
    ("Yordania", "Amman"), ("Yunani", "Athena"), ("Zambia", "Lusaka"), ("Zimbabwe", "Harare"),
    ("Afrika Selatan", "Pretoria (eksekutif), Cape Town (legislatif), dan Bloemfontein (yudikatif)"),
    ("Bolivia", "Sucre (konstitusional); pusat pemerintahan berada di La Paz"),
]

_CURRENCIES = [
    ("Indonesia", "Rupiah (IDR)"), ("Amerika Serikat", "Dolar Amerika Serikat (USD)"), ("Jepang", "Yen (JPY)"),
    ("Inggris (Britania Raya)", "Poundsterling (GBP)"), ("Tiongkok (Cina)", "Renminbi, satuannya yuan (CNY)"),
    ("India", "Rupee India (INR)"), ("Malaysia", "Ringgit (MYR)"), ("Singapura", "Dolar Singapura (SGD)"),
    ("Thailand", "Baht (THB)"), ("Korea Selatan", "Won Korea Selatan (KRW)"), ("Rusia", "Rubel (RUB)"),
    ("Brasil", "Real (BRL)"), ("Australia", "Dolar Australia (AUD)"), ("Arab Saudi", "Riyal Saudi (SAR)"),
    ("Swiss", "Franc Swiss (CHF)"), ("Filipina", "Peso Filipina (PHP)"), ("Vietnam", "Dong (VND)"),
    ("Kanada", "Dolar Kanada (CAD)"), ("Meksiko", "Peso Meksiko (MXN)"), ("Turki", "Lira Turki (TRY)"),
]

_INDONESIA = [
    "Lagu kebangsaan Indonesia adalah Indonesia Raya, diciptakan oleh Wage Rudolf Supratman dan pertama kali dikumandangkan pada Kongres Pemuda II tanggal 28 Oktober 1928.",
    "Bendera nasional Indonesia bernama Sang Saka Merah Putih, terdiri atas warna merah di atas dan putih di bawah.",
    "Lambang negara Indonesia adalah Garuda Pancasila dengan semboyan Bhinneka Tunggal Ika yang berarti berbeda-beda tetapi tetap satu.",
    "Bahasa resmi dan bahasa persatuan Indonesia adalah Bahasa Indonesia.",
    "Indonesia memiliki tiga zona waktu: Waktu Indonesia Barat (WIB, UTC+7), Waktu Indonesia Tengah (WITA, UTC+8), dan Waktu Indonesia Timur (WIT, UTC+9).",
    "Puncak tertinggi di Indonesia adalah Puncak Jaya (Carstensz Pyramid) di Papua dengan ketinggian sekitar 4.884 meter di atas permukaan laut.",
    "Danau terbesar di Indonesia adalah Danau Toba di Sumatera Utara, yang terbentuk dari letusan supervolkano sekitar 74.000 tahun lalu.",
    "Sungai terpanjang di Indonesia adalah Sungai Kapuas di Kalimantan Barat dengan panjang sekitar 1.143 kilometer.",
    "Candi Borobudur di Magelang, Jawa Tengah, adalah candi Buddha terbesar di dunia, dibangun pada masa Dinasti Syailendra sekitar abad ke-8 hingga ke-9 Masehi.",
    "Komodo (Varanus komodoensis) adalah kadal terbesar di dunia dan hanya hidup secara alami di Indonesia, antara lain di Pulau Komodo dan Pulau Rinca.",
    "Letusan Gunung Krakatau pada 26–27 Agustus 1883 termasuk letusan gunung api paling dahsyat dalam sejarah modern dan suaranya terdengar hingga ribuan kilometer.",
    "Proklamasi Kemerdekaan Indonesia dibacakan oleh Soekarno dan Mohammad Hatta pada tanggal 17 Agustus 1945 di Jakarta.",
    "Indonesia adalah negara kepulauan terbesar di dunia dengan lebih dari 17.000 pulau.",
    "Pancasila terdiri atas lima sila: Ketuhanan Yang Maha Esa; Kemanusiaan yang adil dan beradab; Persatuan Indonesia; Kerakyatan yang dipimpin oleh hikmat kebijaksanaan dalam permusyawaratan/perwakilan; dan Keadilan sosial bagi seluruh rakyat Indonesia.",
    "Mata uang Indonesia adalah Rupiah, dengan kode ISO IDR.",
]

_WORLD_GEOGRAPHY = [
    "Puncak tertinggi di dunia di atas permukaan laut adalah Gunung Everest di Pegunungan Himalaya, dengan ketinggian sekitar 8.849 meter.",
    "Samudra terluas di dunia adalah Samudra Pasifik; lima samudra di Bumi adalah Pasifik, Atlantik, Hindia, Selatan (Antartika), dan Arktik.",
    "Terdapat tujuh benua di Bumi: Asia, Afrika, Amerika Utara, Amerika Selatan, Antartika, Eropa, dan Australia (Oseania).",
    "Asia adalah benua terluas dan berpenduduk terbanyak di dunia.",
    "Negara terluas di dunia berdasarkan luas wilayah adalah Rusia, sekitar 17,1 juta kilometer persegi.",
    "Negara terkecil di dunia berdasarkan luas wilayah adalah Vatikan, sekitar 0,44 kilometer persegi.",
    "Gurun panas terbesar di dunia adalah Gurun Sahara di Afrika Utara dengan luas sekitar 9 juta kilometer persegi.",
    "Titik terdalam di lautan adalah Challenger Deep di Palung Mariana, Samudra Pasifik, dengan kedalaman sekitar 10.900 meter.",
    "Danau terdalam di dunia adalah Danau Baikal di Rusia dengan kedalaman maksimum sekitar 1.640 meter.",
    "Sungai Nil dan Sungai Amazon adalah dua sungai terpanjang di dunia; Nil umumnya disebut terpanjang dengan sekitar 6.650 kilometer, sedangkan Amazon memiliki debit air terbesar.",
    "Air terjun tertinggi di dunia adalah Angel Falls di Venezuela dengan ketinggian sekitar 979 meter.",
    "Antartika adalah benua terdingin, terkering, dan berangin paling kencang di Bumi.",
    "Sekitar 71 persen permukaan Bumi tertutup air, dan sisanya sekitar 29 persen adalah daratan.",
    "Luas permukaan Bumi sekitar 510 juta kilometer persegi.",
    "Garis khatulistiwa (ekuator) adalah garis lintang 0 derajat yang membagi Bumi menjadi belahan utara dan selatan; Indonesia dilintasi oleh garis ini.",
    "Cincin Api Pasifik (Ring of Fire) adalah jalur sepanjang sekitar 40.000 kilometer yang sering mengalami gempa bumi dan letusan gunung api; Indonesia termasuk di dalamnya.",
]

_SCIENCE = [
    "Kecepatan cahaya dalam ruang hampa adalah 299.792.458 meter per detik, atau sekitar 300.000 kilometer per detik.",
    "Air murni mendidih pada suhu 100 derajat Celsius dan membeku pada 0 derajat Celsius pada tekanan atmosfer standar satu atmosfer.",
    "Percepatan gravitasi standar di permukaan Bumi adalah 9,80665 meter per detik kuadrat, sering dibulatkan menjadi 9,8 m/s².",
    "Nilai pi (π) adalah perbandingan keliling lingkaran terhadap diameternya, sekitar 3,14159.",
    "Suhu nol mutlak adalah minus 273,15 derajat Celsius atau 0 Kelvin.",
    "Udara di atmosfer Bumi terdiri atas sekitar 78 persen nitrogen, 21 persen oksigen, dan sisanya argon, karbon dioksida, serta gas lain.",
    "Kecepatan suara di udara pada suhu sekitar 20 derajat Celsius adalah sekitar 343 meter per detik.",
    "Rumus kesetaraan massa-energi Albert Einstein adalah E = mc², dipublikasikan pada tahun 1905.",
    "Isaac Newton merumuskan tiga hukum gerak dan hukum gravitasi universal, dipublikasikan dalam Principia pada tahun 1687.",
    "Atom terdiri atas inti (proton dan neutron) yang dikelilingi elektron.",
    "Tabel periodik unsur kimia pertama kali disusun secara sistematis oleh Dmitri Mendeleev pada tahun 1869.",
    "Saat ini diketahui 118 unsur kimia dalam tabel periodik.",
    "Hidrogen adalah unsur teringan dan paling melimpah di alam semesta.",
    "Simbol kimia emas adalah Au, perak Ag, besi Fe, natrium Na, kalium K, tembaga Cu, dan oksigen O.",
    "Rumus kimia air adalah H2O, yaitu dua atom hidrogen dan satu atom oksigen.",
    "Fotosintesis adalah proses tumbuhan mengubah karbon dioksida dan air menjadi glukosa dan oksigen dengan bantuan energi cahaya.",
    "Struktur heliks ganda DNA dipublikasikan oleh James Watson dan Francis Crick pada tahun 1953, dengan dukungan data difraksi sinar-X dari Rosalind Franklin.",
    "Bilangan Avogadro adalah sekitar 6,022 × 10²³ partikel per mol.",
    "Konstanta Planck bernilai sekitar 6,626 × 10⁻³⁴ joule detik.",
    "Pertumbuhan dan kehidupan seluruh organisme bergantung pada sel, yang merupakan unit dasar struktur dan fungsi makhluk hidup.",
]

_HUMAN_BODY = [
    "Tubuh manusia dewasa memiliki 206 tulang.",
    "Jantung manusia memiliki empat ruang: dua serambi (atrium) dan dua bilik (ventrikel).",
    "Otak manusia mengandung sekitar 86 miliar neuron.",
    "Suhu tubuh normal manusia sekitar 36,5 hingga 37,5 derajat Celsius.",
    "Golongan darah utama pada manusia dalam sistem ABO adalah A, B, AB, dan O.",
    "Organ terbesar tubuh manusia adalah kulit.",
    "Tulang terpanjang dan terkuat dalam tubuh manusia adalah tulang paha (femur).",
    "Manusia dewasa memiliki 32 gigi permanen, termasuk empat gigi geraham bungsu.",
    "Penisilin ditemukan oleh Alexander Fleming pada tahun 1928.",
    "Vitamin C (asam askorbat) diperlukan tubuh untuk pembentukan kolagen, dan kekurangannya menyebabkan penyakit skorbut.",
]

_SPACE = [
    "Tata Surya memiliki delapan planet menurut urutan jarak dari Matahari: Merkurius, Venus, Bumi, Mars, Jupiter, Saturnus, Uranus, dan Neptunus.",
    "Matahari adalah bintang di pusat Tata Surya dan mengandung sekitar 99,86 persen dari seluruh massa Tata Surya.",
    "Usia Tata Surya diperkirakan sekitar 4,6 miliar tahun.",
    "Jupiter adalah planet terbesar di Tata Surya dengan diameter sekitar 139.800 kilometer.",
    "Merkurius adalah planet terkecil dan terdekat dengan Matahari, dengan diameter sekitar 4.879 kilometer.",
    "Venus adalah planet terpanas di Tata Surya dengan suhu permukaan sekitar 465 derajat Celsius akibat efek rumah kaca atmosfer karbon dioksida yang pekat.",
    "Venus berotasi sangat lambat dan arah rotasinya terbalik; satu hari Venus (sekitar 243 hari Bumi) lebih panjang daripada satu tahunnya (sekitar 225 hari Bumi).",
    "Mars dijuluki Planet Merah karena permukaannya kaya besi oksida, dan memiliki dua bulan kecil bernama Phobos dan Deimos.",
    "Olympus Mons di Mars adalah gunung tertinggi yang diketahui di Tata Surya, dengan tinggi sekitar 22 kilometer.",
    "Saturnus terkenal karena sistem cincinnya yang terdiri atas es dan batuan.",
    "Uranus berotasi dengan sumbu yang miring sekitar 98 derajat sehingga tampak berputar menggelinding pada sisinya.",
    "Neptunus adalah planet terjauh dari Matahari di antara delapan planet Tata Surya.",
    "Pluto diklasifikasikan ulang sebagai planet kerdil oleh Persatuan Astronomi Internasional (IAU) pada tahun 2006.",
    "Satu satuan astronomi (AU) adalah jarak rata-rata Bumi ke Matahari, sekitar 149,6 juta kilometer.",
    "Jarak rata-rata Bumi ke Bulan sekitar 384.400 kilometer.",
    "Cahaya dari Matahari membutuhkan waktu sekitar 8 menit 20 detik untuk mencapai Bumi.",
    "Bulan adalah satu-satunya satelit alami Bumi dengan diameter sekitar 3.474 kilometer dan gravitasi permukaan sekitar seperenam gravitasi Bumi.",
    "Bumi berotasi satu kali dalam sekitar 24 jam dan mengelilingi Matahari satu kali dalam sekitar 365,25 hari; sumbunya miring sekitar 23,5 derajat.",
    "Suhu permukaan Matahari sekitar 5.500 derajat Celsius, sedangkan suhu intinya sekitar 15 juta derajat Celsius.",
    "Ganimede, bulan Jupiter, adalah bulan terbesar di Tata Surya, bahkan lebih besar daripada planet Merkurius.",
    "Titan, bulan Saturnus, adalah satu-satunya bulan di Tata Surya yang memiliki atmosfer tebal.",
    "Satu tahun cahaya adalah jarak yang ditempuh cahaya dalam satu tahun, sekitar 9,46 triliun kilometer.",
    "Proxima Centauri adalah bintang terdekat dari Matahari, sekitar 4,24 tahun cahaya jauhnya.",
    "Sirius adalah bintang paling terang yang tampak di langit malam dari Bumi.",
    "Galaksi tempat Tata Surya berada disebut Galaksi Bima Sakti (Milky Way), yang berisi ratusan miliar bintang.",
    "Galaksi besar terdekat dari Bima Sakti adalah Galaksi Andromeda, sekitar 2,5 juta tahun cahaya jauhnya.",
    "Alam semesta diperkirakan berusia sekitar 13,8 miliar tahun sejak peristiwa Big Bang.",
    "Lubang hitam adalah wilayah ruang-waktu dengan gravitasi sangat kuat sehingga bahkan cahaya tidak dapat lolos dari dalamnya.",
    "Foto lubang hitam pertama, yaitu M87*, dirilis oleh Event Horizon Telescope pada tanggal 10 April 2019.",
    "Komet Halley kembali mendekati Matahari setiap sekitar 76 tahun; kemunculan terakhirnya pada tahun 1986 dan berikutnya diperkirakan pada tahun 2061.",
    "Sputnik 1, satelit buatan pertama di dunia, diluncurkan oleh Uni Soviet pada tanggal 4 Oktober 1957.",
    "Yuri Gagarin dari Uni Soviet adalah manusia pertama yang terbang ke luar angkasa dan mengorbit Bumi pada tanggal 12 April 1961 dengan wahana Vostok 1.",
    "Valentina Tereshkova dari Uni Soviet adalah wanita pertama di luar angkasa pada tahun 1963.",
    "Neil Armstrong adalah manusia pertama yang berjalan di Bulan pada tanggal 20 Juli 1969 dalam misi Apollo 11, diikuti oleh Buzz Aldrin, sementara Michael Collins tetap berada di modul komando.",
    "Program Apollo membawa dua belas astronaut berjalan di Bulan antara tahun 1969 dan 1972.",
    "Stasiun Luar Angkasa Internasional (ISS) mengorbit Bumi pada ketinggian sekitar 400 kilometer; perakitannya dimulai pada tahun 1998.",
    "Teleskop Luar Angkasa Hubble diluncurkan pada tahun 1990, dan Teleskop Luar Angkasa James Webb (JWST) diluncurkan pada tanggal 25 Desember 2021.",
    "Voyager 1 diluncurkan oleh NASA pada tahun 1977 dan menjadi wahana buatan manusia pertama yang memasuki ruang antarbintang pada tahun 2012.",
    "Wahana New Horizons milik NASA melintas dekat Pluto pada tanggal 14 Juli 2015.",
    "Rover Curiosity mendarat di Mars pada Agustus 2012, dan rover Perseverance mendarat di Kawah Jezero, Mars, pada tanggal 18 Februari 2021.",
    "Chandrayaan-3 milik India mendarat di dekat kutub selatan Bulan pada tanggal 23 Agustus 2023.",
    "Gerhana matahari terjadi ketika Bulan berada di antara Matahari dan Bumi, sedangkan gerhana bulan terjadi ketika Bumi berada di antara Matahari dan Bulan.",
    "Pasang surut air laut di Bumi terutama disebabkan oleh gaya gravitasi Bulan dan, dalam tingkat lebih kecil, Matahari.",
    "Sabuk asteroid utama terletak di antara orbit planet Mars dan Jupiter, dengan Ceres sebagai objek terbesar sekaligus planet kerdil di sabuk tersebut.",
    "Sabuk Kuiper adalah wilayah dingin berisi benda-benda es di luar orbit Neptunus, tempat planet kerdil Pluto, Eris, Haumea, dan Makemake berada.",
    "Bintang neutron adalah inti bintang masif yang sangat padat hasil ledakan supernova; pulsar adalah bintang neutron yang berotasi cepat dan memancarkan gelombang radio secara berkala.",
    "Sinar kosmik dan angin matahari yang berinteraksi dengan medan magnet serta atmosfer atas Bumi menghasilkan fenomena aurora borealis di utara dan aurora australis di selatan.",
    "Tata Surya mengorbit pusat Galaksi Bima Sakti dengan periode satu tahun galaktik sekitar 225 hingga 250 juta tahun.",
    "Exoplanet (eksoplanet) adalah planet yang mengorbit bintang di luar Tata Surya; ribuan eksoplanet telah ditemukan, antara lain oleh teleskop Kepler dan TESS.",
]

_HISTORY = [
    "Perang Dunia I berlangsung dari tahun 1914 hingga 1918, dan Perang Dunia II berlangsung dari tahun 1939 hingga 1945.",
    "Revolusi Prancis dimulai pada tahun 1789.",
    "Deklarasi Kemerdekaan Amerika Serikat ditandatangani pada tanggal 4 Juli 1776.",
    "Tembok Berlin runtuh pada tanggal 9 November 1989.",
    "Uni Soviet bubar pada bulan Desember 1991.",
    "Perserikatan Bangsa-Bangsa (PBB) didirikan pada tanggal 24 Oktober 1945.",
    "Christopher Columbus tiba di benua Amerika pada tahun 1492.",
    "Ekspedisi Ferdinand Magellan yang diselesaikan oleh Juan Sebastián Elcano pada tahun 1522 adalah pelayaran keliling dunia pertama.",
    "Konstantinopel jatuh ke tangan Kesultanan Utsmaniyah pada tahun 1453.",
    "Piramida Agung Giza dibangun untuk Firaun Khufu sekitar tahun 2560 SM.",
    "Revolusi Industri dimulai di Inggris pada abad ke-18.",
    "Olimpiade modern pertama diselenggarakan di Athena, Yunani, pada tahun 1896.",
    "Piala Dunia FIFA pertama diselenggarakan di Uruguay pada tahun 1930.",
    "Bom atom dijatuhkan di Hiroshima pada tanggal 6 Agustus 1945 dan di Nagasaki pada tanggal 9 Agustus 1945.",
    "Konferensi Asia-Afrika diselenggarakan di Bandung pada tanggal 18–24 April 1955.",
]

_TECH = [
    "Satu byte terdiri atas 8 bit.",
    "Bahasa pemrograman Python dibuat oleh Guido van Rossum dan pertama kali dirilis pada tahun 1991.",
    "Kernel Linux pertama kali dirilis oleh Linus Torvalds pada tahun 1991.",
    "World Wide Web diusulkan oleh Tim Berners-Lee pada tahun 1989 di CERN.",
    "ARPANET, pendahulu Internet, melakukan pengiriman pesan pertamanya pada tahun 1969.",
    "Bitcoin diperkenalkan oleh sosok bernama samaran Satoshi Nakamoto pada tahun 2008, dan jaringannya beroperasi sejak Januari 2009.",
    "Alan Turing mengusulkan konsep mesin Turing pada tahun 1936 dan menggagas uji Turing pada tahun 1950.",
    "Telepon dipatenkan oleh Alexander Graham Bell pada tahun 1876, dan Wright bersaudara menerbangkan pesawat bermesin pertama pada tahun 1903.",
]


def get_world_knowledge_corpus() -> List[str]:
    """Return curated world & space facts as self-contained sentences."""
    docs: List[str] = []
    for country, capital in _CAPITALS:
        docs.append(f"Ibu kota {country} adalah {capital}.")
    for country, cur in _CURRENCIES:
        docs.append(f"Mata uang {country} adalah {cur}.")
    for block in (_INDONESIA, _WORLD_GEOGRAPHY, _SCIENCE, _HUMAN_BODY, _SPACE, _HISTORY, _TECH):
        docs.extend(block)
    seen = set()
    unique: List[str] = []
    for d in docs:
        if d not in seen:
            seen.add(d)
            unique.append(d)
    return unique
