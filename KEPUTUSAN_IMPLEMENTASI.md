# Keputusan Implementasi Sistem Utama

Dokumen ini mencatat keputusan teknis yang tidak dirinci di Bab III, beserta alasannya. Semua keputusan ditetapkan pada 29 Oktober 2026, **sebelum** sistem utama dijalankan pada set uji. Setelah evaluasi akhir dimulai, isi dokumen ini tidak diubah. Perubahan apa pun dicatat sebagai versi baru dan tidak dipakai untuk menyetel hasil.

## Data (Bab III 3.4)

| Hal | Keputusan | Alasan |
|---|---|---|
| Database | Database baru `primekgbab3`. `primekgv2` dan `database-clean` tidak disentuh. | Komponen tambahan tetap bisa dijalankan, dan data lama aman. |
| Duplikasi | PrimeKG menyimpan setiap hubungan dua kali (x→y dan y→x). Yang diimpor hanya arah kanonik: obat → kondisi, penyakit → fenotipe/protein. | Sesuai 3.4.2: jumlah baris file tidak sama dengan jumlah relasi unik karena impor menangani duplikasi. |
| Kunci node | `node_index` PrimeKG dijadikan kunci unik. Indeks dibuat pada `id`, `name`, dan `name_lower`. | `id` PrimeKG tidak unik lintas sumber. `name_lower` dipakai untuk pencocokan tanpa membedakan huruf besar/kecil. |

## Query Service (3.3.1)

| Hal | Keputusan | Alasan |
|---|---|---|
| Kamus akronim | Medical Abbreviation and Acronym Meta-Inventory v1.0.0. *Short form* yang punya ≥ 2 kelompok makna (GroupID) dianggap ambigu, sehingga ada 21.363 entri. | Sumber publik yang disebut di Bab III [16], dan tidak disusun dari set uji. |
| Bentuk akronim | Token harus ditulis huruf kapital pada kueri asli, panjang 2–6 karakter. | Tanpa syarat ini, kata biasa seperti "is", "as", dan "at" ikut terdeteksi, karena kata-kata itu juga tercatat sebagai singkatan di Meta-Inventory. |
| Daftar polisemi | 50 istilah NLM WSD Test Collection (Weeber dkk., 2001), diambil apa adanya. | Bersumber dari literatur dan tidak ditambah berdasarkan set uji. Cakupannya terbatas dan dicatat sebagai keterbatasan. |
| Aturan specificity gap | Kata umum (sakit/penyakit/gangguan; disease/disorder/problem) yang bersebelahan dengan nama organ atau sistem tubuh. | Mengikuti contoh di Bab III. Daftar organnya diambil dari kosakata anatomi umum. |
| Urutan aturan | Akronim → polisemi → specificity gap. Aturan pertama yang terpenuhi menentukan label. | Mengikuti urutan penulisan di Bab III. |
| Pemakaian label | Label dan istilah pemicu diberikan kepada LLM sebagai petunjuk. Tipe akhirnya diputuskan LLM. | Aturan bisa keliru, misalnya "ECG" terdeteksi sebagai akronim pada kueri polisemi. Akurasi deteksi aturan dan LLM dilaporkan terpisah. |

## LLM Expander (3.3.2)

| Hal | Keputusan |
|---|---|
| Model | `gpt-4o-mini`, temperature 0, versi prompt `utama-v1 (2026-10-29)`. Mistral-7B bisa dipakai lewat endpoint kompatibel OpenAI (Ollama) dengan mengubah `.env`, tetapi tidak dipakai di eksperimen. |
| Few-shot | Tiga contoh: tension, CP, penyakit kulit. Ketiganya bukan istilah di set uji maupun set dev. |
| Skema | Setiap kandidat berisi `name`, `entity_type` (disease, phenotype, drug effect, drug), `confidence`, dan `reason`, maksimal 5 kandidat. Kalau JSON atau skemanya tidak valid, permintaan diulang satu kali dengan menyebutkan kesalahannya. |
| Cache | Respons yang valid disimpan di `data/cache/llm_utama/` agar hasilnya bisa direproduksi. |

## Retriever (3.3.4)

| Hal | Keputusan | Alasan |
|---|---|---|
| Pemetaan nama | Pencocokan persis tanpa membedakan huruf besar/kecil. Ada dua normalisasi: bentuk posesif dihapus ("Alzheimer's" → "alzheimer"), dan varian berakhiran " (disease)" ikut dicoba. | Mengikuti konvensi penamaan MONDO/HPO, tanpa pencocokan fuzzy. |
| Jenis konsep → label | disease → Disease, phenotype → Phenotype, drug effect → Phenotype, drug → Drug. Tidak ada fallback ke label lain. Kalau nama ada tetapi labelnya berbeda, kandidat dicatat sebagai `type_mismatch`. | 3.3.4 dan 3.4.3: entitas yang namanya mirip tetapi tipenya berbeda tidak boleh dianggap konsep yang sama. |
| Batas tetangga | 30 per relasi (maksimal 180 per node), sama untuk sistem utama dan baseline. Dalam satu relasi, tetangga diurutkan dari degree terkecil (paling spesifik). | Membatasi beban komputasi tanpa membuang relasi mana pun. |

## Disambiguation Ranker (3.3.5)

| Hal | Keputusan |
|---|---|
| Teks sub-graf | `nama (tipe), relasi: tetangga1, tetangga2, ...; relasi: ...`. Paling banyak 8 nama per relasi. Urutan relasinya: fenotipe, efek obat, indikasi, kontraindikasi, off-label use, protein. |
| Teks node | `nama (tipe), relasi: konsep terpilih`. Konsep terpilih sendiri ditulis `nama (tipe)` dan ikut diurutkan. |
| Kemiripan | Embedding all-mpnet-base-v2 yang dinormalisasi, lalu dihitung cosine similarity. Bila skornya sama, node_index terkecil didahulukan. |

## Baseline (3.5)

| Hal | Keputusan |
|---|---|
| Kata kunci | Token kueri dikurangi kata henti bahasa Indonesia dan Inggris. Token yang seluruhnya angka dibuang. |
| Pencocokan | Nama node harus memuat kata kunci sebagai kata utuh. Paling banyak 5 node diambil (setara batas 5 kandidat), diurutkan berdasarkan jumlah kata kunci yang dimuat, lalu nama terpendek. |

## Evaluasi (3.6–3.7)

| Hal | Keputusan |
|---|---|
| Node relevan | Konsep acuan beserta tetangga 1-hop melalui relasi relevan per kueri (kolom `relevant_relations`, ikut divalidasi dosen). Aturannya: kueri yang menanyakan obat memakai `indication`; kueri tentang efek obat memakai `drug_effect`; kueri yang mendeskripsikan gejala untuk mengenali kondisi memakai `disease_phenotype_positive` dan `indication`, mengikuti contoh specificity gap di Tabel 3.2. Aturan ini ditetapkan setelah pemeriksaan data menunjukkan beberapa konsep acuan (mis. major depressive disorder, COPD, breast cancer) tidak memiliki relasi penyakit–fenotipe di subset, dan diterapkan seragam ke semua kueri sebelum sistem dijalankan pada set uji. |
| Konsep setara | Node bernama identik pada ontologi lain (misalnya MONDO "gout" dan HPO "Gout") diterima sebagai konsep acuan dan diberi tanda `auto_equivalent`. |
| Simulasi klarifikasi | Berlaku untuk kueri berkategori specificity gap menurut label acuan. Kandidat terpetakan pertama (urutan keluaran LLM) yang termasuk konsep acuan dipilih. Kalau tidak ada, DA dinyatakan salah dan retrieval memakai pilihan ranker. DA tanpa simulasi tetap dilaporkan sebagai pembanding. |
| Uji statistik | Paired t-test dan uji randomization berpasangan (100.000 permutasi, seed 42) untuk P@5 dan RR; McNemar exact untuk DA; α = 0,05; dihitung pada seluruh kueri. |
| Evaluasi akhir | `python evaluate.py --final` hanya bisa dijalankan bila seluruh kueri sudah `verified: true`, dan dijalankan satu kali. |

## Survei kepuasan pengguna (3.6.4) — pelaksanaan daring

| Hal | Keputusan | Alasan |
|---|---|---|
| Media | Google Form. Keluaran kedua sistem untuk enam kueri survei dihasilkan sekali oleh `export_survey.py` di laptop peneliti, lalu ditampilkan di formulir. | Semua responden berada di lokasi berjauhan. Sistem deterministik (temperature 0, cache), sehingga keluaran yang ditampilkan sama dengan yang akan diperoleh responden bila menjalankan sistem sendiri. |
| Buta dan acak | Posisi Hasil A/B diacak per kueri (seed 2026). Kuncinya disimpan terpisah dan tidak ditunjukkan ke responden. | Sesuai 3.6.4: responden tidak mengetahui sistem mana yang digunakan. |
| Klarifikasi | Untuk kueri specificity gap, pilihan klarifikasi pada sistem utama disimulasikan sesuai maksud yang ditampilkan kepada responden. | Formulir statis tidak dapat menampilkan dialog interaktif. Hal ini dilaporkan sebagai keterbatasan. |
| Penyaringan | Responden yang mengaku tenaga kesehatan tidak dihitung. | 3.6.4: responden non-tenaga medis. |
| Tampilan | Label relasi PrimeKG diterjemahkan ke bahasa Indonesia (mis. indication → "obat yang digunakan untuk kondisi ini"); nama node tetap seperti di PrimeKG. | Responden awam. Terjemahan hanya mengubah tampilan, diterapkan sama pada kedua sistem, dan tidak mengubah isi maupun urutan hasil. |
| Kemudahan bagi awam | Setiap kueri disertai situasi sehari-hari, nama konsep yang dipilih sistem diberi padanan bahasa Indonesia (glosarium tetap untuk semua nama yang muncul), formulir diawali halaman "Sebelum mulai" berisi pedoman nilai 1–5 (5 = tepat sesuai yang dicari … 1 = salah atau tidak ada hasil) dan satu contoh penilaian dengan topik di luar kueri survei. Label skala tetap "Sangat tidak puas"–"Sangat puas" sesuai 3.6.4. | Responden non-tenaga medis kesulitan memahami istilah medis berbahasa Inggris. Semua perubahan hanya pada penyajian, diterapkan sama pada kedua sistem, dan pertanyaan inti tetap "Apakah hasil menjawab maksud Anda?" sesuai 3.6.4. |
| Kueri survei | Enam kueri survei tidak diganti setelah keluaran sistem dilihat, termasuk U3 yang tidak menghasilkan makna pada kedua sistem. | Mengganti kueri berdasarkan keluaran akan menyebabkan bias seleksi pada User Satisfaction Score. |

| Revisi tata bahasa formulir (2 Nov 2026) | Penulisan dirapikan (huruf kapital, padanan Indonesia lebih dulu, hasil kosong ditulis "Tidak ada hasil yang ditemukan", pedoman nilai cukup sekali). Isi hasil kedua sistem dan pengacakan A/B tidak berubah. Satu jawaban dari formulir versi sebelumnya dihapus dan respondennya diminta mengisi ulang, sehingga semua responden melihat versi yang sama. | Konsistensi penyajian antarresponden. |
