# Catatan Perubahan Evaluasi

Dokumen ini mencatat setiap perubahan pada sistem dan set uji **setelah** set uji
pertama kali dijalankan. Tujuannya transparansi: penguji dapat melihat apa yang
diubah, kapan, dan mengapa.

## Uji Pilot (v1) — 29 Oktober 2026

- Set uji: 30 kueri (`ambiguity_testset_mapped.json`), **belum diverifikasi dosen**.
- Konfigurasi: gpt-4o-mini, few-shot, tanpa petunjuk tipe, bobot ranker
  `llm=0.40, graph=0.35, emb=0.15, hier=0.10` (ditetapkan sebelum uji).
- Hasil (Acc@1): baseline 0.033, llm_only 0.767, llm_emb 0.767,
  llm_graph 0.700, full 0.733.
- Laporan tersimpan di `reports/pilot_v1/`.

### Analisis kesalahan pada sistem `full` (7 kueri)

| Item | Sumber kesalahan | Keterangan |
|---|---|---|
| P10, S04, S06 | Anotasi belum lengkap | Sistem memilih node HPO yang konsepnya sama dengan jawaban acuan MONDO |
| A05 | Entity linking | Bentuk posesif "Alzheimer's" tidak cocok dengan "Alzheimer disease" |
| A04, P03, S05 | Penalaran LLM | Makna yang dipilih/kekhususan kandidat keliru |
| A02 | Ranker (bukti graf) | Bukti graf membalikkan jawaban LLM yang benar |

Temuan tambahan: pada 14 dari 24 kueri yang memiliki bukti graf, kandidat dengan
bukti graf terkuat adalah kandidat yang **salah**, umumnya node generik berderajat
tinggi (Arrhythmia, Headache, Shock, anemia, Cirrhosis).

## Perubahan v1 → v2

| # | Komponen | Perubahan | Jenis | Alasan |
|---|---|---|---|---|
| 1 | `retriever.py` | Bentuk posesif dibuang sebelum linking ("'s") | Perbaikan bug | Konvensi penamaan MONDO/HPO tidak memakai posesif; aturan berlaku umum |
| 2 | `build_testset.py` | Node dengan nama identik lintas ontologi otomatis ditambahkan sebagai jawaban acuan (`auto_equivalent: true`) | Kelengkapan anotasi | Konsep yang sama di MONDO dan HPO; aturan berlaku seragam untuk semua item |
| 3 | Set uji, P10 | "Peptic ulcer" (HPO) ditambahkan manual (`added_after_pilot: true`) | Kelengkapan anotasi | Sinonim; menunggu keputusan dosen |

**Yang TIDAK diubah:** bobot ranker, rumus skor, prompt expander, contoh few-shot.

## Keputusan tentang komponen graf (29 Oktober 2026)

Bias bukti graf terhadap node generik **tidak diperbaiki** pada skripsi ini.
Alasannya: perbaikan yang dirancang setelah melihat hasil set uji berisiko
menyetel sistem pada data uji. Bobot dan rumus ranker dibiarkan seperti
ditetapkan sebelum uji pilot. Temuan ini dilaporkan sebagai hasil analisis di
Bab 4. Normalisasi terhadap degree kandidat (mis. indeks Resource Allocation atau
Salton) yang divalidasi pada set dev terpisah diajukan sebagai saran penelitian
lanjutan di Bab 5.

**Sistem dibekukan sejak tanggal ini.** Tidak ada perubahan lagi pada bobot,
rumus skor, prompt, maupun contoh few-shot.

## Rencana

- Set uji diverifikasi dosen → evaluasi akhir dijalankan **sekali** pada versi
  tervalidasi.
- Hasil pilot (v1) dilaporkan di lampiran bersama tabel perubahan ini.

## 29 Oktober 2026 — Penyelarasan sistem dengan Bab I–III

Setelah Bab I–III disetujui dosen pembimbing, sistem utama dibangun ulang mengikuti rancangan Bab III (folder `core/`):
subset 6 relasi dengan tipe relasi `RELATION`, Query Service berbasis aturan (FastAPI), LLM Expander, Retriever melalui LangChain,
Disambiguation Ranker berbasis cosine similarity, baseline kata kunci, metrik P@5/MRR/DA dengan simulasi klarifikasi,
paired t-test, uji randomization, McNemar, dan survei kepuasan pengguna buta (blind).

Perubahan ini mengikuti dokumen yang disetujui, bukan hasil uji, sehingga tidak termasuk penyetelan pada set uji.
Keputusan teknis yang tidak dirinci di Bab III dicatat di `KEPUTUSAN_IMPLEMENTASI.md`.

Sistem versi pengembangan (v2: bukti graf, validasi hierarki, ranker berbobot) tetap disimpan di `modules/` dan `tambahan/`
sebagai komponen tambahan. Komponen ini tidak dipanggil oleh sistem utama dan hasilnya hanya dilaporkan sebagai analisis tambahan.
Hasil uji pilot v1/v2 dicatat sebagai riwayat pengembangan, bukan hasil utama.

## 30 Oktober 2026 — Validasi set uji oleh dosen pembimbing

Set uji (30 kueri) divalidasi oleh Yuda Munarko, S.Kom., M.Sc. Hasil: 30 dari 30 kueri "Sesuai", kesimpulan "Layak digunakan".
Seluruh kueri ditandai `verified: true`; set uji tidak diubah. Evaluasi akhir dijalankan satu kali dengan `python evaluate.py --final`.

## 31 Oktober 2026 — Evaluasi akhir

**Run pertama (07:28) gagal secara teknis dan tidak sah.** API key LLM ditolak (AuthenticationError 401) sejak kueri A04,
sehingga hanya A01–A03 yang memperoleh keluaran LLM (dari cache). Sebanyak 27 kueri tidak pernah diproses LLM Expander,
jadi angka yang tercetak bukan keluaran sistem dan tidak dipakai. Berkas hasil run ini tertimpa oleh run berikutnya.

**Run kedua (07:34) adalah hasil evaluasi akhir yang sah.** Dijalankan setelah API key diganti, tanpa perubahan apa pun
pada sistem, prompt, parameter, maupun set uji. Seluruh 30 kueri diproses tanpa galat LLM. Hasil: `reports/utama/final/`.

Setelah itu, skrip evaluasi (bukan sistem) ditambah pemeriksaan akses LLM sebelum evaluasi dimulai dan penghentian
otomatis tanpa menyimpan hasil bila terjadi galat API/koneksi, untuk mencegah kejadian serupa pada pemakaian berikutnya.
