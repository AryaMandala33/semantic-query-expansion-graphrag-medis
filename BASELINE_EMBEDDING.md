# Baseline tambahan: pencocokan embedding (ditetapkan 23 Nov 2026, sebelum dijalankan)

Alasan: saran dosen pembimbing pada revisi artikel. Baseline kata kunci (DA 0,033) dinilai
terlalu lemah sebagai pembanding, sehingga ditambahkan baseline yang memakai kemiripan semantik
tanpa query expansion.

| Hal | Keputusan |
|---|---|
| Node awal | 5 node dengan cosine similarity tertinggi antara kueri asli dan teks node "nama (tipe)". Seluruh node subset (Disease, Drug, Phenotype, Protein), kecuali nama "obsolete ...". |
| Model | all-mpnet-base-v2, sama dengan Disambiguation Ranker. |
| Retrieval dan pemeringkatan | Sama persis dengan baseline kata kunci: tetangga 1-hop (30 per relasi), Disambiguation Ranker identik, interpretasi = node peringkat pertama. |
| Pembanding | Hasil evaluasi akhir yang sudah ada (reports/utama/final/items.csv). Evaluasi akhir TIDAK dijalankan ulang. |
| Uji | Paired t-test dan randomization (P@5, RR), McNemar eksak (DA dengan dan tanpa simulasi), n = 30, alpha = 0,05. |
| Penyetelan | Tidak ada. Skrip dijalankan satu kali dan hasilnya dilaporkan apa adanya. |
