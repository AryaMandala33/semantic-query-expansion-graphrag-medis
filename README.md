# Semantic Query Expansion Berbasis LLM untuk Disambiguasi Leksikal Istilah Medis pada GraphRAG

Kode, set uji, dan hasil evaluasi untuk artikel:

> Putra, A. M. dan Munarko, Y., *Semantic Query Expansion Berbasis LLM untuk Disambiguasi Leksikal Istilah Medis pada GraphRAG*, JATISI (dalam proses pengajuan).

Sistem memperluas kueri medis yang ambigu dengan GPT-4o mini, memetakan kandidat interpretasi ke *node* subset PrimeKG di Neo4j, lalu memilih interpretasi dengan *Disambiguation Ranker* (all-mpnet-base-v2 + *cosine similarity*). Kueri *specificity gap* ditangani melalui klarifikasi.

## Isi repositori

| Path | Isi |
|---|---|
| `data/testset_30_kueri.csv` | **Set uji 30 kueri** dalam format tabel: kategori, istilah ambigu, kueri, konsep acuan, relasi relevan, alasan klinis (Inggris dan Indonesia), rujukan pustaka medis, tautan rujukan daring, dan status validasi. |
| `data/ambiguity_testset_draft.json` | Set uji yang sama dalam format JSON. Ini input untuk `build_testset.py`. |
| `data/kamus/` | Kamus akronim (Medical Abbreviation and Acronym Meta-Inventory), 50 istilah polisemi (NLM WSD Test Collection), aturan *specificity gap*, dan daftar kata henti. Sumber tiap file dijelaskan di `data/kamus/README.md`. |
| `results/final/details.json` | Hasil evaluasi akhir per kueri untuk sistem yang diusulkan dan *baseline* kata kunci. |
| `results/baseline_embedding/` | Hasil *baseline* pencocokan *embedding* (`summary.txt` berisi ringkasan dan uji signifikansi). |
| `core/` | Sistem yang dievaluasi: Query Service, LLM Expander, GraphRAG Retriever, Disambiguation Ranker, pipeline dan klarifikasi, *baseline*, serta metrik. |
| `scripts/` | Pemfilteran `kg.csv` PrimeKG ke enam relasi dan impor ke Neo4j. |
| `build_testset.py` | Memetakan label acuan ke indeks *node* PrimeKG. |
| `evaluate.py`, `analyze.py` | Evaluasi dan uji statistik (*paired t-test*, *randomization*, McNemar eksak). |
| `baseline_embedding.py` | *Baseline* tambahan berbasis pencocokan *embedding* kueri ke nama *node* (lihat `BASELINE_EMBEDDING.md`). |
| `KEPUTUSAN_IMPLEMENTASI.md` | Keputusan teknis yang ditetapkan sebelum evaluasi akhir. |
| `CATATAN_PERUBAHAN_EVALUASI.md` | Riwayat perubahan evaluasi, termasuk uji pilot dan pelengkapan label. |

## Tentang label acuan

Label acuan bersifat **berbasis rujukan**. Kebenaran konsep acuan setiap kueri bersumber dari pustaka medis yang dirujuk pada kueri tersebut, yaitu Harrison's Principles of Internal Medicine edisi ke-20 (29 kueri) dan Nelson Textbook of Pediatrics edisi ke-21 (1 kueri). Setiap kueri juga dilengkapi artikel StatPearls (NCBI Bookshelf) yang dapat diakses daring. Validasi dilakukan oleh penulis kedua (dosen Informatika, bukan klinisi), dengan memeriksa kesesuaian label dan alasan klinis terhadap rujukan tersebut. Pembaca dapat memeriksa setiap label melalui kolom `rationale`, `source`, dan `source_online_url` di `data/testset_30_kueri.csv`.

## Menjalankan

Prasyarat: Python 3.10, Neo4j 5, dan `kg.csv` PrimeKG (unduh dari Harvard Dataverse, doi:10.7910/DVN/IXA7BM). File PrimeKG tidak disertakan di repositori ini.

```bash
pip install -r requirements.txt
cp .env.example .env        # isi LLM_API_KEY dan kredensial Neo4j
python -m tests.test_offline

python scripts/filter_kg.py
python scripts/import_neo4j_bab3.py
python build_testset.py
python evaluate.py --final
python analyze.py --dir reports/utama/final
python baseline_embedding.py
```

Respons LLM disimpan di `data/cache/llm_utama/` sehingga evaluasi dapat diulang dengan keluaran yang sama. Folder cache tidak disertakan.

## Catatan

- File `.env` berisi API key dan tidak boleh diunggah.
- Komponen versi pengembangan tidak disertakan karena tidak dipakai pada evaluasi akhir.
