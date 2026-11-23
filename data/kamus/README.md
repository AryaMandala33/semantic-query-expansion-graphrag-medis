# Kamus Query Service (Bab III 3.3.1)

| File | Isi | Sumber |
|---|---|---|
| akronim_meta_inventory.json | 21.363 akronim medis ambigu (≥ 2 kelompok makna) | Grossman Liu L, dkk. (2021). A deep database of medical abbreviations and acronyms for natural language processing. *Scientific Data*, 8:149. Meta-Inventory v1.0.0, github.com/lisavirginia/clinical-abbreviations |
| polisemi_nlm_wsd.json | 50 istilah ambigu | Weeber M, Mork JG, Aronson AR (2001). Developing a test collection for biomedical word sense disambiguation. *Proc AMIA Symp*, 746–750. Daftar istilah sesuai Jimeno-Yepes & Aronson (2010), *BMC Bioinformatics* 11:569, Tabel 4 |
| specificity_gap_rules.json | kata umum + nama organ/sistem tubuh | disusun dari kosakata anatomi umum (bukan dari set uji) |
| stopwords_en.txt, stopwords_id.txt | kata henti untuk baseline kata kunci | daftar kata fungsi umum, tidak diambil dari satu sumber baku. 234 dari 239 kata bahasa Indonesia terdapat pada daftar Tala (2003), dan 162 dari 177 kata bahasa Inggris terdapat pada daftar kata henti NLTK |

Kamus-kamus ini tidak disusun dari kueri evaluasi, sehingga tidak terjadi kebocoran data.
