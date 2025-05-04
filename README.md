# Eksperimen_SML_Rakha

**Deskripsi**  
Repo ini berisi eksperimen manual dan pipeline otomatis untuk praproses data (Kriteria 1 submission Dicoding Machine Learning).

## Struktur Repository
* Eksperimen_SML_Rakha/
    * github/ # (opsional) workflow CI advance
    * namadataset_raw/ # folder data mentah (raw)
        * demand_history.csv
    * preprocessing/
        * Eksperimen_Rakha.ipynb # notebook eksplorasi & EDA manual
        * automate_Rakha.py # script otomatisasi preprocessing
        * namadataset_preprocessing/ # output cleaned data
            * clean.csv
            * clean.parquet
    * README.md