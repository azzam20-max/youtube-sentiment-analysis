# 🎯 YouTube Sentiment Analysis - Bahasa Indonesia

Aplikasi Streamlit untuk **mengambil komentar YouTube tanpa API key**, menganalisis sentimen menggunakan model **Indonesian RoBERTa** dari Hugging Face, serta menampilkan hasil dalam bentuk **tabel, pie chart, dan word cloud**.  
Hasil analisis dapat diunduh dalam format **Excel (.xlsx)**.

---

## 🚀 Fitur
- Ambil komentar dari video YouTube tanpa API key.
- Analisis sentimen bahasa Indonesia menggunakan model `w11wo/indonesian-roberta-base-sentiment-classifier`.
- Visualisasi **pie chart** distribusi sentimen.
- **Word Cloud** untuk kata-kata yang sering muncul.
- Ekspor hasil ke file Excel dan unduh langsung.

---

## 📦 Instalasi Lokal

### 1️⃣ Clone Repository
```bash
git clone https://github.com/username/repo-name.git
cd repo-name
```

###2️⃣ Buat Virtual Environment (Opsional tapi Disarankan)
```bash
python -m venv env
# Windows
env\Scripts\activate
# Mac/Linux
source env/bin/activate
```
###3️⃣ Install Dependencies
```bash
pip install -r requirements.txt
```

###▶️ Menjalankan Aplikasi di Lokal
```bash
streamlit run streamlit_app.py
http://localhost:8501
```
