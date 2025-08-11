# 🎯 YouTube Sentiment Analysis

Aplikasi ini menganalisis sentimen komentar pada video YouTube menggunakan **Hugging Face Transformers** dan menampilkan hasilnya dalam bentuk tabel, grafik distribusi, dan word cloud.

---

## ✨ Fitur
- 🎥 Input URL YouTube langsung dari halaman web.
- 🤖 Analisis sentimen komentar menggunakan model **transformers**.
- 📊 Visualisasi distribusi sentimen (positive/negative).
- ☁ Auto-cache model & stopwords agar lebih cepat di cloud.
- ☁ Bisa dijalankan secara lokal atau di **Streamlit Cloud**.

---

## 📦 Persyaratan

Pastikan sudah menginstal:
- Python **3.8 – 3.11**
- pip (Python package manager)

---

## 💻 Menjalankan Secara Lokal

1. **Clone repository**
   ```bash
   git clone https://github.com/username/youtube-sentiment-analysis.git
   cd youtube-sentiment-analysis
   ```
2. **Buat virtual environment & aktifkan**
    ```bash
    python -m venv env
    # Windows
    env\Scripts\activate
    # Mac/Linux
    source env/bin/activate
    ```
3. **Install dependencies**
    ```bash
    pip install -r requirements.txt
    ```
4. **Jalankan aplikasi**
    ```bash
    streamlit run app.py
    ```
Local URL: http://localhost:8501

## ☁ Deploy ke Streamlit Cloud
1. **Push project ke GitHub**
- Pastikan semua file (app.py, requirements.txt, dll) sudah ada di repo.

2. **Buka Streamlit Cloud**
- Login menggunakan akun GitHub.

3. **Klik "New app"**
- Pilih repository GitHub project.
- Pilih branch (misalnya main).
- File path: app.py

4. **Deploy**
- Klik tombol Deploy.
- Tunggu proses instalasi dan aplikasi akan langsung berjalan di cloud.
