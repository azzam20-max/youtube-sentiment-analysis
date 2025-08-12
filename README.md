# YouTube Comment Sentiment Analysis

Proyek ini adalah aplikasi berbasis **Streamlit** untuk menganalisis sentimen komentar YouTube menggunakan model NLP dari **HuggingFace Transformers**.  
Aplikasi dapat menampilkan **visualisasi WordCloud** dan grafik sentimen dari komentar yang diunduh.

## 🚀 Fitur
- Mengunduh komentar dari video YouTube.
- Membersihkan dan memproses teks (Natural Language Processing).
- Analisis sentimen menggunakan model multilingual.
- Visualisasi:
  - WordCloud
  - Grafik sentimen (positif, netral, negatif)

## 📦 Instalasi

### Clone repository
```bash
git clone https://github.com/username/repository-name.git
cd repository-name
```

### Buat virtual environment (opsional)
```bash
python -m venv venv
source venv/bin/activate  # MacOS/Linux
venv\Scripts\activate     # Windows
```

### Install dependencies
```bash
pip install -r requirements.txt
```
### Jalankan aplikasi
```bash
streamlit run app.py
```

##⚙️ Konfigurasi
  - Pastikan Anda memiliki koneksi internet untuk mengunduh model NLP.
  - Gunakan URL video YouTube yang dapat diakses publik.
  - Perhatikan aturan penggunaan dan ToS dari YouTube.

##📚 Teknologi yang Digunakan
  - Python
  - Streamlit - Apache 2.0 License
  - Pandas - BSD 3-Clause License
  - Matplotlib - PSF License
  - WordCloud - MIT License
  - NLTK - Apache 2.0 License
  - YouTube Comment Downloader - MIT License
  - Transformers (HuggingFace) - Apache 2.0 License
  - PyTorch - BSD-3 License

##📜 Lisensi
```LISENSI
Proyek ini dirilis di bawah lisensi MIT.
Anda bebas untuk menggunakan, menyalin, memodifikasi, dan mendistribusikan proyek ini, dengan atau tanpa perubahan, selama mencantumkan kredit kepada pembuat asli.
```
MIT License

Copyright (c) 2025 Azzam

Permission is hereby granted, free of charge, to any person obtaining a copy...
