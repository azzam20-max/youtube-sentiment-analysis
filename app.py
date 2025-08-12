import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import nltk
from nltk.corpus import stopwords
from youtube_comment_downloader import YoutubeCommentDownloader
from transformers import pipeline

# Download stopwords
nltk.download('stopwords', quiet=True)

# Cache model sentiment
@st.cache_resource
def load_model():
    return pipeline(
        "sentiment-analysis",
        model="cardiffnlp/twitter-xlm-roberta-base-sentiment",  # MIT License
        tokenizer="cardiffnlp/twitter-xlm-roberta-base-sentiment",
        device=-1  # CPU
    )

# Cache stopwords
@st.cache_resource
def load_stopwords():
    return set(stopwords.words('indonesian') + stopwords.words('english'))

# Ambil komentar YouTube dengan progress bar
@st.cache_data
def get_youtube_comments(url, max_comments=None):
    downloader = YoutubeCommentDownloader()
    comments = []

    # Progress bar
    progress_bar = st.progress(0)
    total_estimate = max_comments if max_comments is not None else 500  # estimasi awal

    for i, comment in enumerate(downloader.get_comments_from_url(url, sort_by=0), start=1):
        comments.append(comment['text'])

        # Update progress bar
        if max_comments is not None:
            progress = min(i / max_comments, 1.0)
        else:
            progress = min(i / total_estimate, 1.0)
        progress_bar.progress(progress)

        # Break jika mencapai batas komentar
        if max_comments is not None and len(comments) >= max_comments:
            break

    progress_bar.empty()  # hapus progress bar
    return comments

# Analisis komentar
def analyze_comments(comments, model):
    results = model(comments)
    df = pd.DataFrame({
        "Comment": comments,
        "Label": [r['label'] for r in results],
        "Score": [r['score'] for r in results]
    })
    return df

# Buat Word Cloud
def generate_wordcloud(comments, stop_words):
    text = " ".join(comments).lower()
    wc = WordCloud(
        stopwords=stop_words,
        background_color="white",
        width=800,
        height=400
    ).generate(text)
    return wc

# ==== UI Streamlit ====
st.title("🎯 YouTube Sentiment Analysis (INGGRIS & INDONESIA)")
st.write("Masukkan URL video YouTube untuk analisis komentar dan visualisasi sentimen.")

# Input URL YouTube
youtube_url = st.text_input("🔗 Masukkan URL YouTube:", "")

# Pilihan jumlah komentar
option = st.selectbox(
    "Jumlah komentar yang dianalisis",
    ["10", "50", "100", "200", "Semua"],
    index=1
)

if option == "Semua":
    max_comments = None
else:
    max_comments = int(option)

if st.button("🚀 Jalankan Analisis") and youtube_url:
    with st.spinner("Mengunduh komentar & memproses..."):
        try:
            comments = get_youtube_comments(youtube_url, max_comments)
            if not comments:
                st.error("Tidak ada komentar ditemukan.")
            else:
                model = load_model()
                stop_words = load_stopwords()
                df = analyze_comments(comments, model)

                # Tampilkan tabel hasil
                st.subheader("📋 Hasil Analisis Sentimen")
                st.dataframe(df)

                # Grafik distribusi
                st.subheader("📊 Distribusi Sentimen")
                fig, ax = plt.subplots()
                df['Label'].value_counts().plot(kind='bar', ax=ax, color=['green', 'gray', 'red'])
                plt.xticks(rotation=0)
                st.pyplot(fig)

                # Word Cloud
                st.subheader("☁ Word Cloud")
                wc = generate_wordcloud(comments, stop_words)
                fig_wc, ax_wc = plt.subplots()
                ax_wc.imshow(wc, interpolation='bilinear')
                ax_wc.axis("off")
                st.pyplot(fig_wc)

        except Exception as e:
            st.error(f"Terjadi kesalahan: {e}")
elif youtube_url == "":
    st.info("Silakan masukkan URL video YouTube terlebih dahulu.")
