import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import nltk
from nltk.corpus import stopwords
from youtube_comment_downloader import YoutubeCommentDownloader
from transformers import pipeline

nltk.download('stopwords', quiet=True)

@st.cache_resource
def load_model():
    return pipeline("sentiment-analysis", framework="tf")  # pakai TensorFlow

@st.cache_resource
def load_stopwords():
    return set(stopwords.words('indonesian') + stopwords.words('english'))

@st.cache_data
def get_youtube_comments(url, max_comments=50):
    downloader = YoutubeCommentDownloader()
    comments = []
    for comment in downloader.get_comments_from_url(url, sort_by=0):
        comments.append(comment['text'])
        if len(comments) >= max_comments:
            break
    return comments

def analyze_comments(comments, model):
    results = model(comments)
    df = pd.DataFrame({
        "Comment": comments,
        "Label": [r['label'] for r in results],
        "Score": [r['score'] for r in results]
    })
    return df

def generate_wordcloud(comments, stop_words):
    text = " ".join(comments).lower()
    wc = WordCloud(
        stopwords=stop_words,
        background_color="white",
        width=800,
        height=400
    ).generate(text)
    return wc

st.title("🎯 YouTube Sentiment Analysis (Light Version)")
st.write("Masukkan URL video YouTube untuk analisis komentar dan visualisasi sentimen.")

youtube_url = st.text_input("🔗 Masukkan URL YouTube:", "")
max_comments = st.slider("Jumlah komentar yang dianalisis", 10, 200, 50)

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

                st.subheader("📋 Hasil Analisis Sentimen")
                st.dataframe(df)

                st.subheader("📊 Distribusi Sentimen")
                fig, ax = plt.subplots()
                df['Label'].value_counts().plot(kind='bar', ax=ax, color=['green', 'red'])
                plt.xticks(rotation=0)
                st.pyplot(fig)

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
