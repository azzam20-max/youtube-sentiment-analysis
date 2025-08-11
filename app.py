import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from youtube_comment_downloader import YoutubeCommentDownloader
from transformers import pipeline
import nltk
from nltk.corpus import stopwords
import re
import io

# Cache NLTK stopwords
@st.cache_data
def get_stopwords():
    nltk.download('stopwords')
    return set(stopwords.words('indonesian') + stopwords.words('english'))

# Cache sentiment model
@st.cache_resource
def load_model():
    return pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")

# Download YouTube comments
@st.cache_data
def fetch_comments(video_url, limit=50):
    downloader = YoutubeCommentDownloader()
    comments = []
    try:
        for comment in downloader.get_comments_from_url(video_url, sort_by='top'):
            comments.append(comment["text"])
            if len(comments) >= limit:
                break
    except Exception as e:
        st.error(f"Gagal mengambil komentar: {e}")
    return comments

# Bersihkan teks
def clean_text(text, stop_words):
    text = re.sub(r"http\S+|www\S+|https\S+", "", text)
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    text = text.lower()
    text = " ".join([word for word in text.split() if word not in stop_words])
    return text

# Buat WordCloud
def create_wordcloud(text):
    wc = WordCloud(width=800, height=400, background_color="white").generate(text)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.imshow(wc, interpolation="bilinear")
    ax.axis("off")
    return fig

# Streamlit UI
st.set_page_config(page_title="YouTube Sentiment Analysis", layout="wide")
st.title("🎯 YouTube Sentiment Analysis")
st.write("Masukkan URL video YouTube untuk menganalisis sentimen komentar.")

video_url = st.text_input("URL Video YouTube", placeholder="https://www.youtube.com/watch?v=xxxxxxx")
limit = st.slider("Jumlah komentar yang dianalisis", 10, 200, 50)

if st.button("🔍 Analisis"):
    if video_url:
        stop_words = get_stopwords()
        model = load_model()
        
        comments = fetch_comments(video_url, limit)
        if not comments:
            st.warning("Tidak ada komentar yang ditemukan.")
        else:
            cleaned_comments = [clean_text(c, stop_words) for c in comments]
            results = model(cleaned_comments)

            df = pd.DataFrame({
                "Comment": comments,
                "Cleaned": cleaned_comments,
                "Sentiment": [r['label'] for r in results],
                "Score": [r['score'] for r in results]
            })

            st.subheader("📊 Hasil Analisis Sentimen")
            st.dataframe(df)

            # Pie chart
            sentiment_counts = df['Sentiment'].value_counts()
            fig, ax = plt.subplots()
            ax.pie(sentiment_counts, labels=sentiment_counts.index, autopct='%1.1f%%')
            ax.axis("equal")
            st.pyplot(fig)

            # WordCloud
            st.subheader("☁️ WordCloud")
            text_all = " ".join(df["Cleaned"])
            st.pyplot(create_wordcloud(text_all))

            # Download Excel
            output = io.BytesIO()
            with pd.ExcelWriter(output, engine='openpyxl') as writer:
                df.to_excel(writer, index=False)
            st.download_button(
                label="📥 Download Hasil (Excel)",
                data=output.getvalue(),
                file_name="sentiment_results.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
    else:
        st.error("Masukkan URL video terlebih dahulu!")
