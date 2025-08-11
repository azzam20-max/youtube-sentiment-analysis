# app.py
import streamlit as st
from youtube_comment_downloader import YoutubeCommentDownloader
from transformers import pipeline
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import nltk
import re
from io import BytesIO

# Download stopwords
nltk.download('stopwords')
from nltk.corpus import stopwords

# -------------------------------
# Fungsi Bersih Teks
# -------------------------------
def clean_text(text):
    text = re.sub(r"http\S+", "", text)  # hapus URL
    text = re.sub(r"[^a-zA-Z\s]", "", text)  # hapus simbol/angka
    text = text.lower().strip()
    return text

# -------------------------------
# Halaman Utama
# -------------------------------
st.set_page_config(page_title="YouTube Sentiment Analysis 🇮🇩", layout="wide")
st.title("📊 Analisis Sentimen Komentar YouTube (Bahasa Indonesia)")

video_url = st.text_input("Masukkan URL Video YouTube:")
max_comments = st.number_input("Jumlah komentar (0 = semua)", min_value=0, value=50)

if st.button("🔍 Analisis Sentimen"):
    if not video_url:
        st.warning("Masukkan URL YouTube terlebih dahulu.")
    else:
        with st.spinner("Mengambil komentar..."):
            try:
                downloader = YoutubeCommentDownloader()
                comments_gen = downloader.get_comments_from_url(video_url, sort_by=0)

                comments_list = []
                for i, comment in enumerate(comments_gen):
                    if max_comments > 0 and i >= max_comments:
                        break
                    comments_list.append(comment['text'])

                if not comments_list:
                    st.error("❌ Tidak ada komentar yang berhasil diambil.")
                    st.stop()

                st.success(f"✅ Berhasil mengambil {len(comments_list)} komentar.")

                # Analisis Sentimen
                st.info("Memproses analisis sentimen...")
                MODEL_NAME = "agufsamudra/indo-sentiment-analysis"
                sentiment_analyzer = pipeline(
                    "sentiment-analysis",
                    model=MODEL_NAME,
                    tokenizer=MODEL_NAME
                )

                results = []
                for c in comments_list:
                    sentiment = sentiment_analyzer(c)[0]
                    results.append({
                        "komentar": c,
                        "sentimen": sentiment["label"],
                        "skor": round(sentiment["score"], 3)
                    })

                df = pd.DataFrame(results)

                # Tampilkan tabel
                st.subheader("📄 Hasil Analisis")
                st.dataframe(df)

                # Visualisasi Pie Chart
                st.subheader("📊 Distribusi Sentimen")
                sentiment_counts = df["sentimen"].value_counts()
                fig1, ax1 = plt.subplots()
                ax1.pie(sentiment_counts, labels=sentiment_counts.index, autopct="%1.1f%%", startangle=90)
                ax1.axis("equal")
                st.pyplot(fig1)

                # Word Cloud
                st.subheader("☁ Word Cloud Komentar")
                stop_words = set(stopwords.words("indonesian"))
                all_text = " ".join([clean_text(c) for c in df["komentar"]])
                all_text = " ".join([word for word in all_text.split() if word not in stop_words])
                wordcloud = WordCloud(width=800, height=400, background_color="white").generate(all_text)

                fig2, ax2 = plt.subplots(figsize=(10, 5))
                ax2.imshow(wordcloud, interpolation="bilinear")
                ax2.axis("off")
                st.pyplot(fig2)

                # Download Excel
                st.subheader("💾 Download Hasil")
                output = BytesIO()
                df.to_excel(output, index=False)
                st.download_button(
                    label="Download Hasil Analisis (.xlsx)",
                    data=output,
                    file_name="hasil_sentiment.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )

            except Exception as e:
                st.error(f"Terjadi kesalahan: {e}")
