# ============================================
# 2️⃣ Import Library
# ============================================
from youtube_comment_downloader import YoutubeCommentDownloader
from transformers import pipeline
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import nltk
import re

# Download stopwords NLTK
nltk.download('stopwords')
from nltk.corpus import stopwords

# ============================================
# 3️⃣ Fungsi Bersih Teks
# ============================================
def clean_text(text):
    text = re.sub(r"http\S+", "", text)  # hapus URL
    text = re.sub(r"[^a-zA-Z\s]", "", text)  # hapus simbol/angka
    text = text.lower().strip()
    return text

# ============================================
# 4️⃣ Ambil Komentar dari YouTube (Tanpa API Key)
# ============================================
video_url = "https://youtu.be/6deJ_lSHnYg"  # GANTI link video YouTube di sini
max_comments = None  # None = ambil semua, angka = batas jumlah komentar

downloader = YoutubeCommentDownloader()
comments_gen = downloader.get_comments_from_url(video_url, sort_by=0)  # 0=Top, 1=Newest

comments_list = []
for i, comment in enumerate(comments_gen):
    if max_comments is not None and i >= max_comments:
        break
    comments_list.append(comment['text'])

print(f"✅ Berhasil mengambil {len(comments_list)} komentar.")

# ============================================
# 5️⃣ Analisis Sentimen
# ============================================
MODEL_NAME = "w11wo/indonesian-roberta-base-sentiment-classifier"  # Model Sentiment Indonesia
sentiment_analyzer = pipeline("sentiment-analysis", model=MODEL_NAME)

results = []
for c in comments_list:
    sentiment = sentiment_analyzer(c)[0]
    results.append({
        "komentar": c,
        "sentimen": sentiment["label"],
        "skor": round(sentiment["score"], 3)
    })

df = pd.DataFrame(results)
print(df.head())

# ============================================
# 6️⃣ Visualisasi Pie Chart
# ============================================
sentiment_counts = df["sentimen"].value_counts()
fig, ax = plt.subplots()
ax.pie(sentiment_counts, labels=sentiment_counts.index, autopct="%1.1f%%", startangle=90)
ax.axis("equal")
plt.title("Distribusi Sentimen Komentar YouTube")
plt.show()

# ============================================
# 7️⃣ Word Cloud
# ============================================
stop_words = set(stopwords.words("indonesian"))
all_text = " ".join([clean_text(c) for c in df["komentar"]])
all_text = " ".join([word for word in all_text.split() if word not in stop_words])

wordcloud = WordCloud(width=800, height=400, background_color="white").generate(all_text)
plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.show()

# ============================================
# 8️⃣ Simpan ke Excel
# ============================================
df.to_excel("hasil_sentiment.xlsx", index=False)
print("💾 Hasil disimpan ke hasil_sentiment.xlsx")
