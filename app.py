# app.py - Dec 2025
import streamlit as st
import pandas as pd
import numpy as np
import re
from gensim import corpora
from gensim.models import LdaModel
import pyLDAvis
import pyLDAvis.gensim_models
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

st.set_page_config(page_title="Social Media Trends", layout="wide")
st.title("Social Media Trend & Sentiment Analysis")
st.markdown("**Twitter trends using LDA + VADER** • Solo project by Pardis")

# Load pre-processed balanced 30k tweets (8 MB)
@st.cache_data
def load_data():
    url = "https://github.com/pardissh/Social_Media_Trend_Analysis/blob/ce21f62d82d7273354376feb9b4d1226758ff081/tweets_30k_balanced.pkl"
    # Change the URL above to your actual GitHub raw link after upload!
    df = pd.read_pickle(url)
    df["polarity"] = df["polarity"].map({0: "negative", 1: "positive"})
    return df

@st.cache_data(show_spinner="Cleaning text...")
def clean_text(text):
    text = re.sub(r"http\S+|www\S+|https\S+", "", text, flags=re.MULTILINE)
    text = re.sub(r"@\w+|#\w+", "", text)
    text = text.lower()
    return text

@st.cache_data(show_spinner="Tokenizing...")
def tokenize(texts):
    return [[word for word in doc.lower().split() 
             if word.isalpha() and len(word) > 2 and word not in 
             {'http', 'https', 'www', 'com', 'like', 'get', 'one', 'would', 'go'}]
            for doc in texts]

@st.cache_resource(show_spinner="Training LDA model...")
def train_lda(tokens):
    dictionary = corpora.Dictionary(tokens)
    dictionary.filter_extremes(no_below=10, no_above=0.5)
    corpus = [dictionary.doc2bow(t) for t in tokens]
    lda = LdaModel(corpus, num_topics=10, id2word=dictionary, passes=10, random_state=42)
    return lda, corpus, dictionary

@st.cache_data(show_spinner="Running sentiment analysis...")
def analyze(df, lda, corpus, dictionary):
    analyzer = SentimentIntensityAnalyzer()
    df["clean"] = df["text"].apply(clean_text)
    df["sentiment"] = df["clean"].apply(lambda x: analyzer.polarity_scores(x)["compound"])
    df["topic"] = [max(lda.get_document_topics(bow), key=lambda x: x[1])[0] for bow in corpus]

    summary = df.groupby("topic").agg(
        size=("text", "count"),
        avg_sentiment=("sentiment", "mean")
    ).round(3).reset_index()
    summary["top_words"] = [" | ".join([w for w,_ in lda.show_topic(i, 10)]) for i in range(10)]
    summary["trend"] = [f"Trend {i+1}" for i in range(10)]
    summary = summary[["trend", "size", "top_words", "avg_sentiment"]]
    summary["feeling"] = summary["avg_sentiment"].apply(
        lambda x: "Positive" if x > 0.05 else ("Negative" if x < -0.05 else "Neutral"))
    return summary

# === RUN ===
df = load_data()
tokens = tokenize(df["text"])
lda_model, corpus, dictionary = train_lda(tokens)
summary = analyze(df, lda_model, corpus, dictionary)

# === DISPLAY ===
st.success("Analysis ready.")
st.dataframe(summary.style.format({"avg_sentiment": "{:.3f}"}))

st.download_button("Download CSV", summary.to_csv(index=False).encode(), "trends.csv")

# pyLDAvis
vis = pyLDAvis.gensim_models.prepare(lda_model, corpus, dictionary)
pyLDAvis.save_html(vis, "viz.html")
with open("viz.html", "r", encoding="utf-8") as f:
    st.components.v1.html(f.read(), height=800, scrolling=True)

# Word clouds
c1, c2 = st.columns(2)
with c1:
    st.subheader("Positive tweets")
    pos = " ".join(df[df.sentiment > 0.3]["clean"])
    if pos:
        wc = WordCloud(width=600, height=300, background_color="white").generate(pos)
        plt.figure(figsize=(8,4)); plt.imshow(wc); plt.axis("off")
        st.pyplot(plt)
with c2:
    st.subheader("Negative tweets")
    neg = " ".join(df[df.sentiment < -0.3]["clean"])
    if neg:
        wc = WordCloud(width=600, height=300, background_color="black", colormap="Reds").generate(neg)
        plt.figure(figsize=(8,4)); plt.imshow(wc); plt.axis("off")
        st.pyplot(plt)