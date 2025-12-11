# app.py - Dec 2025 
import streamlit as st
import pandas as pd
import numpy as np
import re
import requests
from io import BytesIO
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

# CORRECT WAY TO LOAD PICKLE FROM GITHUB
@st.cache_data(show_spinner="Loading 30,000 tweets...")
def load_data():
    # CHANGE THIS URL TO YOUR REAL GITHUB RAW LINK
    url = "https://github.com/pardissh/Social_Media_Trend_Analysis/raw/main/tweets_30k_balanced.pkl"

    response = requests.get(url)
    response.raise_for_status() 
    df = pd.read_pickle(BytesIO(response.content))
    df["polarity"] = df["polarity"].map({0: "negative", 1: "positive"})
    return df

@st.cache_data(show_spinner="Cleaning & tokenizing...")
def preprocess(df):
    def clean(text):
        text = re.sub(r"http\S+|www\S+|https\S+", "", str(text))
        text = re.sub(r"@\w+|#\w+", "", text)
        return text.lower().strip()

    df["clean"] = df["text"].apply(clean)
    # Simple whitespace tokenizer + filter
    stop_words = {"the", "a", "an", "and", "or", "but", "if", "while", "at", "by", "for", "with", "about", "to", "from", "of", "in", "on", "is", "are", "was", "were", "been", "have", "has", "had", "do", "does", "did", "i", "you", "he", "she", "it", "we", "they", "this", "that", "rt"}
    df["tokens"] = df["clean"].apply(lambda x: [w for w in x.split() if len(w)>2 and w.isalpha() and w not in stop_words])
    return df

@st.cache_resource(show_spinner="Training LDA (10 topics)...")
def train_lda(tokens):
    dictionary = corpora.Dictionary(tokens)
    dictionary.filter_extremes(no_below=10, no_above=0.5)
    corpus = [dictionary.doc2bow(text) for text in tokens)]
    lda = LdaModel(corpus, num_topics=10, id2word=dictionary, passes=12, random_state=42)
    return lda, corpus, dictionary

@st.cache_data(show_spinner="Analyzing sentiment...")
def analyze_sentiment_and_topics(df, lda, corpus, dictionary):
    analyzer = SentimentIntensityAnalyzer()
    df["sentiment"] = df["clean"].apply(lambda x: analyzer.polarity_scores(x)["compound"])

    # Get dominant topic
    topics = []
    for bow in corpus:
        topic_dist = lda.get_document_topics(bow, minimum_probability=0)
        dominant = max(topic_dist, key=lambda x: x[1])[0] if topic_dist else 0
        topics.append(dominant)
    df["topic"] = topics

    # Summary table
    summary = df.groupby("topic").agg(
        size=("text", "count"),
        avg_sentiment=("sentiment", "mean")
    ).round(3).reset_index()

    summary["top_words"] = [ " | ".join([w for w,_ in lda.show_topic(i, 10)]) for i in range(10) ]
    summary["trend"] = [f"Trend {i+1}" for i in range(10)]
    summary = summary[["trend", "size", "top_words", "avg_sentiment"]]
    summary["feeling"] = summary["avg_sentiment"].apply(lambda x: "Positive" if x > 0.05 else ("Negative" if x < -0.05 else "Neutral"))
    return summary

# ==================== RUN ====================
df = load_data()
df = preprocess(df)
lda_model, corpus, dictionary = train_lda(df["tokens"].tolist())
summary = analyze_sentiment_and_topics(df, lda_model, corpus, dictionary)

# ==================== DISPLAY ====================
st.success("Analysis complete!")
st.dataframe(summary.style.format({"avg_sentiment": "{:.3f}"}))

st.download_button("Download Summary CSV", summary.to_csv(index=False).encode(), "trend_summary.csv")

# Interactive topic visualization
vis = pyLDAvis.gensim_models.prepare(lda_model, corpus, dictionary, sort_topics=False)
pyLDAvis.save_html(vis, "lda.html")
with open("lda.html", "r", encoding="utf-8") as f:
    st.components.v1.html(f.read(), height=800, scrolling=True)

# Word clouds
c1, c2 = st.columns(2)
with c1:
    st.subheader("Positive Tweets")
    pos_text = " ".join(df[df.sentiment > 0.3]["clean"])
    if pos_text:
        wc = WordCloud(width=600, height=400, background_color="white").generate(pos_text)
        plt.figure(figsize=(8,5))
        plt.imshow(wc, interpolation="bilinear")
        plt.axis("off")
        st.pyplot(plt)

with c2:
    st.subheader("Negative Tweets")
    neg_text = " ".join(df[df.sentiment < -0.3]["clean"])
    if neg_text:
        wc = WordCloud(width=600, height=400, background_color="black", colormap="Reds").generate(neg_text)
        plt.figure(figsize=(8,5))
        plt.imshow(wc, interpolation="bilinear")
        plt.axis("off")
        st.pyplot(plt)

st.caption("Solo project by Pardis • NLP Final Project 2025")