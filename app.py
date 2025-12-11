# app.py - Dec 2025 
import streamlit as st
import pandas as pd
import numpy as np
import re
import spacy
from gensim import corpora
from gensim.models import LdaModel
import pyLDAvis
import pyLDAvis.gensim_models
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import pickle

st.set_page_config(page_title="Social Media Trend Analysis", layout="wide")
st.title("Social Media Trend Analysis")
st.markdown("**Trending topics + sentiment intensity on Twitter** • Solo project by Par")

# Load pre-processed 30k tweets (super fast)
@st.cache_data
def load_data():
    # I used small version of dataset for better performance
    df = pd.read_pickle("tweets_30k_balanced.pkl")
    df["polarity"] = df["polarity"].map({0: "negative", 1: "positive"})
    return df

@st.cache_data(show_spinner="Cleaning & tokenizing...")
def preprocess(df):
    nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])
    stopwords = nlp.Defaults.stop_words

    def clean(text):
        text = re.sub(r"http\S+|www\S+|https\S+", "", text)
        text = re.sub(r"@\w+|#\w+", "", text)
        text = text.lower()
        return text

    def tokenize(text):
        doc = nlp(text)
        return [token.lemma_ for token in doc if token.is_alpha and token.lemma_ not in stopwords and len(token.lemma_) > 2]

    df["clean"] = df["text"].apply(clean)
    df["tokens"] = df["clean"].apply(tokenize)
    return df

@st.cache_resource(show_spinner="Training LDA (10 topics)...")
def train_lda(tokens):
    dictionary = corpora.Dictionary(tokens)
    dictionary.filter_extremes(no_below=10, no_above=0.5)
    dictionary = corpora.Dictionary(tokens_list)
    dictionary.filter_extremes(no_below=10, no_above=0.5)
    corpus = [dictionary.doc2bow(t) for t in tokens_list]
    lda = LdaModel(corpus, num_topics=10, id2word=dictionary, passes=15, random_state=42)
    return lda, corpus, dictionary

@st.cache_data(show_spinner="Sentiment & summary...")
def analyze(df, lda, corpus, dictionary):
    analyzer = SentimentIntensityAnalyzer()
    df["sentiment"] = df["clean"].apply(lambda x: analyzer.polarity_scores(x)["compound"])
    topics = [max(lda.get_document_topics(bow), key=lambda x: x[1])[0] for bow in corpus]
    df["topic"] = topics

    summary = df.groupby("topic").agg(
        size=("text", "count"),
        avg_sentiment=("sentiment", "mean")
    ).round(3).reset_index()

    summary["top_words"] = [ " | ".join([w for w,_ in lda.show_topic(i, 10)]) for i in range(10) ]
    summary["trend"] = [f"Trend {i+1}" for i in range(10)]
    summary = summary[["trend", "size", "top_words", "avg_sentiment"]]
    summary["feeling"] = summary["avg_sentiment"].apply(lambda x: "Positive" if x>0.05 else ("Negative" if x<-0.05 else "Neutral"))
    return summary

# Run everything
df = load_data()
df = preprocess(df)
tokens_list = df["tokens"].tolist()
lda, corpus, dict_ = train_lda(tokens_list)
summary = analyze(df, lda, corpus, dict_)

# Display
st.success("Ready in seconds!")
st.dataframe(summary.style.format({"avg_sentiment": "{:.3f}"})

st.download_button("Download summary CSV", summary.to_csv(index=False).encode(), "trend_summary.csv")

vis = pyLDAvis.gensim_models.prepare(lda, corpus, dict_, sort_topics=False)
html_file = "viz.html"
pyLDAvis.save_html(vis, html_file)
with open(html_file, "r", encoding="utf-8") as f:
    st.components.v1.html(f.read(), height=800, scrolling=True)

# Word clouds
col1, col2 = st.columns(2)
with col1:
    st.subheader("Positive tweets")
    pos_text = " ".join(df[df.sentiment > 0.3]["clean"])
    if pos_text:
        wc = WordCloud(width=600, height=300, background_color="white").generate(pos_text)
        fig, ax = plt.subplots()
        ax.imshow(wc)
        ax.axis("off")
        st.pyplot(fig)

with col2:
    st.subheader("Negative tweets")
    neg_text = " ".join(df[df.sentiment < -0.3]["clean"])
    if neg_text:
        wc = WordCloud(width=600, height=300, background_color="black", colormap="Reds").generate(neg_text)
        fig2, ax2 = plt.subplots()
        ax2.imshow(wc)
        ax2.axis("off")
        st.pyplot(fig2)