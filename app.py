# app.py
# Social Media Trend Analysis – Streamlit Web App
# Author: Par (Solo Project)
# Final Project – NLP Course

import streamlit as st
import pandas as pd
import numpy as np
import re
import emoji
import spacy
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.decomposition import TruncatedSVD
from gensim import corpora
from gensim.models import LdaModel
import pyLDAvis
import pyLDAvis.gensim_models
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import os
import warnings
warnings.filterwarnings("ignore")

# -------------------------- Page Config --------------------------
st.set_page_config(page_title="Social Media Trend Analysis", layout="wide")
st.title("Social Media Trend Analysis")
st.markdown("""
**Discover trending topics on Twitter and see how positive or negative people feel about them**  
Built using LDA Topic Modeling + VADER Sentiment Analysis  
Dataset: Sentiment140 (1.6 million tweets) → sampled & balanced
""")

# -------------------------- Sidebar --------------------------
st.sidebar.header("About")
st.sidebar.info("Solo project by **Pardis Shoumali**  \nFinal NLP Project  \nDecember 2025")
st.sidebar.markdown("### Features")
st.sidebar.markdown("- Interactive topic exploration (pyLDAvis)  \n- Sentiment intensity per trend  \n- Word clouds & frequency charts")

# -------------------------- Caching Heavy Steps --------------------------
@st.cache_data(show_spinner="Downloading and loading 1.6M tweets...")
def load_data():
    # Auto-download if file not present (works on Streamlit Cloud too)
    url = "http://cs.stanford.edu/people/alecmgo/trainingandtestdata.zip"
    zip_path = "trainingandtestdata.zip"
    csv_path = "training.140.csv"

    if not os.path.exists(csv_path):
        import urllib.request
        st.info("Downloading Sentiment140 dataset (~80 MB)...")
        urllib.request.urlretrieve(url, zip_path)
        st.info("Extracting...")
        import zipfile
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(".")
        os.remove(zip_path)

    # Load without headers
    df = pd.read_csv(csv_path, encoding="latin-1", header=None)
    df.columns = ["polarity", "id", "date", "query", "user", "text"]
    df = df[["polarity", "text"]]
    df["polarity"] = df["polarity"].map({0: "negative", 4: "positive"})
    return df

@st.cache_data(show_spinner="Stratified sampling 60,000 tweets...")
def sample_data(df):
    # Balanced 30k positive + 30k negative
    pos = df[df.polarity == "positive"].sample(n=30000, random_state=42)
    neg = df[df.polarity == "negative"].sample(n=30000, random_state=42)
    sample_df = pd.concat([pos, neg]).reset_index(drop=True)
    return sample_df

@st.cache_data(show_spinner="Cleaning text...")
def clean_text(text):
    if pd.isna(text):
        return ""
    # Remove URLs, mentions, hashtags
    text = re.sub(r"http\S+|www\S+|https\S+", "", text, flags=re.MULTILINE)
    text = re.sub(r"@\w+|#\w+", "", text)
    # Emoji → text description
    text = emoji.demojize(text, delimiters=(" ", " "))
    text = re.sub(r":[a-z_]+:", " ", text)
    text = text.lower()
    return text.strip()

@st.cache_data(show_spinner="Tokenizing & lemmatizing with spaCy...")
def preprocess_texts(df):
    nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])
    stopwords = spacy.lang.en.stop_words.STOP_WORDS

    def tokenize_lemmatize(text):
        doc = nlp(text)
        tokens = [token.lemma_ for token in doc if token.is_alpha and token.lemma_ not in stopwords and len(token.lemma_) > 2]
        return tokens

    df["clean_text"] = df["text"].apply(clean_text)
    df["tokens"] = df["clean_text"].apply(tokenize_lemmatize)
    return df

@st.cache_resource(show_spinner="Training LDA model (10 topics)...")
def train_lda_model(tokens_list):
    dictionary = corpora.Dictionary(tokens_list)
    dictionary.filter_extremes(no_below=15, no_above=0.5, keep_n=100000)
    corpus = [dictionary.doc2bow(tokens) for tokens in tokens_list]
    lda = LdaModel(corpus, num_topics=10, id2word=dictionary, passes=15, random_state=42)
    return lda, corpus, dictionary

@st.cache_data(show_spinner="Running sentiment analysis + topic assignment...")
def analyze_sentiment_and_topics(df, lda, corpus, dictionary):
    analyzer = SentimentIntensityAnalyzer()

    # Get dominant topic for each tweet
    topics = []
    for bow in corpus:
        topic_dist = lda.get_document_topics(bow, minimum_probability=0.0)
        dominant = max(topic_dist, key=lambda x: x[1])[0]
        topics.append(dominant)
    df["topic"] = topics

    # VADER compound score
    df["sentiment_score"] = df["clean_text"].apply(lambda x: analyzer.polarity_scores(x)["compound"])

    # Aggregate per topic
    summary = df.groupby("topic").agg(
        topic_size=("id", "count"),
        avg_sentiment=("sentiment_score", "mean")
    ).reset_index()

    # Get top 10 words for each topic
    topic_words = []
    for i in range(10):
        words = [word for word, _ in lda.show_topic(i, topn=10)]
        topic_words.append(" | ".join(words))
    summary["top_words"] = topic_words

    summary = summary.sort_values("topic_size", ascending=False)
    summary["topic_name"] = [f"Trend {i+1}" for i in summary.index]
    summary = summary[["topic_name", "topic_size", "top_words", "avg_sentiment"]]
    summary["sentiment_label"] = summary["avg_sentiment"].apply(
        lambda x: "Positive" if x > 0.1 else ("Negative" if x < -0.1 else "Neutral")
    )
    return df, summary

# -------------------------- Main Execution --------------------------
df_raw = load_data()
df = sample_data(df_raw)
df = preprocess_texts(df)

tokens_list = df["tokens"].tolist()

lda_model, corpus, dictionary = train_lda_model(tokens_list)

df, trend_summary = analyze_sentiment_and_topics(df, lda_model, corpus, dictionary)

# -------------------------- Display Results --------------------------
st.success("Analysis Complete! Explore the results below")

st.header("1. Trend Sentiment Summary")
st.dataframe(trend_summary.style.format({"avg_sentiment": "{:.3f}"}))

st.download_button(
    label="Download summary as CSV",
    data=trend_summary.to_csv(index=False).encode(),
    file_name="trend_sentiment_summary.csv",
    mime="text/csv"
)

st.header("2. Interactive Topic Exploration")
vis = pyLDAvis.gensim_models.prepare(lda_model, corpus, dictionary, sort_topics=False)
pyLDAvis.save_html(vis, "lda_vis.html")
with open("lda_vis.html", "r", encoding="utf-8") as f:
    html_string = f.read()
st.components.v1.html(html_string, height=800, scrolling=True)

st.header("3. Word Clouds")
col1, col2 = st.columns(2)

with col1:
    st.subheader("Positive Tweets")
    pos_text = " ".join(df[df.sentiment_score > 0.3]["clean_text"])
    if pos_text:
        wc_pos = WordCloud(width=800, height=400, background_color="white").generate(pos_text)
        fig, ax = plt.subplots()
        ax.imshow(wc_pos, interpolation="bilinear")
        ax.axis("off")
        st.pyplot(fig)

with col2:
    st.subheader("Negative Tweets")
    neg_text = " ".join(df[df.sentiment_score < -0.3]["clean_text"])
    if neg_text:
        wc_neg = WordCloud(width=800, height=400, background_color="black", colormap="Reds").generate(neg_text)
        fig2, ax2 = plt.subplots()
        ax2.imshow(wc_neg, interpolation="bilinear")
        ax2.axis("off")
        st.pyplot(fig2)

st.info("App finished.")