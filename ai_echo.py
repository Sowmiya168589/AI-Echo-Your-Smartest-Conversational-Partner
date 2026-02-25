

import streamlit as st
import pandas as pd
import numpy as np
import re
import os
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from wordcloud import WordCloud
import matplotlib.pyplot as plt

# ML
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import LabelEncoder

# DL
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Embedding, Conv1D, GlobalMaxPooling1D, Bidirectional
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical

# -----------------------------------------------------
# STREAMLIT CONFIG
# -----------------------------------------------------
st.set_page_config(page_title="AI Echo Industry", layout="wide")

# -----------------------------------------------------
# LOAD DATA
# -----------------------------------------------------
DATA_PATH = "D:/sowmiya/Python/chatgpt_style_reviews_dataset (11).csv"

if not os.path.exists(DATA_PATH):
    st.error("Dataset not found! Check file path.")
    st.stop()

df = pd.read_csv(DATA_PATH)

# -----------------------------------------------------
# CREATE 3-CLASS SENTIMENT
# -----------------------------------------------------
if "sentiment" not in df.columns:
    def label_sentiment(r):
        if r <= 2:
            return "Negative"
        elif r == 3:
            return "Neutral"
        else:
            return "Positive"
    df["sentiment"] = df["rating"].apply(label_sentiment)

# -----------------------------------------------------
# NLP CLEANING
# -----------------------------------------------------
nltk.download("stopwords")
nltk.download("wordnet")

stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"[^a-zA-Z ]", "", text)
    words = text.split()
    words = [lemmatizer.lemmatize(w) for w in words if w not in stop_words]
    return " ".join(words)

df["clean_review"] = df["review"].apply(clean_text)

# Safe date handling
if "date" in df.columns:
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"])

# -----------------------------------------------------
# SIDEBAR NAVIGATION
# -----------------------------------------------------
menu = st.sidebar.selectbox(
    "Navigation",
    ["EDA", "Model Comparison", "Prediction"]
)

# ===============================================================
# TRAIN ML + DL MODELS
# ===============================================================
@st.cache_resource
def train_models():

    X = df["clean_review"]
    y = df["sentiment"]

    # ---------------- ML ----------------
    tfidf = TfidfVectorizer(max_features=5000)
    X_tfidf = tfidf.fit_transform(X)

    X_train_ml, X_test_ml, y_train_ml, y_test_ml = train_test_split(
        X_tfidf, y, test_size=0.2, random_state=42
    )

    models = {}

    ml_models = {
        "Logistic Regression": LogisticRegression(),
        "Naive Bayes": MultinomialNB(),
        "Random Forest": RandomForestClassifier(),
        "SVM": LinearSVC()
    }

    for name, model in ml_models.items():
        model.fit(X_train_ml, y_train_ml)
        pred = model.predict(X_test_ml)

        models[name] = {
            "Model": model,
            "Accuracy": accuracy_score(y_test_ml, pred),
            "Precision": precision_score(y_test_ml, pred, average="weighted"),
            "Recall": recall_score(y_test_ml, pred, average="weighted"),
            "F1": f1_score(y_test_ml, pred, average="weighted")
        }

    # ---------------- DL ----------------
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    tokenizer = Tokenizer(num_words=10000)
    tokenizer.fit_on_texts(X)
    sequences = tokenizer.texts_to_sequences(X)
    padded = pad_sequences(sequences, maxlen=200)

    X_train_dl, X_test_dl, y_train_dl, y_test_dl = train_test_split(
        padded, y_encoded, test_size=0.2, random_state=42
    )

    y_train_cat = to_categorical(y_train_dl)
    y_test_cat = to_categorical(y_test_dl)

    def build_and_train(model):
        model.compile(loss="categorical_crossentropy",
                      optimizer="adam",
                      metrics=["accuracy"])
        model.fit(X_train_dl, y_train_cat, epochs=5, batch_size=32, verbose=0)
        loss, acc = model.evaluate(X_test_dl, y_test_cat, verbose=0)
        return model, acc

    # LSTM
    lstm = Sequential([
        Embedding(10000, 128, input_length=200),
        LSTM(64),
        Dense(3, activation="softmax")
    ])
    lstm_model, lstm_acc = build_and_train(lstm)
    models["LSTM"] = {"Model": lstm_model, "Accuracy": lstm_acc}

    # BiLSTM
    bilstm = Sequential([
        Embedding(10000, 128, input_length=200),
        Bidirectional(LSTM(64)),
        Dense(3, activation="softmax")
    ])
    bilstm_model, bilstm_acc = build_and_train(bilstm)
    models["BiLSTM"] = {"Model": bilstm_model, "Accuracy": bilstm_acc}

    # CNN
    cnn = Sequential([
        Embedding(10000, 128, input_length=200),
        Conv1D(128, 5, activation="relu"),
        GlobalMaxPooling1D(),
        Dense(3, activation="softmax")
    ])
    cnn_model, cnn_acc = build_and_train(cnn)
    models["CNN"] = {"Model": cnn_model, "Accuracy": cnn_acc}

    return models, tfidf, tokenizer, le


models, tfidf, tokenizer, label_encoder = train_models()

# ===============================================================
# EDA PAGE
# ===============================================================
if menu == "EDA":

    st.title("📊 Advanced EDA")

    st.subheader("Rating Distribution")
    st.bar_chart(df["rating"].value_counts())

    st.subheader("Sentiment Distribution")
    st.bar_chart(df["sentiment"].value_counts())

    if "date" in df.columns:
        st.subheader("Average Rating Over Time")
        df["year_month"] = df["date"].dt.to_period("M")
        trend = df.groupby("year_month")["rating"].mean()
        st.line_chart(trend)

    if "platform" in df.columns:
        st.subheader("Platform Analysis")
        st.bar_chart(df.groupby("platform")["rating"].mean())

    if "version" in df.columns:
        st.subheader("Version Analysis")
        st.bar_chart(df.groupby("version")["rating"].mean())

# ===============================================================
# MODEL COMPARISON PAGE
# ===============================================================
if menu == "Model Comparison":

    st.title("📈 ML + DL Model Comparison")

    comparison_df = pd.DataFrame(
        [(model, models[model]["Accuracy"]) for model in models],
        columns=["Model", "Accuracy"]
    )

    st.bar_chart(comparison_df.set_index("Model"))

    selected_model = st.selectbox("Select Model", list(models.keys()))

    st.write("Accuracy:", round(models[selected_model]["Accuracy"], 4))

# ===============================================================
# PREDICTION PAGE
# ===============================================================
if menu == "Prediction":

    st.title("🔮 Predict Sentiment")

    selected_model = st.selectbox("Choose Model", list(models.keys()))
    user_input = st.text_area("Enter Review Text")

    if st.button("Predict"):

        cleaned = clean_text(user_input)

        if selected_model in ["Logistic Regression", "Naive Bayes", "Random Forest", "SVM"]:
            vec = tfidf.transform([cleaned])
            prediction = models[selected_model]["Model"].predict(vec)[0]
        else:
            seq = tokenizer.texts_to_sequences([cleaned])
            pad = pad_sequences(seq, maxlen=200)
            pred = models[selected_model]["Model"].predict(pad)
            prediction = label_encoder.inverse_transform([np.argmax(pred)])[0]

        st.success(f"Predicted Sentiment: {prediction}")