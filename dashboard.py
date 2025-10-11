import streamlit as st
import joblib
import re
import nltk
import emoji

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Load saved model and vectorizer
model = joblib.load('src/sentiment_model.pkl')
vectorizer = joblib.load('src/tfidf_vectorizer.pkl')

# NLTK setup
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4')

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

# Text cleaning function (same logic used in training)
def clean_tweet(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\S+|https\S+", '', text)
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'#', '', text)
    text = re.sub(r'[^A-Za-z\s]', '', text)
    tokens = nltk.word_tokenize(text)
    tokens = [word for word in tokens if word not in stop_words and len(word) > 2]
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    text = emoji.demojize(" ".join(tokens))
    return text

# Streamlit UI
st.set_page_config(page_title="Sentiment Analyzer", page_icon="💬", layout="centered")

st.title("💬 Twitter Sentiment Analysis App")
st.write("Enter a tweet or sentence below to analyze its sentiment (positive or negative).")

# Input text
user_input = st.text_area("✍️ Write something here...")

if st.button("Analyze Sentiment"):
    if user_input.strip():
        cleaned_text = clean_tweet(user_input)
        transformed_text = vectorizer.transform([cleaned_text])
        prediction = model.predict(transformed_text)[0]
        sentiment = "😊 Positive" if prediction == 1 else "😠 Negative"
        st.subheader(f"Predicted Sentiment: {sentiment}")
    else:
        st.warning("Please enter some text first.")
