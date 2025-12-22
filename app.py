import streamlit as st
import pandas as pd
import string
from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from nltk.stem import WordNetLemmatizer
import nltk

# تحميل WordNet لكل جلسة تشغيل
for pkg in ['wordnet', 'omw-1.4']:
    try:
        nltk.data.find(f'corpora/{pkg}')
    except LookupError:
        nltk.download(pkg)

lemmatizer = WordNetLemmatizer()

# التحيات
greetings = {
    "hi": "Hello! How can I help you today?",
    "hello": "Hi there! How can I help you?",
    "hey": "Hey! How are you feeling today?",
    "bye": "Goodbye! Take care!",
    "goodbye": "See you later!"
}

# Load CSV
df = pd.read_csv("Mental_Health_FAQ.csv")
patterns = df['Questions'].tolist()
labels = df['Answers'].tolist()

# Preprocessing
def preprocess_text(text):
    # تأكد من تحويل النص لأي شيء فارغ إلى string لتجنب أي خطأ
    if not isinstance(text, str):
        text = str(text)
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = text.strip()
    words = [w for w in text.split() if w not in ENGLISH_STOP_WORDS]
    text = " ".join(words)
    tokens = [lemmatizer.lemmatize(w) for w in text.split()]
    return " ".join(tokens)

patterns = [preprocess_text(p) for p in patterns]

# TF-IDF + KNN
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(patterns)

le = LabelEncoder()
y = le.fit_transform(labels)

knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X, y)

# Chatbot Function
def get_bot_response(user_input):
    user_input_lower = user_input.lower()
    for key in greetings:
        if key in user_input_lower:
            return greetings[key]

    # لو مش تحية، استخدم KNN
    user_input = preprocess_text(user_input)
    user_vec = vectorizer.transform([user_input])
    dist, _ = knn.kneighbors(user_vec, n_neighbors=1)
    threshold = 0.7
    if dist[0][0] > threshold:
        return "I'm not sure how to respond to that. Can you rephrase?"
    predicted_label = knn.predict(user_vec)
    answer = le.inverse_transform(predicted_label)[0]
    return answer

# Streamlit Interface
st.set_page_config(page_title="Mental Health Chatbot", page_icon="💬", layout="wide")
st.title(" Mental Health Chatbot")

if "messages" not in st.session_state:
    st.session_state.messages = []

# Clear Chat button
if st.button(" Clear Chat"):
    st.session_state.messages = []

# Chat Input (live)
user_input = st.chat_input("Type your message here...")

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    bot_response = get_bot_response(user_input)
    st.session_state.messages.append({"role": "assistant", "content": bot_response})

# عرض المحادثة
for msg in st.session_state.messages:
    if msg["role"] == "user":
        with st.chat_message("user"):
            st.markdown(msg["content"])
    else:
        with st.chat_message("assistant"):
            st.markdown(msg["content"])
