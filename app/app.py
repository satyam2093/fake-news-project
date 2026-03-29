import streamlit as st
import pickle
import re
import pandas as pd
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# ================== LOAD MODEL ==================
model = pickle.load(open("models/model.pkl", "rb"))
vectorizer = pickle.load(open("models/vectorizer.pkl", "rb"))

# ================== CLEAN FUNCTION ==================
ps = PorterStemmer()
stop_words = set(stopwords.words('english'))

def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    words = text.split()
    words = [ps.stem(word) for word in words if word not in stop_words]
    return " ".join(words)

# ================== STYLE ==================
st.markdown("""
<style>
body {
    background: linear-gradient(to right, #eef2f3, #dfe9f3);
}
.title {
    text-align: center;
    font-size: 45px;
    font-weight: bold;
    color: #1f77b4;
}
.subtitle {
    text-align: center;
    font-size: 20px;
    color: #555;
    margin-bottom: 30px;
}
.card {
    background-color: white;
    padding: 25px;
    border-radius: 15px;
    box-shadow: 0px 4px 15px rgba(0,0,0,0.1);
    margin-bottom: 20px;
}
.stTextArea textarea {
    border-radius: 10px;
    border: 2px solid #1f77b4;
    padding: 10px;
}
.stButton>button {
    background-color: #1f77b4;
    color: white;
    font-size: 18px;
    border-radius: 10px;
    padding: 10px;
    font-weight: bold;
    width: 100%;
}
.stButton>button:hover {
    background-color: #145a86;
}
</style>
""", unsafe_allow_html=True)

# ================== SIDEBAR ==================
st.sidebar.title("📌 Navigation")

page = st.sidebar.radio(
    "Go to",
    ["🔍 Prediction", "📊 Data Analysis"]
)

st.sidebar.markdown("---")
st.sidebar.title("ℹ️ About")
st.sidebar.write("""
Fake News Detection App

Built using:
- NLP (TF-IDF)
- Logistic Regression
- Streamlit
""")

# ================== PAGE 1: PREDICTION ==================
if page == "🔍 Prediction":

    st.markdown('<div class="title">📰 Fake News Detection</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">AI-powered system to detect Fake vs Real News</div>', unsafe_allow_html=True)

    st.markdown('<div class="card">', unsafe_allow_html=True)
    user_input = st.text_area("Enter News Content", height=200, key="news_input")
    st.markdown('</div>', unsafe_allow_html=True)

    col1, col2, col3 = st.columns([1,2,1])
    with col2:
        analyze = st.button("🔍 Analyze News")

    if analyze:
        if user_input.strip() == "":
            st.warning("⚠️ Please enter some text")
        else:
            cleaned = clean_text(user_input)
            vector = vectorizer.transform([cleaned])

            prediction = model.predict(vector)
            probability = model.predict_proba(vector)

            confidence = max(probability[0]) * 100

            st.markdown('<div class="card">', unsafe_allow_html=True)

            if prediction[0] == 0:
                st.error(f"❌ Fake News ({confidence:.2f}% confidence)")
            else:
                st.success(f"✅ Real News ({confidence:.2f}% confidence)")

            st.markdown('</div>', unsafe_allow_html=True)

# ================== PAGE 2: DATA ANALYSIS ==================
elif page == "📊 Data Analysis":

    st.title("📊 Data Analysis Dashboard")

    fake_df = pd.read_csv("../data/Fake.csv")
    true_df = pd.read_csv("../data/True.csv")

    fake_df['label'] = 0
    true_df['label'] = 1

    df = pd.concat([fake_df, true_df])

    st.subheader("Fake vs Real Distribution")

    fig, ax = plt.subplots()
    df['label'].value_counts().plot(kind='bar', ax=ax)

    ax.set_title("Fake vs Real News")
    ax.set_xlabel("Label (0 = Fake, 1 = Real)")
    ax.set_ylabel("Count")

    st.pyplot(fig)


    #Subject chart

    fig2, ax2 = plt.subplots()

 
    df['subject'].value_counts().head(10).plot(kind='bar', ax=ax2)

    ax2.set_title("Top Subjects")

    st.pyplot(fig2)

    #article length chart

    df['length'] = df['text'].apply(len)

    fig3, ax3 = plt.subplots()

    df['length'].hist(ax=ax3, bins=50)

    ax3.set_title("Article Length Distribution")

    st.pyplot(fig3)

    