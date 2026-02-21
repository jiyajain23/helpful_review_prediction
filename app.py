import streamlit as st
import pickle
import re
from nltk.corpus import stopwords
from scipy.sparse import hstack
import numpy as np

# --- 1. Load the Saved Model & Vectorizer ---
@st.cache_resource # This keeps the model in memory so it stays fast
def load_assets():
    with open('helpful_model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('tfidf_vectorizer.pkl', 'rb') as f:
        vectorizer = pickle.load(f)
    return model, vectorizer

model, tfidf = load_assets()

# --- 2. The Text Cleaning Function (Must match your training logic) ---
def clean_review(text):
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    words = text.lower().split()
    return " ".join(words)

# --- 3. Streamlit UI ---
st.title("🛒 Review Helpfulness Predictor")
st.write("Type a product review below to see if our AI thinks it's helpful!")

user_input = st.text_area("Enter review text here:", "This coffee is delicious and has a great aroma. I've been buying it for 2 years!")

if st.button("Predict"):
    # Preprocess
    clean_text = clean_review(user_input)
    text_vector = tfidf.transform([clean_text])
    length = np.array([[len(user_input)]])
    
    # Combine and Predict
    final_features = hstack([text_vector, length])
    prediction = model.predict(final_features)
    
    # Display Result
    if prediction[0] == 1:
        st.success("✅ This review is likely to be HELPFUL to others.")
    else:
        st.error("❌ This review is likely NOT HELPFUL (too generic or lacks detail).")