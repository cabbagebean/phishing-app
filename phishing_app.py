import streamlit as st
import pandas as pd
import joblib
import os
import re
import numpy as np
import nltk
import gdown

nltk.download("vader_lexicon")
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# Download model from Google Drive if not already present
model_path = "phishing_detection_random_tuned.joblib"
file_id = "1UbPPC3XoxMuOeHp0Rfa4QVCNJL0FUpr-"  # Replace with your actual file ID
url = f"https://drive.google.com/uc?id={file_id}"

if not os.path.exists(model_path):
    with st.spinner("Downloading model file..."):
        gdown.download(url, model_path, quiet=False)

# Load model
model = joblib.load(model_path)

# Phishing-related keywords list
phishing_keywords = ["verify", "login", "password", "urgent", "account", "update", "click", "security"]

# Initialize sentiment analyzer
sia = SentimentIntensityAnalyzer()

# App UI
st.title("📧 Phishing Email Detection")
st.subheader("Paste your email content below:")

email_text = st.text_area("Email Text", height=200)
sender_address = st.text_input("Sender Email Address")

# Feature extraction
def extract_features(email_text, sender_address):
    sender_length = len(sender_address)
    sentiment_score = sia.polarity_scores(email_text)["compound"]
    keyword_count = sum(1 for word in phishing_keywords if word in email_text.lower())
    contains_keywords = int(keyword_count > 0)
    is_free_email = int(any(domain in sender_address.lower() for domain in ["gmail.com", "yahoo.com", "outlook.com", "hotmail.com"]))
    is_disposable_email = int(any(domain in sender_address.lower() for domain in ["tempmail", "10minutemail", "mailinator"]))
    has_suspicious_chars = int(bool(re.search(r"[!#$%^&*()=+{}[\]|\\;:'\",<>/?]", email_text)))
    url_count = len(re.findall(r"http[s]?://", email_text))
    url_x_keyword = url_count * keyword_count

    return pd.DataFrame([{
        "email_text": email_text,
        "Log_Sender_Length": sender_length,
        "Sentiment_Score": sentiment_score,
        "Phishing_Keyword_Count": keyword_count,
        "Contains_Phish_Keywords": contains_keywords,
        "Is_Free_Email": is_free_email,
        "Is_Disposable_Email": is_disposable_email,
        "Has_Suspicious_Chars": has_suspicious_chars,
        "URL_x_Keyword": url_x_keyword
    }])

# Predict
if st.button("Detect Phishing"):
    if email_text.strip() and sender_address.strip():
        features_df = extract_features(email_text, sender_address)
        prediction = model.predict(features_df)[0]
        probabilities = model.predict_proba(features_df)[0]
        confidence = round(np.max(probabilities) * 100, 1)
        predicted_label = "Phishing" if prediction == 1 else "Legitimate"
        label_display = "🚨 Phishing Email Detected!" if prediction == 1 else "✅ Legitimate Email"
        st.success(label_display)
        st.markdown(f"**Model Confidence:** {confidence}%")
        st.markdown(f"The model is **{confidence}% confident** this email is **{predicted_label}**.")
    else:
        st.error("Please provide both the email text and sender email address.")
