 
#  Smart Spam Detector + LIME Explanation


import streamlit as st
import joblib
import string
import re
from scipy.sparse import hstack

import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

from lime.lime_text import LimeTextExplainer

 
# Load Model
 
model = joblib.load("spam_model_sms.pkl")
vectorizer = joblib.load("spam_vectorizer_sms.pkl")

 
# NLTK Setup
 
nltk.download('stopwords')

ps = PorterStemmer()
stop_words = set(stopwords.words('english'))

 
# Text Cleaning (same as training)
 
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'http\S+|www\S+', '', text)
    text = re.sub(r'\d+', '', text)
    text = ''.join([c for c in text if c not in string.punctuation])
    words = text.split()
    words = [ps.stem(w) for w in words if w not in stop_words]
    return " ".join(words)

 
# Feature Functions
 
def has_url(text):
    return int(bool(re.search(r'http|www|\.com|\.in', text)))

def message_length(text):
    return len(text)

def count_digits(text):
    return sum(c.isdigit() for c in text)

def count_uppercase(text):
    return sum(1 for c in text if c.isupper())

def exclamation_count(text):
    return text.count('!')

 
# LIME Prediction Function
 
def predict_proba(texts):
    cleaned = [clean_text(t) for t in texts]
    vectors = vectorizer.transform(cleaned)

    url_features = [[has_url(t)] for t in texts]
    len_features = [[message_length(t)] for t in texts]
    digit_features = [[count_digits(t)] for t in texts]
    upper_features = [[count_uppercase(t)] for t in texts]
    exclaim_features = [[exclamation_count(t)] for t in texts]

    final_input = hstack([
        vectors,
        url_features,
        len_features,
        digit_features,
        upper_features,
        exclaim_features
    ])

    return model.predict_proba(final_input)

 
# Streamlit UI
 
st.set_page_config(page_title="Smart Spam Detector", layout="centered")

st.title("📩 Smart Spam Detector (with Explainable AI)")
st.markdown("🚀 High-accuracy spam detection + LIME explanations")

message = st.text_area("✉️ Enter your message")

if st.button("Predict"):

    if message.strip() == "":
        st.warning("⚠️ Please enter a message")
    else:
        # Clean text
        cleaned = clean_text(message)

        # TF-IDF
        vectorized = vectorizer.transform([cleaned])

        # Features
        url_feature = has_url(message)
        len_feature = message_length(message)
        digit_feature = count_digits(message)
        upper_feature = count_uppercase(message)
        exclaim_feature = exclamation_count(message)

        final_input = hstack([
            vectorized,
            [[url_feature]],
            [[len_feature]],
            [[digit_feature]],
            [[upper_feature]],
            [[exclaim_feature]]
        ])

        # Prediction
        prediction = model.predict(final_input)[0]
        proba = model.predict_proba(final_input)[0]

        if prediction == 1:
            st.error("SPAM MESSAGE")
            st.write(f"Confidence: {proba[1]*100:.2f}%")
        else:
            st.success("NOT SPAM")
            st.write(f"Confidence: {proba[0]*100:.2f}%")

         
        #  LIME Explanation
         
        st.subheader(" Why this prediction?")

        explainer = LimeTextExplainer(class_names=["Not Spam", "Spam"])

        explanation = explainer.explain_instance(
            message,
            predict_proba,
            num_features=8
        )

        st.components.v1.html(explanation.as_html(), height=500)

        # Optional debug
        with st.expander("⚙️ Feature Values"):
            st.write({
                "URL": url_feature,
                "Length": len_feature,
                "Digits": digit_feature,
                "Uppercase": upper_feature,
                "Exclamations": exclaim_feature
            })













# # ============================================
# # 📩 Smart Spam Detector (Optimized Version)
# # ============================================

# import streamlit as st
# import joblib
# import string
# import re
# from scipy.sparse import hstack

# import nltk
# from nltk.corpus import stopwords
# from nltk.stem import PorterStemmer

# # -------------------------------
# # Load Model
# # -------------------------------
# model = joblib.load("spam_model_sms.pkl")
# vectorizer = joblib.load("spam_vectorizer_sms.pkl")

# # -------------------------------
# # NLTK Setup
# # -------------------------------
# nltk.download('stopwords')

# ps = PorterStemmer()
# stop_words = set(stopwords.words('english'))

# # -------------------------------
# # Text Cleaning (same as training)
# # -------------------------------
# def clean_text(text):
#     text = str(text).lower()
#     text = re.sub(r'http\S+|www\S+', '', text)
#     text = re.sub(r'\d+', '', text)
#     text = ''.join([c for c in text if c not in string.punctuation])
#     words = text.split()
#     words = [ps.stem(w) for w in words if w not in stop_words]
#     return " ".join(words)

# # -------------------------------
# # Feature Functions
# # -------------------------------
# def has_url(text):
#     return int(bool(re.search(r'http|www|\.com|\.in', text)))

# def message_length(text):
#     return len(text)

# def count_digits(text):
#     return sum(c.isdigit() for c in text)

# def count_uppercase(text):
#     return sum(1 for c in text if c.isupper())

# def exclamation_count(text):
#     return text.count('!')

# # -------------------------------
# # Streamlit UI
# # -------------------------------
# st.set_page_config(page_title="Smart Spam Detector", layout="centered")

# st.title("📩 Smart Spam Detector (Optimized)")
# st.markdown("🚀 High-accuracy spam detection using advanced ML")

# message = st.text_area("✉️ Enter your message")

# if st.button("Predict"):

#     if message.strip() == "":
#         st.warning("⚠️ Please enter a message")
#     else:
#         # Clean text
#         cleaned = clean_text(message)

#         # TF-IDF
#         vectorized = vectorizer.transform([cleaned])

#         # Extra features (IMPORTANT: use original message)
#         url_feature = has_url(message)
#         len_feature = message_length(message)
#         digit_feature = count_digits(message)
#         upper_feature = count_uppercase(message)
#         exclaim_feature = exclamation_count(message)

#         # Combine features
#         final_input = hstack([
#             vectorized,
#             [[url_feature]],
#             [[len_feature]],
#             [[digit_feature]],
#             [[upper_feature]],
#             [[exclaim_feature]]
#         ])

#         # Prediction
#         prediction = model.predict(final_input)[0]
#         proba = model.predict_proba(final_input)[0]

#         # Output
#         if prediction == 1:
#             st.error("🚨 SPAM MESSAGE")
#             st.write(f"Confidence: {proba[1]*100:.2f}%")
#         else:
#             st.success("✅ NOT SPAM")
#             st.write(f"Confidence: {proba[0]*100:.2f}%")

#         # Debug info (optional for viva/demo)
#         with st.expander("🔍 Feature Details"):
#             st.write({
#                 "URL Detected": url_feature,
#                 "Message Length": len_feature,
#                 "Digits Count": digit_feature,
#                 "Uppercase Count": upper_feature,
#                 "Exclamation Count": exclaim_feature
#             })

