
#   Smart Spam Detector (Flask Optimized)
 

from flask import Flask, render_template, request
import joblib
import string
import re
from scipy.sparse import hstack

import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

 
# Load Model
 
model = joblib.load("spam_model_sms.pkl")
vectorizer = joblib.load("spam_vectorizer_sms.pkl")

app = Flask(__name__)

 
# NLTK Setup
 
nltk.download('stopwords')

ps = PorterStemmer()
stop_words = set(stopwords.words('english'))

 
# Clean Text (same as training)
 
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

 
# Routes
 
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    message = request.form['message']

    # Clean text
    cleaned = clean_text(message)

    # TF-IDF
    vectorized = vectorizer.transform([cleaned])

    # Extra features (use ORIGINAL message)
    url_feature = has_url(message)
    len_feature = message_length(message)
    digit_feature = count_digits(message)
    upper_feature = count_uppercase(message)
    exclaim_feature = exclamation_count(message)

    # Combine features
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

    result = "Spam" if prediction == 1 else "Not Spam"
    confidence = proba[1] if prediction == 1 else proba[0]

    return render_template(
        'result.html',
        prediction=result,
        confidence=round(confidence * 100, 2),
        message=message
    )

 
# Run
 
if __name__ == "__main__":
    app.run(debug=True)
 