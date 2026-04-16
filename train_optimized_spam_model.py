 
#   SMS Spam Detection - Optimized Training
 

import pandas as pd
import numpy as np
import string
import re
import joblib

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, precision_score, f1_score, confusion_matrix
from sklearn.linear_model import LogisticRegression
from scipy.sparse import hstack

import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

 
# Download NLTK data (first run only)
 
nltk.download('stopwords')

 
# 1. Load dataset (SMS Spam)
 
df = pd.read_csv("spam.csv", encoding="latin-1")

# Keep only required columns
df = df[['v1', 'v2']]
df.columns = ['label', 'text']

# Encode labels
df['label'] = df['label'].map({'ham': 0, 'spam': 1})

# Remove duplicates & missing
df.drop_duplicates(inplace=True)
df.dropna(inplace=True)

# Shuffle data
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

 
# 2. Advanced Text Cleaning
 
ps = PorterStemmer()
stop_words = set(stopwords.words('english'))

def clean_text(text):
    text = str(text).lower()

    # Remove URLs
    text = re.sub(r'http\S+|www\S+', '', text)

    # Remove numbers
    text = re.sub(r'\d+', '', text)

    # Remove punctuation
    text = ''.join([c for c in text if c not in string.punctuation])

    # Tokenize
    words = text.split()

    # Remove stopwords + stemming
    words = [ps.stem(w) for w in words if w not in stop_words]

    return " ".join(words)

df['text'] = df['text'].apply(clean_text)

 
# 3. Feature Engineering
 
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

 
# 4. Train-Test Split
 
X_train, X_test, y_train, y_test = train_test_split(
    df['text'],
    df['label'],
    test_size=0.2,
    random_state=42,
    stratify=df['label']
)

 
# 5. TF-IDF Vectorization

vectorizer = TfidfVectorizer(
    ngram_range=(1,2),      # uni + bi-grams
    max_features=5000,
    min_df=2,
    max_df=0.9,
    sublinear_tf=True
)

X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

 
# 6. Extra Features
 
url_train = X_train.apply(has_url).values.reshape(-1,1)
url_test = X_test.apply(has_url).values.reshape(-1,1)

len_train = X_train.apply(message_length).values.reshape(-1,1)
len_test = X_test.apply(message_length).values.reshape(-1,1)

digit_train = X_train.apply(count_digits).values.reshape(-1,1)
digit_test = X_test.apply(count_digits).values.reshape(-1,1)

upper_train = X_train.apply(count_uppercase).values.reshape(-1,1)
upper_test = X_test.apply(count_uppercase).values.reshape(-1,1)

exclaim_train = X_train.apply(exclamation_count).values.reshape(-1,1)
exclaim_test = X_test.apply(exclamation_count).values.reshape(-1,1)

 
# 7. Combine Features
 
X_train_final = hstack([
    X_train_vec,
    url_train,
    len_train,
    digit_train,
    upper_train,
    exclaim_train
])

X_test_final = hstack([
    X_test_vec,
    url_test,
    len_test,
    digit_test,
    upper_test,
    exclaim_test
])

 
# 8. Model Training (Logistic Regression)
 
model = LogisticRegression(max_iter=1000)
model.fit(X_train_final, y_train)

 
# 9. Cross Validation
 
cv_scores = cross_val_score(model, X_train_final, y_train, cv=5)

 
# 10. Evaluation
 
y_pred = model.predict(X_test_final)

print("\n Evaluation Results")
print("---------------------------")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred))
print("F1 Score:", f1_score(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

print("\n Cross-validation Accuracy:", cv_scores.mean())

 
# 11. Save Model
 
joblib.dump(model, "spam_model_sms.pkl")
joblib.dump(vectorizer, "spam_vectorizer_sms.pkl")

print("\n Model trained and saved successfully!")