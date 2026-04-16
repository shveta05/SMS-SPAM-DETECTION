# SMS Spam Detection

A machine learning project that classifies SMS messages as spam or not spam using natural language processing techniques.

## Features

* TF-IDF text vectorization
* Logistic Regression model
* Feature engineering (URL, length, digits, uppercase, exclamations)
* Streamlit web application
* LIME for model explanation

## Tech Stack

* Python
* Scikit-learn
* Streamlit
* LIME

## Run the App

```bash
pip install -r requirements.txt
streamlit run app_streamlit_optimized.py
```

## Output

* Predicts spam or not spam
* Displays confidence score
* Provides explanation using LIME
