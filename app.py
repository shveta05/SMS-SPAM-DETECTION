from flask import Flask, render_template, url_for, request
import pandas as pd
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score

import joblib


app = Flask(__name__)


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    df = pd.read_csv("spam.csv", encoding="latin-1")
    df.drop(["Unnamed: 2", "Unnamed: 3", "Unnamed: 4"], axis=1, inplace=True)
    df["label"] = df["v1"].map({"ham": 0, "spam": 1})
    X = df["v2"]  
    y = df["label"]

    cv = CountVectorizer()
    X = cv.fit_transform(X)   
    from sklearn.model_selection import train_test_split

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.33, random_state=42
    )

    from sklearn.naive_bayes import MultinomialNB
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.svm import SVC
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.tree import DecisionTreeClassifier

    mnb = MultinomialNB()
    knc = KNeighborsClassifier()
    svc = SVC(kernel="sigmoid", gamma=1.0)
    rfc = RandomForestClassifier(n_estimators=50, random_state=2)
    dtc = DecisionTreeClassifier(max_depth=5)

    knc.fit(X_train, y_train)
    clfs = {"SVC": svc, "KN": knc, "NB": mnb, "RF": rfc, "DT": dtc}

    def train_classifier(clf, X_train, y_train, X_test, y_test):
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        return accuracy, precision

    accuracy_scores = []
    precision_scores = []
    for name, clf in clfs.items():
        current_accuracy, current_precision = train_classifier(
            clf, X_train, y_train, X_test, y_test
        )
        print("For ", name)
        print("Accuracy - ", current_accuracy)
        print("Precision - ", current_precision)

        accuracy_scores.append(current_accuracy)
        precision_scores.append(current_precision)

    performance_df = pd.DataFrame(
        {
            "Algorithm": clfs.keys(),
            "Accuracy": accuracy_scores,
            "Precision": precision_scores,
        }
    ).sort_values("Precision", ascending=False)

    svc = SVC(kernel="sigmoid", gamma=1.0, probability=True)
    mnb = MultinomialNB()
    knc = KNeighborsClassifier()
    rfc = RandomForestClassifier(n_estimators=50, random_state=2)
    dtc = DecisionTreeClassifier(max_depth=5)
    from sklearn.ensemble import VotingClassifier

    voting = VotingClassifier(
        estimators=[("nb", mnb), ("kn", knc), ("rf", rfc)],
        voting="soft",
    )
    voting.fit(X_train, y_train)

    y_pred = voting.predict(X_test)

    estimators = [("nb", mnb), ("kn", knc), ("rf", rfc)]
    final_estimator = RandomForestClassifier()
    from sklearn.ensemble import StackingClassifier

    clf = StackingClassifier(estimators=estimators, final_estimator=final_estimator)

    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    # clf.score(X_test,y_test)
    # joblib.dump(clf, 'NB_spam_model.pkl')
    # NB_spam_model = open('NB_spam_model.pkl','rb')
    # clf = joblib.load(NB_spam_model)

    if request.method == "POST":
        message = request.form["message"]
        data = [message]
        print(data)
        vect = cv.transform(data).toarray()
        print(vect)
        my_prediction = clf.predict(vect)
    return render_template("result.html", prediction=my_prediction, message=message)


if __name__ == "__main__":
    app.run(debug=True)
