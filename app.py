from flask import Flask, render_template, request
import json
import math
from pathlib import Path

import joblib

from preprocessing import preprocess_text

app = Flask(__name__)

PROJECT_DIR = Path(__file__).resolve().parent
MODEL_PATH = PROJECT_DIR / "sentiment_model.pkl"
VECTORIZER_PATH = PROJECT_DIR / "tfidf_vectorizer.pkl"
METADATA_PATH = PROJECT_DIR / "sentiment_model_metadata.json"

if not METADATA_PATH.is_file():
    raise RuntimeError(
        "Trained model metadata is missing. Run "
        "'python train_model.py' before starting the app."
    )

with METADATA_PATH.open(encoding="utf-8") as file:
    model_metadata = json.load(file)

model = joblib.load(MODEL_PATH)
tfidf = joblib.load(VECTORIZER_PATH)

expected_labels = {"negative", "neutral", "positive"}
loaded_labels = {str(label).strip().lower() for label in model.classes_}
if loaded_labels != expected_labels:
    raise RuntimeError(
        "The saved model does not contain all three expected sentiment labels. "
        "Run 'python train_model.py' to retrain it."
    )

if (
    model.n_features_in_ != len(tfidf.vocabulary_)
    or model_metadata.get("feature_count") != model.n_features_in_
    or sorted(loaded_labels) != model_metadata.get("labels")
):
    raise RuntimeError(
        "The saved model, TF-IDF vectorizer, and metadata do not match. "
        "Run 'python train_model.py' to regenerate the artifacts together."
    )


def _softmax(scores):
    max_score = max(scores)
    exp_scores = [math.exp(score - max_score) for score in scores]
    total = sum(exp_scores)
    return [score / total for score in exp_scores]


def get_sentiment_percentages(text_vector):
    classes = list(model.classes_)

    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(text_vector)[0]
    elif hasattr(model, "decision_function"):
        scores = model.decision_function(text_vector)
        scores = scores[0] if getattr(scores, "ndim", 1) > 1 else scores

        if len(classes) == 2 and not hasattr(scores, "__len__"):
            positive_probability = 1 / (1 + math.exp(-float(scores)))
            probabilities = [1 - positive_probability, positive_probability]
        else:
            probabilities = _softmax([float(score) for score in scores])
    else:
        predicted_class = model.predict(text_vector)[0]
        probabilities = [
            1.0 if sentiment_class == predicted_class else 0.0
            for sentiment_class in classes
        ]

    raw_percentages = {
        str(sentiment_class).lower(): round(float(probability) * 100, 2)
        for sentiment_class, probability in zip(classes, probabilities)
    }

    return {
        "positive": raw_percentages.get("positive", 0),
        "negative": raw_percentages.get("negative", 0),
        "neutral": raw_percentages.get("neutral", 0),
    }


@app.route("/", methods=["GET", "POST"])
def home():

    prediction = None
    percentages = None
    comment = ""

    if request.method == "POST":

        print("POST request received!")

        comment = request.form.get("comment", "")

        print("Comment:", comment)

        if comment.strip():

            cleaned_text = preprocess_text(comment)

            print("Cleaned text:", cleaned_text)

            text_vector = tfidf.transform([cleaned_text])

            prediction = model.predict(text_vector)[0]
            percentages = get_sentiment_percentages(text_vector)

            print("Prediction:", prediction)
            print("Percentages:", percentages)

    return render_template(
        "index.html",
        prediction=prediction,
        percentages=percentages,
        comment=comment
    )


if __name__ == "__main__":
    app.run(debug=True)
