import argparse
import json
from pathlib import Path

import joblib
import pandas as pd
import sklearn
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC

from preprocessing import preprocess_text


PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_DATASET = PROJECT_DIR / "Aditya_University_Sentiment_Final_Dataset.xlsx"
MODEL_PATH = PROJECT_DIR / "sentiment_model.pkl"
VECTORIZER_PATH = PROJECT_DIR / "tfidf_vectorizer.pkl"
METADATA_PATH = PROJECT_DIR / "sentiment_model_metadata.json"
LABELS = {"negative", "neutral", "positive"}
SAMPLE_COMMENTS = [
    "The faculty are helpful and the teaching is excellent.",
    "The teaching is poor and the classrooms are dirty.",
    "The college library is located in the main building.",
]


def load_dataset(path):
    if path.suffix.lower() == ".csv":
        frame = pd.read_csv(path)
    elif path.suffix.lower() in {".xlsx", ".xls"}:
        frame = pd.read_excel(path)
    else:
        raise ValueError("Dataset must be a .csv, .xls, or .xlsx file.")

    missing_columns = {"Comment", "Sentiment"} - set(frame.columns)
    if missing_columns:
        raise ValueError(
            "Dataset must contain Comment and Sentiment columns; "
            f"missing: {', '.join(sorted(missing_columns))}"
        )

    frame = frame.dropna(subset=["Comment", "Sentiment"]).copy()
    frame["Comment"] = frame["Comment"].astype(str)
    frame["Sentiment"] = frame["Sentiment"].astype(str).str.strip().str.lower()
    frame = frame[frame["Comment"].str.strip().ne("")]
    frame = frame[frame["Sentiment"].ne("")]

    unexpected_labels = set(frame["Sentiment"]) - LABELS
    if unexpected_labels:
        raise ValueError(
            "Unexpected sentiment labels: "
            + ", ".join(sorted(unexpected_labels))
            + ". Expected negative, neutral, and positive."
        )

    frame["Cleaned_Comment"] = frame["Comment"].map(preprocess_text)
    frame = frame[frame["Cleaned_Comment"].ne("")]
    return frame


def make_vectorizer():
    return TfidfVectorizer(
        max_features=5000,
        ngram_range=(1, 2),
        min_df=1,
        max_df=0.95,
    )


def evaluate_models(texts, labels):
    x_train, x_test, y_train, y_test = train_test_split(
        texts,
        labels,
        test_size=0.20,
        random_state=42,
        stratify=labels,
    )
    vectorizer = make_vectorizer()
    x_train_tfidf = vectorizer.fit_transform(x_train)
    x_test_tfidf = vectorizer.transform(x_test)

    logistic = LogisticRegression(max_iter=2000, random_state=42)
    svm = LinearSVC(random_state=42)
    random_forest = RandomForestClassifier(
        n_estimators=300,
        random_state=42,
        class_weight="balanced",
    )
    voting = VotingClassifier(
        estimators=[
            ("lr", logistic),
            ("svm", svm),
            ("rf", random_forest),
        ],
        voting="hard",
    )

    models = {
        "Logistic Regression": logistic,
        "SVM": svm,
        "Random Forest": random_forest,
        "Ensemble Voting": voting,
    }
    results = {}
    for name, model in models.items():
        model.fit(x_train_tfidf, y_train)
        predictions = model.predict(x_test_tfidf)
        results[name] = {
            "accuracy": float(accuracy_score(y_test, predictions)),
            "weighted_f1": float(
                f1_score(y_test, predictions, average="weighted", zero_division=0)
            ),
        }
        print(
            f"{name}: accuracy={results[name]['accuracy']:.4f}, "
            f"weighted_f1={results[name]['weighted_f1']:.4f}"
        )

    print("\nLogistic Regression holdout report:")
    print(classification_report(y_test, logistic.predict(x_test_tfidf), zero_division=0))
    return results


def train_and_save(frame, results, dataset_path):
    texts = frame["Cleaned_Comment"]
    labels = frame["Sentiment"]

    vectorizer = make_vectorizer()
    training_matrix = vectorizer.fit_transform(texts)
    model = LogisticRegression(max_iter=2000, random_state=42)
    model.fit(training_matrix, labels)

    joblib.dump(model, MODEL_PATH)
    joblib.dump(vectorizer, VECTORIZER_PATH)

    metadata = {
        "model": "LogisticRegression",
        "labels": sorted(str(label) for label in model.classes_),
        "feature_count": int(training_matrix.shape[1]),
        "training_rows": int(len(frame)),
        "class_counts": {
            str(label): int(count)
            for label, count in labels.value_counts().sort_index().items()
        },
        "dataset": dataset_path.name,
        "sklearn_version": sklearn.__version__,
        "evaluation": results,
    }
    METADATA_PATH.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return model, vectorizer


def main():
    parser = argparse.ArgumentParser(
        description="Train and save the student sentiment classifier."
    )
    parser.add_argument(
        "dataset",
        nargs="?",
        type=Path,
        default=DEFAULT_DATASET,
        help="CSV or Excel dataset with Comment and Sentiment columns.",
    )
    args = parser.parse_args()
    dataset_path = args.dataset.expanduser().resolve()
    if not dataset_path.is_file():
        raise FileNotFoundError(
            f"Dataset not found: {dataset_path}. Pass its path as an argument."
        )

    frame = load_dataset(dataset_path)
    class_counts = frame["Sentiment"].value_counts().sort_index()
    if set(class_counts.index) != LABELS or class_counts.min() < 2:
        raise ValueError(
            "Training data must contain at least two examples of each "
            f"sentiment; found {class_counts.to_dict()}."
        )

    print(f"Usable comments: {len(frame)}")
    print("Sentiment distribution:")
    print(class_counts.to_string())

    results = evaluate_models(frame["Cleaned_Comment"], frame["Sentiment"])
    model, vectorizer = train_and_save(frame, results, dataset_path)
    print(
        f"\nSaved Logistic Regression and matching TF-IDF vectorizer "
        f"({len(vectorizer.vocabulary_)} features) using scikit-learn "
        f"{sklearn.__version__}."
    )
    print("\nPredictions from the saved model:")
    sample_matrix = vectorizer.transform(
        [preprocess_text(comment) for comment in SAMPLE_COMMENTS]
    )
    probabilities = model.predict_proba(sample_matrix)
    for comment, prediction, scores in zip(
        SAMPLE_COMMENTS, model.predict(sample_matrix), probabilities
    ):
        confidence = {
            label: round(
                float(scores[list(model.classes_).index(label.lower())]) * 100, 2
            )
            for label in ("Positive", "Negative", "Neutral")
        }
        print(f"\n{comment}\nPrediction: {str(prediction).capitalize()}")
        print(f"Probabilities: {confidence}")


if __name__ == "__main__":
    main()
