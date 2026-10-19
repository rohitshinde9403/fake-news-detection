"""
Fake News Detection on Social Media
Educational baseline using TF-IDF + Logistic Regression.

Important:
- This is a learning/demo implementation, not a reliable fact-checking system.
- Train with a properly labeled, trustworthy dataset before drawing conclusions.
- The included sample data is tiny and illustrative only; it is not suitable for
  measuring real-world accuracy.
"""
from pathlib import Path
import argparse
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "news.csv"
MODEL_PATH = ROOT / "fake_news_model.joblib"
LABEL_MAP = {"fake": 0, "real": 1}


def load_data(path: Path) -> pd.DataFrame:
    """Load a CSV with required columns: text,label."""
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {path}\n"
            "Add a labeled CSV with columns 'text' and 'label' (fake/real)."
        )
    df = pd.read_csv(path)
    required = {"text", "label"}
    if not required.issubset(df.columns):
        raise ValueError("CSV must contain exactly/at least these columns: text,label")
    df = df[["text", "label"]].dropna()
    df["text"] = df["text"].astype(str).str.strip()
    df["label"] = df["label"].astype(str).str.strip().str.lower()
    df = df[(df["text"] != "") & (df["label"].isin(LABEL_MAP))]
    if df["label"].nunique() < 2:
        raise ValueError("Dataset must contain both labels: fake and real.")
    if len(df) < 10:
        raise ValueError("Please use at least 10 labeled rows; a larger dataset is strongly recommended.")
    return df


def build_model() -> Pipeline:
    """Convert text to TF-IDF features and train a Logistic Regression classifier."""
    return Pipeline([
        ("tfidf", TfidfVectorizer(stop_words="english", max_features=20000)),
        ("classifier", LogisticRegression(max_iter=1000, class_weight="balanced")),
    ])


def train(data_path: Path = DATA_PATH) -> None:
    df = load_data(data_path)
    X_train, X_test, y_train, y_test = train_test_split(
        df["text"], df["label"], test_size=0.25, random_state=42, stratify=df["label"]
    )
    model = build_model()
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    print(f"Rows used: {len(df)}")
    print(f"Accuracy: {accuracy_score(y_test, predictions):.3f}")
    print("\nClassification report:")
    print(classification_report(y_test, predictions, zero_division=0))
    print("Confusion matrix (label order: fake, real):")
    print(confusion_matrix(y_test, predictions, labels=["fake", "real"]))

    joblib.dump(model, MODEL_PATH)
    print(f"\nSaved model to: {MODEL_PATH}")


def predict(text: str) -> None:
    if not MODEL_PATH.exists():
        raise FileNotFoundError("Trained model not found. Run: python fake_news_detection.py --train")
    model = joblib.load(MODEL_PATH)
    prediction = model.predict([text])[0]
    probabilities = model.predict_proba([text])[0]
    classes = list(model.classes_)
    confidence = float(probabilities[classes.index(prediction)])
    print(f"Model label: {prediction.upper()}")
    print(f"Model confidence: {confidence:.1%}")
    print("Note: This is a model prediction, not proof that the claim is true or false.")


def main():
    parser = argparse.ArgumentParser(description="Train or use a basic fake-news text classifier.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--train", action="store_true", help="Train/evaluate model using data/news.csv")
    group.add_argument("--predict", type=str, help="Predict a label for a piece of news text")
    parser.add_argument("--data", type=str, default=str(DATA_PATH), help="Path to labeled CSV dataset")
    args = parser.parse_args()
    if args.train:
        train(Path(args.data))
    else:
        predict(args.predict)


if __name__ == "__main__":
    main()
