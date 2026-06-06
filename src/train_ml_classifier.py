from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import MultiLabelBinarizer


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
MODEL_DIR = ROOT / "models"
MODEL_PATH = MODEL_DIR / "ml_classifier.joblib"

DEFAULT_THRESHOLDS = {
    "safety_signal": 0.35,
    "content_request": 0.55,
    "criticism": 0.60,
    "praise": 0.65,
    "bug_or_access_problem": 0.60,
    "metadata_issue": 0.60,
    "unclear": 0.50,
    "other": 0.50,
}


def load_training_data(
    feedback_path: str | Path = DATA_DIR / "feedback_messages.csv",
    labels_path: str | Path = DATA_DIR / "expected_labels.csv",
) -> pd.DataFrame:
    feedback = pd.read_csv(feedback_path)
    labels = pd.read_csv(labels_path)
    if "expected_labels" not in labels.columns and "expected_primary_label" in labels.columns:
        labels["expected_labels"] = labels["expected_primary_label"]
    if "expected_labels" not in labels.columns:
        raise ValueError("expected_labels or expected_primary_label is required.")

    merged = feedback.merge(labels, on="message_id", how="inner")
    if len(merged) != len(feedback):
        raise ValueError("Every feedback row must have an expected-label row.")
    if merged["expected_labels"].isna().any():
        raise ValueError("Every training row must have expected labels.")

    merged["label_list"] = merged["expected_labels"].apply(_parse_labels)
    if merged["label_list"].apply(len).eq(0).any():
        raise ValueError("Every training row must have at least one parsed label.")
    merged["expected_safety_flag"] = merged["expected_safety_flag"].apply(_parse_bool)
    return merged


def train_ml_classifier(
    feedback_path: str | Path = DATA_DIR / "feedback_messages.csv",
    labels_path: str | Path = DATA_DIR / "expected_labels.csv",
    output_path: str | Path = MODEL_PATH,
) -> dict[str, Any]:
    training_data = load_training_data(feedback_path, labels_path)
    artifact = fit_ml_classifier(
        training_data,
        source_files={
            "feedback_messages": str(feedback_path),
            "expected_labels": str(labels_path),
        },
    )

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, output_path)
    return artifact


def fit_ml_classifier(
    training_data: pd.DataFrame, source_files: dict[str, str] | None = None
) -> dict[str, Any]:
    texts = training_data["user_message"].astype(str).tolist()

    vectorizer = TfidfVectorizer(
        lowercase=True,
        ngram_range=(1, 2),
        min_df=1,
        max_features=10000,
    )
    features = vectorizer.fit_transform(texts)

    label_binarizer = MultiLabelBinarizer()
    label_targets = label_binarizer.fit_transform(training_data["label_list"])
    multilabel_classifier = OneVsRestClassifier(
        LogisticRegression(max_iter=1000, class_weight="balanced")
    )
    multilabel_classifier.fit(features, label_targets)

    safety_classifier = LogisticRegression(max_iter=1000, class_weight="balanced")
    safety_classifier.fit(features, training_data["expected_safety_flag"].astype(int))

    return {
        "vectorizer": vectorizer,
        "multilabel_classifier": multilabel_classifier,
        "label_binarizer": label_binarizer,
        "safety_classifier": safety_classifier,
        "label_list": list(label_binarizer.classes_),
        "thresholds": DEFAULT_THRESHOLDS,
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "training_size": len(training_data),
        "source_files": source_files or {},
    }


def _parse_labels(value: str) -> list[str]:
    return [label.strip() for label in str(value).split(";") if label.strip()]


def _parse_bool(value) -> bool:
    return str(value).strip().casefold() in {"true", "1", "yes", "y"}


def main() -> None:
    artifact = train_ml_classifier()
    print(
        "Trained ML classifier on "
        f"{artifact['training_size']} synthetic feedback examples. "
        f"Wrote artifact to {MODEL_PATH}."
    )
    if artifact["training_size"] < 200:
        print(
            "Warning: training dataset is small and synthetic. "
            "Use ML predictions conservatively and rely on hybrid fallback."
        )


if __name__ == "__main__":
    main()
