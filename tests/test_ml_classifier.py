import csv
from pathlib import Path

import pytest

from src import classify as classify_module
from src.classify import (
    MLClassifierModelNotFoundError,
    _classify_feedback_ml_with_artifact,
    classify_feedback_hybrid,
    classify_feedback_ml,
)
from src.schemas import ClassifiedFeedback, FeedbackMessage
from src.train_ml_classifier import train_ml_classifier


class DummyVectorizer:
    def transform(self, texts):
        return texts


class DummyMultilabelClassifier:
    def __init__(self, probabilities):
        self.probabilities = probabilities

    def predict_proba(self, features):
        return [self.probabilities]


class DummyLabelBinarizer:
    classes_ = ["content_request", "criticism", "safety_signal"]


class DummySafetyClassifier:
    classes_ = [0, 1]

    def __init__(self, safety_probability):
        self.safety_probability = safety_probability

    def predict_proba(self, features):
        return [[1.0 - self.safety_probability, self.safety_probability]]


def test_train_ml_classifier_creates_artifact(tmp_path) -> None:
    feedback_path, labels_path = _write_sample_training_data(tmp_path)
    output_path = tmp_path / "ml_classifier.joblib"

    artifact = train_ml_classifier(feedback_path, labels_path, output_path)

    assert output_path.exists()
    assert artifact["training_size"] == 8
    assert "vectorizer" in artifact
    assert "multilabel_classifier" in artifact


def test_classify_feedback_ml_returns_valid_classification(tmp_path, monkeypatch) -> None:
    feedback_path, labels_path = _write_sample_training_data(tmp_path)
    output_path = tmp_path / "ml_classifier.joblib"
    train_ml_classifier(feedback_path, labels_path, output_path)
    monkeypatch.setattr(classify_module, "CLASSIFIER_MODEL_PATH", output_path)

    classified = classify_feedback_ml(
        FeedbackMessage(
            message_id="msg_live",
            user_message="Bitte mehr Knieübungen ohne Geräte.",
        )
    )

    assert isinstance(classified, ClassifiedFeedback)
    assert classified.message_id == "msg_live"
    assert classified.classifier_source in {"ml", "ml_abstain"}


def test_missing_model_raises_helpful_error(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(classify_module, "CLASSIFIER_MODEL_PATH", tmp_path / "missing.joblib")

    with pytest.raises(MLClassifierModelNotFoundError) as exc_info:
        classify_feedback_ml(
            FeedbackMessage(message_id="msg_test", user_message="Bitte mehr Übungen.")
        )

    assert "Run python -m src.train_ml_classifier first" in str(exc_info.value)


def test_safety_probability_routes_to_safety_review() -> None:
    artifact = _dummy_artifact(
        probabilities=[0.2, 0.1, 0.4],
        safety_probability=0.5,
    )

    classified = _classify_feedback_ml_with_artifact(
        FeedbackMessage(message_id="msg_test", user_message="Mir wird schwindelig."),
        artifact,
    )

    assert classified.routing == "safety_review"
    assert classified.safety_flag is True
    assert "safety_signal" in classified.labels


def test_ml_abstain_returns_needs_review() -> None:
    artifact = _dummy_artifact(
        probabilities=[0.56, 0.52, 0.1],
        safety_probability=0.1,
    )

    classified = _classify_feedback_ml_with_artifact(
        FeedbackMessage(message_id="msg_test", user_message="Bitte mehr Übungen."),
        artifact,
    )

    assert classified.routing == "needs_review"
    assert classified.abstained is True
    assert classified.classifier_source == "ml_abstain"


def test_hybrid_uses_ml_if_auto_accepted(monkeypatch) -> None:
    ml_result = ClassifiedFeedback(
        message_id="msg_test",
        user_message="Bitte mehr Knieübungen.",
        labels=["content_request"],
        safety_flag=False,
        evidence_quote="Bitte mehr Knieübungen.",
        summary="ML accepted.",
        confidence=0.9,
        routing="process",
        classifier_source="ml",
        abstained=False,
        top2_margin=0.2,
    )
    monkeypatch.setattr(classify_module, "classify_feedback_ml", lambda message: ml_result)

    classified = classify_feedback_hybrid(
        FeedbackMessage(message_id="msg_test", user_message="Bitte mehr Knieübungen.")
    )

    assert classified.classifier_source == "hybrid_ml"


def test_hybrid_falls_back_when_ml_abstains(monkeypatch) -> None:
    ml_result = ClassifiedFeedback(
        message_id="msg_test",
        user_message="Bitte mehr Knieübungen.",
        labels=["content_request"],
        safety_flag=False,
        evidence_quote="Bitte mehr Knieübungen.",
        summary="ML abstained.",
        confidence=0.5,
        routing="needs_review",
        classifier_source="ml_abstain",
        abstained=True,
        top2_margin=0.01,
    )
    llm_result = ClassifiedFeedback(
        message_id="msg_test",
        user_message="Bitte mehr Knieübungen.",
        labels=["content_request"],
        safety_flag=False,
        evidence_quote="Bitte mehr Knieübungen.",
        summary="LLM classified.",
        confidence=0.88,
        routing="process",
        classifier_source="llm",
        abstained=False,
    )
    monkeypatch.setattr(classify_module, "classify_feedback_ml", lambda message: ml_result)
    monkeypatch.setattr(classify_module, "classify_feedback_llm", lambda message: llm_result)

    classified = classify_feedback_hybrid(
        FeedbackMessage(message_id="msg_test", user_message="Bitte mehr Knieübungen.")
    )

    assert classified.classifier_source == "hybrid_llm"
    assert classified.summary == "LLM classified."


def _dummy_artifact(probabilities, safety_probability):
    return {
        "vectorizer": DummyVectorizer(),
        "multilabel_classifier": DummyMultilabelClassifier(probabilities),
        "label_binarizer": DummyLabelBinarizer(),
        "safety_classifier": DummySafetyClassifier(safety_probability),
        "thresholds": {},
    }


def _write_sample_training_data(tmp_path: Path) -> tuple[Path, Path]:
    feedback_path = tmp_path / "feedback_messages.csv"
    labels_path = tmp_path / "expected_labels.csv"
    feedback_rows = [
        ("msg_001", "Bitte mehr Knieübungen ohne Geräte."),
        ("msg_002", "Die Übung war für Anfänger zu schwer."),
        ("msg_003", "Das Video war sehr verständlich."),
        ("msg_004", "Mir wird schwindelig."),
        ("msg_005", "Das Video lädt nicht."),
        ("msg_006", "Die Suche zeigt falsche Übungen."),
        ("msg_007", "Weiß nicht."),
        ("msg_008", "Anderes Thema."),
    ]
    label_rows = [
        ("msg_001", "content_request", "false"),
        ("msg_002", "criticism", "false"),
        ("msg_003", "praise", "false"),
        ("msg_004", "safety_signal", "true"),
        ("msg_005", "bug_or_access_problem", "false"),
        ("msg_006", "metadata_issue", "false"),
        ("msg_007", "unclear", "false"),
        ("msg_008", "other", "false"),
    ]
    with feedback_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["message_id", "user_message"])
        writer.writerows(feedback_rows)
    with labels_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["message_id", "expected_labels", "expected_safety_flag"])
        writer.writerows(label_rows)
    return feedback_path, labels_path
