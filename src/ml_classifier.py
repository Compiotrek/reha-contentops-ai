from pathlib import Path

import joblib

from src.config import (
    ML_AUTO_ACCEPT_CONFIDENCE,
    ML_AUTO_ACCEPT_MARGIN,
    ML_SAFETY_THRESHOLD,
)
from src.schemas import ClassifiedFeedback, FeedbackMessage
from src.train_ml_classifier import DEFAULT_THRESHOLDS


ROOT = Path(__file__).resolve().parents[1]
CLASSIFIER_MODEL_PATH = ROOT / "models" / "ml_classifier.joblib"
MEANINGFUL_LABELS = {
    "content_request",
    "criticism",
    "praise",
    "bug_or_access_problem",
    "metadata_issue",
}


class MLClassifierModelNotFoundError(FileNotFoundError):
    """Raised when the local ML classifier artifact has not been trained yet."""


def classify_feedback_ml(message: FeedbackMessage) -> ClassifiedFeedback:
    artifact = _load_ml_artifact()
    return _classify_feedback_ml_with_artifact(message, artifact)


def _load_ml_artifact() -> dict:
    if not CLASSIFIER_MODEL_PATH.exists():
        raise MLClassifierModelNotFoundError(
            "ML classifier model not found. Run python -m src.train_ml_classifier first."
        )
    return joblib.load(CLASSIFIER_MODEL_PATH)


def _classify_feedback_ml_with_artifact(
    message: FeedbackMessage, artifact: dict
) -> ClassifiedFeedback:
    features = artifact["vectorizer"].transform([message.user_message])
    label_probs = _predict_label_probabilities(artifact, features)
    safety_probability = _predict_safety_probability(artifact, features)
    thresholds = {**DEFAULT_THRESHOLDS, **artifact.get("thresholds", {})}

    max_confidence = max(label_probs.values()) if label_probs else 0.0
    top2_margin = _top2_margin(list(label_probs.values()))
    predicted_labels = [
        label
        for label, probability in label_probs.items()
        if probability >= thresholds.get(label, 0.60)
    ]

    if safety_probability >= ML_SAFETY_THRESHOLD:
        labels = _dedupe_labels(["safety_signal", *predicted_labels])
        return ClassifiedFeedback(
            message_id=message.message_id,
            user_message=message.user_message,
            labels=labels,
            safety_flag=True,
            evidence_quote=message.user_message[:160],
            summary="ML classifier detected a possible safety signal.",
            confidence=round(max(max_confidence, safety_probability), 3),
            routing="safety_review",
            classifier_source="ml",
            abstained=False,
            safety_probability=round(safety_probability, 3),
            top2_margin=round(top2_margin, 3),
        )

    if not predicted_labels:
        predicted_labels = ["unclear"]

    accepted = (
        max_confidence >= ML_AUTO_ACCEPT_CONFIDENCE
        and top2_margin >= ML_AUTO_ACCEPT_MARGIN
        and any(label in MEANINGFUL_LABELS for label in predicted_labels)
    )
    if accepted:
        return ClassifiedFeedback(
            message_id=message.message_id,
            user_message=message.user_message,
            labels=predicted_labels,
            safety_flag=False,
            evidence_quote=message.user_message[:160],
            summary="ML classifier auto-accepted a high-confidence prediction.",
            confidence=round(max_confidence, 3),
            routing="process",
            classifier_source="ml",
            abstained=False,
            safety_probability=round(safety_probability, 3),
            top2_margin=round(top2_margin, 3),
        )

    return ClassifiedFeedback(
        message_id=message.message_id,
        user_message=message.user_message,
        labels=predicted_labels,
        safety_flag=False,
        evidence_quote=message.user_message[:160],
        summary="ML classifier abstained due to low confidence or low top-2 margin.",
        confidence=round(max_confidence, 3),
        routing="needs_review",
        classifier_source="ml_abstain",
        abstained=True,
        abstain_reason="low confidence or low top-2 margin",
        safety_probability=round(safety_probability, 3),
        top2_margin=round(top2_margin, 3),
    )


def _predict_label_probabilities(artifact: dict, features) -> dict[str, float]:
    classifier = artifact["multilabel_classifier"]
    label_binarizer = artifact["label_binarizer"]
    probabilities = classifier.predict_proba(features)[0]
    return {
        label: float(probability)
        for label, probability in zip(label_binarizer.classes_, probabilities)
    }


def _predict_safety_probability(artifact: dict, features) -> float:
    classifier = artifact["safety_classifier"]
    if len(classifier.classes_) == 1:
        return float(classifier.classes_[0])
    class_index = list(classifier.classes_).index(1)
    return float(classifier.predict_proba(features)[0][class_index])


def _top2_margin(probabilities: list[float]) -> float:
    if not probabilities:
        return 0.0
    sorted_probs = sorted(probabilities, reverse=True)
    if len(sorted_probs) == 1:
        return sorted_probs[0]
    return sorted_probs[0] - sorted_probs[1]


def ml_auto_accepted(result: ClassifiedFeedback, threshold: float) -> bool:
    return (
        result.routing == "process"
        and result.abstained is False
        and result.confidence >= threshold
        and (result.top2_margin or 0.0) >= ML_AUTO_ACCEPT_MARGIN
        and any(label in MEANINGFUL_LABELS for label in result.labels)
    )


def _dedupe_labels(labels: list[str]) -> list[str]:
    deduped: list[str] = []
    for label in labels:
        if label not in deduped:
            deduped.append(label)
    return deduped
