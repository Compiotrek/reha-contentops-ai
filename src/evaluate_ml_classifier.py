from pathlib import Path

import numpy as np
from sklearn.metrics import classification_report, f1_score, recall_score
from sklearn.model_selection import train_test_split

from src.classify import (
    MEANINGFUL_LABELS,
    ML_AUTO_ACCEPT_CONFIDENCE,
    ML_AUTO_ACCEPT_MARGIN,
    ML_SAFETY_THRESHOLD,
    _predict_safety_probability,
    _top2_margin,
)
from src.train_ml_classifier import DEFAULT_THRESHOLDS, fit_ml_classifier, load_training_data


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_PATH = ROOT / "outputs" / "ml_evaluation.md"
MIN_SPLIT_SIZE = 80


def evaluate_ml_classifier(output_path: str | Path = OUTPUT_PATH) -> str:
    data = load_training_data()
    warning = ""
    if len(data) >= MIN_SPLIT_SIZE:
        train_data, eval_data = train_test_split(
            data,
            test_size=0.25,
            random_state=42,
            shuffle=True,
        )
    else:
        train_data = data
        eval_data = data
        warning = (
            "Warning: dataset is small, so this report evaluates on training data. "
            "Metrics are illustrative only."
        )

    artifact = fit_ml_classifier(train_data)
    vectorizer = artifact["vectorizer"]
    label_binarizer = artifact["label_binarizer"]
    labels = list(label_binarizer.classes_)
    features = vectorizer.transform(eval_data["user_message"].astype(str).tolist())

    y_true = label_binarizer.transform(eval_data["label_list"])
    probabilities = artifact["multilabel_classifier"].predict_proba(features)
    y_pred = _threshold_predictions(labels, probabilities)

    safety_true = eval_data["expected_safety_flag"].astype(int).to_numpy()
    safety_probabilities = np.array(
        [_predict_safety_probability(artifact, features[index]) for index in range(features.shape[0])]
    )
    safety_pred = safety_probabilities >= ML_SAFETY_THRESHOLD

    abstained = _abstention_mask(labels, probabilities, safety_probabilities)
    report = classification_report(
        y_true,
        y_pred,
        target_names=labels,
        zero_division=0,
        output_dict=True,
    )
    micro_f1 = f1_score(y_true, y_pred, average="micro", zero_division=0)
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    safety_recall = recall_score(safety_true, safety_pred, zero_division=0)
    content_request_recall = _label_recall(
        labels, y_true, y_pred, label="content_request"
    )

    lines = [
        "# ML Classifier Evaluation",
        "",
        "This report is illustrative. The dataset is synthetic and not clinically validated.",
        "",
    ]
    if warning:
        lines.extend([warning, ""])
    lines.extend(
        [
            f"- Training rows: {len(train_data)}",
            f"- Evaluation rows: {len(eval_data)}",
            f"- Micro F1: {micro_f1:.3f}",
            f"- Macro F1: {macro_f1:.3f}",
            f"- Safety recall: {safety_recall:.3f}",
            f"- Content request recall: {content_request_recall:.3f}",
            f"- Abstain rate: {abstained.mean():.3f}",
            f"- Estimated LLM fallback rate: {abstained.mean():.3f}",
            "",
            "## Per-Label Metrics",
        ]
    )
    for label in labels:
        label_report = report[label]
        lines.append(
            f"- {label}: precision {label_report['precision']:.3f}, "
            f"recall {label_report['recall']:.3f}, "
            f"f1 {label_report['f1-score']:.3f}"
        )

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return "\n".join(lines)


def _threshold_predictions(labels: list[str], probabilities) -> np.ndarray:
    predictions = np.zeros_like(probabilities, dtype=int)
    for index, label in enumerate(labels):
        threshold = DEFAULT_THRESHOLDS.get(label, 0.60)
        predictions[:, index] = probabilities[:, index] >= threshold
    return predictions


def _abstention_mask(
    labels: list[str],
    probabilities,
    safety_probabilities: np.ndarray,
) -> np.ndarray:
    abstained = []
    for row, safety_probability in zip(probabilities, safety_probabilities):
        label_probs = dict(zip(labels, row))
        predicted_labels = [
            label
            for label, probability in label_probs.items()
            if probability >= DEFAULT_THRESHOLDS.get(label, 0.60)
        ]
        max_confidence = max(label_probs.values()) if label_probs else 0.0
        top2_margin = _top2_margin(list(label_probs.values()))
        auto_accept = (
            safety_probability < ML_SAFETY_THRESHOLD
            and max_confidence >= ML_AUTO_ACCEPT_CONFIDENCE
            and top2_margin >= ML_AUTO_ACCEPT_MARGIN
            and any(label in MEANINGFUL_LABELS for label in predicted_labels)
        )
        abstained.append(not auto_accept and safety_probability < ML_SAFETY_THRESHOLD)
    return np.array(abstained)


def _label_recall(labels: list[str], y_true, y_pred, label: str) -> float:
    if label not in labels:
        return 0.0
    index = labels.index(label)
    return recall_score(y_true[:, index], y_pred[:, index], zero_division=0)


def main() -> None:
    report = evaluate_ml_classifier()
    print(report)
    print(f"\nWrote ML evaluation report to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
