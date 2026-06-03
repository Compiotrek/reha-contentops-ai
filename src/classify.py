import json
import os
import re
from pathlib import Path

import joblib
from dotenv import load_dotenv
from pydantic import ValidationError

from src.schemas import ClassifiedFeedback, FeedbackMessage
from src.train_ml_classifier import DEFAULT_THRESHOLDS


ROOT = Path(__file__).resolve().parents[1]
CLASSIFIER_PROMPT_PATH = ROOT / "prompts" / "classifier_prompt.md"
DEFAULT_OPENAI_MODEL = "gpt-5.4-nano"
CLASSIFIER_MODEL_PATH = ROOT / "models" / "ml_classifier.joblib"
ML_AUTO_ACCEPT_CONFIDENCE = 0.85
ML_AUTO_ACCEPT_MARGIN = 0.15
ML_SAFETY_THRESHOLD = 0.35
MEANINGFUL_LABELS = {
    "content_request",
    "criticism",
    "praise",
    "bug_or_access_problem",
    "metadata_issue",
}
DEFAULT_CLASSIFIER_PROMPT = """
Classify German rehabilitation content feedback into one strict JSON object.
Use the existing schema fields exactly. Use only allowed labels:
content_request, criticism, praise, safety_signal, bug_or_access_problem,
metadata_issue, unclear, other. Include an evidence_quote copied exactly from
the user message. Do not provide medical advice. If the message mentions pain,
dizziness, unsafe movement, or post-operative uncertainty, set safety_flag true
and route to safety_review.

Normalized metadata values:
- body_region: knee, back, shoulder, hip, ankle, neck, general, or null
- difficulty_requested: beginner, intermediate, advanced, or null
- equipment: none, chair, theraband, miniband, wall, mat, towel, or null
- position: sitting, standing, lying, kneeling, all_fours, or null
""".strip()

SAFETY_KEYWORDS = ["pain", "schmerzen", "schwindelig", "unsicher", "op darf"]
REQUEST_KEYWORDS = ["mehr", "hätte gern", "bitte", "ich brauche"]
CRITICISM_KEYWORDS = ["schwer", "zu lang", "nervt", "problem"]
PRAISE_KEYWORDS = ["super", "gut", "verständlich"]
BUG_KEYWORDS = ["startbutton", "reagiert nicht", "app"]
METADATA_KEYWORDS = ["tag", "steht", "passt nicht", "anfänger"]
RULE_GATE_SAFETY_PHRASES = [
    "starke schmerzen",
    "schmerzen bei der übung",
    "schmerz bei der übung",
    "mir wird schwindelig",
    "schwindelig",
    "wurde schlimmer",
    "verschlimmert",
    "nach meiner op",
    "nach der op",
    "darf ich",
    "unsicher ob ich",
]
RULE_GATE_BUG_PHRASES = [
    "video lädt nicht",
    "ton fehlt",
    "app stürzt ab",
    "lässt sich nicht öffnen",
    "link funktioniert nicht",
    "login funktioniert nicht",
]
RULE_GATE_PRAISE_PHRASES = [
    "sehr verständlich",
    "super erklärt",
    "hat mir gut geholfen",
    "gute erklärung",
    "kurze übungen gefallen mir",
]
RULE_GATE_PRAISE_BLOCKERS = [
    "bitte",
    "ich brauche",
    "hätte gern",
    "mehr",
    "schmerzen",
    "schwindelig",
    "unsicher",
    "darf ich",
]


class MLClassifierModelNotFoundError(FileNotFoundError):
    """Raised when the local ML classifier artifact has not been trained yet."""


def classify_feedback_mock(message: FeedbackMessage) -> ClassifiedFeedback:
    """Classify feedback with deterministic keyword rules.

    This is a mock implementation intended to be replaced by an LLM-backed
    classifier later. It does not provide medical advice.
    """
    text = message.user_message
    normalized = text.casefold()

    labels: list[str] = []
    safety_flag = _contains_any(normalized, SAFETY_KEYWORDS)
    if safety_flag:
        labels.append("safety_signal")
    if _contains_any(normalized, REQUEST_KEYWORDS):
        labels.append("content_request")
    if _contains_any(normalized, CRITICISM_KEYWORDS):
        labels.append("criticism")
    if _contains_any(normalized, PRAISE_KEYWORDS):
        labels.append("praise")
    if _contains_any(normalized, BUG_KEYWORDS):
        labels.append("bug_or_access_problem")
    if _contains_any(normalized, METADATA_KEYWORDS):
        labels.append("metadata_issue")

    if not labels:
        labels = ["unclear"]

    confidence = _estimate_confidence(labels, safety_flag)
    return ClassifiedFeedback(
        message_id=message.message_id,
        user_message=message.user_message,
        labels=labels,
        body_region=_detect_body_region(normalized),
        therapy_goal=_detect_therapy_goal(normalized),
        difficulty_requested=_detect_difficulty(normalized),
        equipment=_detect_equipment(normalized),
        position=_detect_position(normalized),
        sentiment=_detect_sentiment(labels),
        safety_flag=safety_flag,
        evidence_quote=text[:160],
        summary=_build_summary(labels, message.message_id),
        confidence=confidence,
        routing="needs_review",
        classifier_source="mock",
        abstained=False,
    )


def classify_feedback_ml(message: FeedbackMessage) -> ClassifiedFeedback:
    artifact = _load_ml_artifact()
    return _classify_feedback_ml_with_artifact(message, artifact)


def classify_feedback_hybrid(
    message: FeedbackMessage, threshold: float = ML_AUTO_ACCEPT_CONFIDENCE
) -> ClassifiedFeedback:
    gated_result = classify_with_rule_gates(message)
    if gated_result:
        return gated_result

    ml_result = classify_feedback_ml(message)
    if ml_result.routing == "safety_review":
        ml_result.classifier_source = "hybrid_ml"
        return ml_result
    if _ml_auto_accepted(ml_result, threshold):
        ml_result.classifier_source = "hybrid_ml"
        return ml_result

    llm_result = classify_feedback_llm(message)
    if llm_result.confidence > 0.0 or llm_result.summary != "LLM classification failed validation.":
        llm_result.classifier_source = "hybrid_llm"
        return llm_result

    ml_result.routing = "needs_review"
    ml_result.classifier_source = "ml_abstain"
    ml_result.abstained = True
    ml_result.abstain_reason = (
        "ML classifier abstained and LLM fallback was unavailable or failed."
    )
    return ml_result


def classify_with_rule_gates(message: FeedbackMessage) -> ClassifiedFeedback | None:
    normalized = message.user_message.casefold()
    if _contains_any(normalized, RULE_GATE_SAFETY_PHRASES):
        return ClassifiedFeedback(
            message_id=message.message_id,
            user_message=message.user_message,
            labels=["safety_signal"],
            sentiment="negative",
            safety_flag=True,
            evidence_quote=message.user_message[:120],
            summary="Rule gate detected a safety-related signal.",
            confidence=0.99,
            routing="safety_review",
            classifier_source="hybrid_rule_safety",
            abstained=False,
        )

    if _contains_any(normalized, RULE_GATE_BUG_PHRASES):
        return ClassifiedFeedback(
            message_id=message.message_id,
            user_message=message.user_message,
            labels=["bug_or_access_problem"],
            sentiment="negative",
            safety_flag=False,
            evidence_quote=message.user_message[:120],
            summary="Rule gate detected a technical or access issue.",
            confidence=0.95,
            routing="process",
            classifier_source="hybrid_rule_bug",
            abstained=False,
        )

    if _contains_any(normalized, RULE_GATE_PRAISE_PHRASES) and not _contains_any(
        normalized, RULE_GATE_PRAISE_BLOCKERS
    ):
        return ClassifiedFeedback(
            message_id=message.message_id,
            user_message=message.user_message,
            labels=["praise"],
            sentiment="positive",
            safety_flag=False,
            evidence_quote=message.user_message[:120],
            summary="Rule gate detected clear praise.",
            confidence=0.93,
            routing="process",
            classifier_source="hybrid_rule_praise",
            abstained=False,
        )

    return None


def classify_feedback_llm(message: FeedbackMessage) -> ClassifiedFeedback:
    """Classify feedback with the OpenAI API and validate the JSON response.

    The mock classifier remains the default pipeline path. This LLM classifier is
    optional, schema-validated, and falls back to human review on failures.
    """
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    model = os.getenv("OPENAI_MODEL") or DEFAULT_OPENAI_MODEL

    if not api_key:
        return _llm_fallback_classification(message)

    prompt = _load_classifier_prompt()
    last_error: Exception | None = None
    for _ in range(2):
        try:
            raw_response = _call_openai_classifier(
                api_key=api_key,
                model=model,
                system_prompt=prompt,
                message=message,
            )
            payload = _parse_json_object(raw_response)
            classified = ClassifiedFeedback(**payload)
            _validate_llm_classification(classified, message)
            classified.classifier_source = "llm"
            classified.abstained = False
            return classified
        except (json.JSONDecodeError, KeyError, TypeError, ValueError, ValidationError) as exc:
            last_error = exc
        except Exception as exc:
            last_error = exc
            break

    return _llm_fallback_classification(message, last_error)


def _call_openai_classifier(
    api_key: str,
    model: str,
    system_prompt: str,
    message: FeedbackMessage,
) -> str:
    from openai import OpenAI

    client = OpenAI(api_key=api_key)
    user_payload = {
        "message_id": message.message_id,
        "user_message": message.user_message,
    }
    completion = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": (
                    "Classify this German rehabilitation feedback. "
                    f"Return only strict JSON:\n{json.dumps(user_payload, ensure_ascii=False)}"
                ),
            },
        ],
        response_format={"type": "json_object"},
    )
    content = completion.choices[0].message.content
    if not content:
        raise ValueError("OpenAI response did not contain message content.")
    return content


def _load_classifier_prompt() -> str:
    if CLASSIFIER_PROMPT_PATH.exists():
        return CLASSIFIER_PROMPT_PATH.read_text(encoding="utf-8")
    return DEFAULT_CLASSIFIER_PROMPT


def _parse_json_object(raw_response: str) -> dict:
    cleaned = raw_response.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`").strip()
        if cleaned.startswith("json"):
            cleaned = cleaned[4:].strip()
    parsed = json.loads(cleaned)
    if not isinstance(parsed, dict):
        raise TypeError("LLM response must be a JSON object.")
    return parsed


def _validate_llm_classification(
    classified: ClassifiedFeedback, message: FeedbackMessage
) -> None:
    if classified.message_id != message.message_id:
        raise ValueError("LLM response changed message_id.")
    if classified.user_message != message.user_message:
        raise ValueError("LLM response changed user_message.")
    if not classified.evidence_quote:
        raise ValueError("LLM evidence_quote must not be empty.")
    if classified.evidence_quote not in message.user_message:
        raise ValueError("LLM evidence_quote must be copied from the user message.")


def _llm_fallback_classification(
    message: FeedbackMessage, error: Exception | None = None
) -> ClassifiedFeedback:
    return ClassifiedFeedback(
        message_id=message.message_id,
        user_message=message.user_message,
        labels=["unclear"],
        body_region=None,
        therapy_goal=None,
        difficulty_requested=None,
        equipment=None,
        position=None,
        sentiment=None,
        safety_flag=False,
        evidence_quote=message.user_message[:120],
        summary="LLM classification failed validation.",
        confidence=0.0,
        routing="needs_review",
        classifier_source="llm",
        abstained=True,
        abstain_reason="LLM classification failed validation.",
    )


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


def _ml_auto_accepted(result: ClassifiedFeedback, threshold: float) -> bool:
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


def _contains_any(text: str, keywords: list[str]) -> bool:
    return any(keyword in text for keyword in keywords)


def _detect_body_region(text: str) -> str | None:
    if "knie" in text:
        return "knee"
    if "rücken" in text:
        return "back"
    if "schulter" in text:
        return "shoulder"
    if "hüfte" in text:
        return "hip"
    if "sprunggelenk" in text or "knöchel" in text:
        return "ankle"
    if "nacken" in text:
        return "neck"
    if "allgemein" in text:
        return "general"
    return None


def _detect_therapy_goal(text: str) -> str | None:
    if "mobilisation" in text or "beweglichkeit" in text:
        return "mobility"
    if "balance" in text or "stabil" in text:
        return "stability"
    if "kräft" in text or "stärke" in text or "schwerere" in text:
        return "strength"
    return None


def _detect_difficulty(text: str) -> str | None:
    if "leichter" in text or "leichtere" in text or "anfänger" in text or "senioren" in text:
        return "beginner"
    if "schwerere" in text or "fortgeschritten" in text:
        return "advanced"
    if "schwer" in text:
        return "beginner"
    return None


def _detect_equipment(text: str) -> str | None:
    if "ohne geräte" in text:
        return "none"
    if "ohne equipment" in text:
        return "none"
    if "theraband" in text:
        return "theraband"
    if "miniband" in text:
        return "miniband"
    if "stuhl" in text:
        return "chair"
    if "wand" in text:
        return "wall"
    if "matte" in text:
        return "mat"
    if "handtuch" in text:
        return "towel"
    return None


def _detect_position(text: str) -> str | None:
    if "sitzen" in text or "stuhl" in text:
        return "sitting"
    if "liegen" in text:
        return "lying"
    if "knien" in text:
        return "kneeling"
    if "vierfüßler" in text or "vierfuessler" in text:
        return "all_fours"
    if "wand" in text or re.search(
        r"\b(?:im stand|im stehen|standposition|stehen|stehend)\b", text
    ):
        return "standing"
    return None


def _detect_sentiment(labels: list[str]) -> str | None:
    if "safety_signal" in labels:
        return "concerned"
    if "praise" in labels and "criticism" not in labels:
        return "positive"
    if "criticism" in labels or "bug_or_access_problem" in labels:
        return "negative"
    return "neutral"


def _estimate_confidence(labels: list[str], safety_flag: bool) -> float:
    if safety_flag:
        return 0.9
    if labels == ["unclear"]:
        return 0.45
    if len(labels) > 2:
        return 0.72
    return 0.82


def _build_summary(labels: list[str], message_id: str) -> str:
    return f"{message_id} classified as {', '.join(labels)} by mock keyword rules."
