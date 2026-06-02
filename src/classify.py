import json
import os
from pathlib import Path

from dotenv import load_dotenv
from pydantic import ValidationError

from src.schemas import ClassifiedFeedback, FeedbackMessage


ROOT = Path(__file__).resolve().parents[1]
CLASSIFIER_PROMPT_PATH = ROOT / "prompts" / "classifier_prompt.md"
DEFAULT_OPENAI_MODEL = "gpt-5.4-nano"

SAFETY_KEYWORDS = ["pain", "schmerzen", "schwindelig", "unsicher", "op darf"]
REQUEST_KEYWORDS = ["mehr", "hätte gern", "bitte", "ich brauche"]
CRITICISM_KEYWORDS = ["schwer", "zu lang", "nervt", "problem"]
PRAISE_KEYWORDS = ["super", "gut", "verständlich"]
BUG_KEYWORDS = ["startbutton", "reagiert nicht", "app"]
METADATA_KEYWORDS = ["tag", "steht", "passt nicht", "anfänger"]


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
    )


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

    prompt = CLASSIFIER_PROMPT_PATH.read_text(encoding="utf-8")
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
    )


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
    if "theraband" in text:
        return "theraband"
    if "miniband" in text:
        return "miniband"
    if "stuhl" in text or "sitzen" in text:
        return "chair"
    return None


def _detect_position(text: str) -> str | None:
    if "sitzen" in text or "stuhl" in text:
        return "seated"
    if "liegen" in text:
        return "supine"
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
