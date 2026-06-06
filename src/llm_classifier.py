import json
import os
from pathlib import Path

from dotenv import load_dotenv
from pydantic import ValidationError

from src.schemas import ClassifiedFeedback, FeedbackMessage


ROOT = Path(__file__).resolve().parents[1]
CLASSIFIER_PROMPT_PATH = ROOT / "prompts" / "classifier_prompt.md"
DEFAULT_OPENAI_MODEL = "gpt-5.4-nano"
LLM_METADATA_ALLOWED_VALUES = {
    "body_region": {"knee", "back", "shoulder", "hip", "ankle", "neck", "jaw", "general"},
    "therapy_goal": {
        "balance",
        "coordination",
        "endurance",
        "mobility",
        "relaxation",
        "stability",
        "strength",
    },
    "difficulty_requested": {"beginner", "intermediate", "advanced"},
    "equipment": {"none", "chair", "mat", "miniband", "theraband", "towel", "wall"},
    "position": {"all_fours", "kneeling", "lying", "sitting", "standing"},
}
DEFAULT_CLASSIFIER_PROMPT = """
Classify German rehabilitation content feedback into one strict JSON object.
Return the classification fields at the top level. Do not wrap them inside
classification, result, data, or any other object. Use only allowed labels:
content_request, criticism, praise, safety_signal, bug_or_access_problem,
metadata_issue, unclear, other. Include an evidence_quote copied exactly from
the user message. Do not provide medical advice. If the message mentions pain,
dizziness, unsafe movement, or post-operative uncertainty, set safety_flag true
and route to safety_review.

Normalized metadata values:
- body_region: knee, back, shoulder, hip, ankle, neck, jaw, general, or null
- difficulty_requested: beginner, intermediate, advanced, or null
- equipment: none, chair, theraband, miniband, wall, mat, towel, or null
- position: sitting, standing, lying, kneeling, all_fours, or null
- request_theme: short canonical English noun phrase for the requested content
  topic, or null. Translate German wording, normalize synonyms, and prefer the
  underlying anatomy/problem/action over copying the message literally.

Required JSON shape:
{
  "message_id": "same message_id from input",
  "user_message": "same user_message from input",
  "labels": ["content_request"],
  "body_region": null,
  "therapy_goal": null,
  "difficulty_requested": null,
  "equipment": null,
  "position": null,
  "request_theme": null,
  "sentiment": "neutral",
  "safety_flag": false,
  "evidence_quote": "exact substring copied from user_message",
  "summary": "short classification summary",
  "confidence": 0.85,
  "routing": "process"
}
""".strip()


def classify_feedback_llm(message: FeedbackMessage) -> ClassifiedFeedback:
    """Classify feedback with the OpenAI API and validate the JSON response."""
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    model = os.getenv("OPENAI_MODEL") or DEFAULT_OPENAI_MODEL

    if not api_key:
        return _llm_fallback_classification(
            message,
            ValueError("OPENAI_API_KEY is missing."),
        )

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
            payload = _normalize_llm_payload(payload, message)
            classified = ClassifiedFeedback(**payload)
            _validate_llm_classification(classified, message)
            classified.classifier_source = "llm"
            if _has_request_metadata(classified):
                classified.metadata_source = "llm_classifier"
            classified.abstained = False
            return classified
        except (
            json.JSONDecodeError,
            KeyError,
            TypeError,
            ValueError,
            ValidationError,
        ) as exc:
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
                    "Return only one JSON object with the required top-level "
                    "fields from the system prompt. Input:\n"
                    f"{json.dumps(user_payload, ensure_ascii=False)}"
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


def _normalize_llm_payload(payload: dict, message: FeedbackMessage) -> dict:
    normalized = _unwrap_llm_payload(payload)
    if "message_id" not in normalized:
        normalized["message_id"] = message.message_id
    if "user_message" not in normalized:
        normalized["user_message"] = message.user_message
    _normalize_llm_metadata_fields(normalized)
    return normalized


def _unwrap_llm_payload(payload: dict) -> dict:
    for key in ("classification", "classified_feedback", "result", "data", "feedback"):
        nested = payload.get(key)
        if isinstance(nested, dict):
            return {**payload, **nested}
    return payload


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


def _normalize_llm_metadata_fields(payload: dict) -> None:
    for field, allowed_values in LLM_METADATA_ALLOWED_VALUES.items():
        value = payload.get(field)
        if value is None:
            payload[field] = None
            continue
        normalized = str(value).strip().casefold()
        if normalized in allowed_values:
            payload[field] = normalized
        else:
            payload[field] = None
    payload["request_theme"] = _normalize_request_theme(payload.get("request_theme"))


def _normalize_request_theme(value) -> str | None:
    if value is None:
        return None
    normalized = " ".join(str(value).strip().casefold().split())
    if not normalized or normalized in {"null", "none", "unknown", "n/a"}:
        return None
    return normalized[:80]


def _has_request_metadata(classified: ClassifiedFeedback) -> bool:
    return any(
        value is not None
        for value in (
            classified.body_region,
            classified.therapy_goal,
            classified.difficulty_requested,
            classified.equipment,
            classified.position,
            classified.request_theme,
        )
    )


def _llm_fallback_classification(
    message: FeedbackMessage, error: Exception | None = None
) -> ClassifiedFeedback:
    reason = "LLM classification failed validation."
    if error is not None:
        reason = f"{reason} Error: {type(error).__name__}: {error}"
    return ClassifiedFeedback(
        message_id=message.message_id,
        user_message=message.user_message,
        labels=["unclear"],
        body_region=None,
        therapy_goal=None,
        difficulty_requested=None,
        equipment=None,
        position=None,
        request_theme=None,
        sentiment=None,
        safety_flag=False,
        evidence_quote=message.user_message[:120],
        summary="LLM classification failed validation.",
        confidence=0.0,
        routing="needs_review",
        classifier_source="llm",
        abstained=True,
        abstain_reason=reason,
    )
