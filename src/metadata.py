import json
import os
from pathlib import Path

from dotenv import load_dotenv
from pydantic import BaseModel, ValidationError

from src.llm_classifier import DEFAULT_OPENAI_MODEL
from src.rules import enrich_classification_metadata
from src.schemas import ClassifiedFeedback
from src.text_signals import extract_topic_tokens


ROOT = Path(__file__).resolve().parents[1]
METADATA_PROMPT_PATH = ROOT / "prompts" / "metadata_extractor_prompt.md"
DEFAULT_METADATA_PROMPT = """
Extract normalized request metadata from German rehabilitation content feedback.
Return only one JSON object. Do not classify the message and do not provide
medical advice.

Allowed values:
- body_region: knee, back, shoulder, hip, ankle, neck, jaw, general, or null
- therapy_goal: balance, coordination, endurance, mobility, relaxation, stability, strength, or null
- difficulty_requested: beginner, intermediate, advanced, or null
- equipment: none, chair, mat, miniband, theraband, towel, wall, or null
- position: all_fours, kneeling, lying, sitting, standing, or null
- request_theme: short canonical English noun phrase for the requested content
  topic, or null. Translate German wording, normalize synonyms, and prefer the
  underlying anatomy/problem/action over copying the message literally.

Required JSON shape:
{
  "body_region": null,
  "therapy_goal": null,
  "difficulty_requested": null,
  "equipment": null,
  "position": null,
  "request_theme": null
}
""".strip()

ALLOWED_VALUES = {
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
METADATA_FIELDS = tuple(ALLOWED_VALUES)
REQUEST_METADATA_FIELDS = (*METADATA_FIELDS, "request_theme")


class RequestMetadata(BaseModel):
    body_region: str | None = None
    therapy_goal: str | None = None
    difficulty_requested: str | None = None
    equipment: str | None = None
    position: str | None = None
    request_theme: str | None = None


def enrich_request_metadata(
    classified: ClassifiedFeedback,
    mode: str = "auto",
    classifier_mode: str | None = None,
    matcher_mode: str | None = None,
) -> ClassifiedFeedback:
    """Fill request metadata after classification without changing labels."""
    if not _should_extract_metadata(classified):
        return classified

    selected_mode = _select_metadata_mode(mode, classifier_mode, matcher_mode)
    if selected_mode == "none" or _has_complete_metadata(classified):
        return classified
    if selected_mode == "rules":
        return extract_request_metadata_rules(classified)
    if selected_mode == "llm":
        return extract_request_metadata_llm(classified)
    raise ValueError(f"Unsupported metadata extractor: {mode}")


def extract_request_metadata_rules(classified: ClassifiedFeedback) -> ClassifiedFeedback:
    before = _metadata_snapshot(classified)
    enriched = enrich_classification_metadata(classified)
    if enriched.request_theme is None:
        enriched.request_theme = _fallback_request_theme(enriched.user_message)
    if _metadata_snapshot(enriched) != before and enriched.metadata_source is None:
        enriched.metadata_source = "rules"
    return enriched


def extract_request_metadata_llm(classified: ClassifiedFeedback) -> ClassifiedFeedback:
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    model = (
        os.getenv("OPENAI_METADATA_MODEL")
        or os.getenv("OPENAI_MODEL")
        or DEFAULT_OPENAI_MODEL
    )
    if not api_key:
        return classified

    try:
        raw_response = _call_openai_metadata_extractor(
            api_key=api_key,
            model=model,
            system_prompt=_load_metadata_prompt(),
            classified=classified,
        )
        metadata = _normalize_metadata_payload(_parse_json_object(raw_response))
    except (json.JSONDecodeError, TypeError, ValueError, ValidationError):
        return classified
    except Exception:
        return classified

    return _merge_missing_metadata(
        classified,
        metadata,
        metadata_source="llm_metadata",
    )


def _select_metadata_mode(
    mode: str,
    classifier_mode: str | None,
    matcher_mode: str | None,
) -> str:
    if mode != "auto":
        return mode
    load_dotenv()
    if classifier_mode == "mock" or matcher_mode == "placeholder":
        return "rules"
    if os.getenv("OPENAI_API_KEY"):
        return "llm"
    return "none"


def _should_extract_metadata(classified: ClassifiedFeedback) -> bool:
    return "content_request" in classified.labels and not classified.safety_flag


def _has_complete_metadata(classified: ClassifiedFeedback) -> bool:
    return all(
        getattr(classified, field) is not None for field in REQUEST_METADATA_FIELDS
    )


def _merge_missing_metadata(
    classified: ClassifiedFeedback,
    metadata: RequestMetadata,
    metadata_source: str,
) -> ClassifiedFeedback:
    changed = False
    for field in REQUEST_METADATA_FIELDS:
        if getattr(classified, field) is None:
            value = getattr(metadata, field)
            if value is not None:
                setattr(classified, field, value)
                changed = True
    if changed and classified.metadata_source is None:
        classified.metadata_source = metadata_source
    return classified


def _normalize_metadata_payload(payload: dict) -> RequestMetadata:
    normalized = {}
    for field, allowed_values in ALLOWED_VALUES.items():
        normalized[field] = _normalize_metadata_value(payload.get(field), allowed_values)
    normalized["request_theme"] = _normalize_request_theme(payload.get("request_theme"))
    return RequestMetadata(**normalized)


def _normalize_metadata_value(value, allowed_values: set[str]) -> str | None:
    if value is None:
        return None
    normalized = str(value).strip().casefold()
    if normalized in allowed_values:
        return normalized
    if not normalized or normalized in {"null", "none", "unknown", "n/a"}:
        return None
    return None


def _normalize_request_theme(value) -> str | None:
    if value is None:
        return None
    normalized = " ".join(str(value).strip().casefold().split())
    if not normalized or normalized in {"null", "none", "unknown", "n/a"}:
        return None
    return normalized[:80]


def _call_openai_metadata_extractor(
    api_key: str,
    model: str,
    system_prompt: str,
    classified: ClassifiedFeedback,
) -> str:
    from openai import OpenAI

    client = OpenAI(api_key=api_key)
    user_payload = {
        "message_id": classified.message_id,
        "user_message": classified.user_message,
        "labels": classified.labels,
        "summary": classified.summary,
    }
    completion = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": (
                    "Extract request metadata from this feedback. "
                    "Return only the required JSON object. Input:\n"
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


def _load_metadata_prompt() -> str:
    if METADATA_PROMPT_PATH.exists():
        return METADATA_PROMPT_PATH.read_text(encoding="utf-8")
    return DEFAULT_METADATA_PROMPT


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


def _metadata_snapshot(classified: ClassifiedFeedback) -> tuple:
    return tuple(
        [
            *(getattr(classified, field) for field in METADATA_FIELDS),
            classified.request_theme,
        ]
    )


def _fallback_request_theme(text: str) -> str | None:
    topics = sorted(extract_topic_tokens(text))
    if not topics:
        return None
    return " ".join(topics[:4])
