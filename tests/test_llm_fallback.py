from src.classify import _llm_fallback_classification, _normalize_llm_payload
from src.schemas import ClassifiedFeedback, FeedbackMessage


def test_llm_fallback_classification_object_is_valid() -> None:
    message = FeedbackMessage(
        message_id="msg_test",
        user_message="Ich verstehe die Übung nicht und brauche Hilfe.",
    )

    classified = _llm_fallback_classification(message)

    assert isinstance(classified, ClassifiedFeedback)
    assert classified.message_id == "msg_test"
    assert classified.labels == ["unclear"]
    assert classified.safety_flag is False
    assert classified.confidence == 0.0
    assert classified.routing == "needs_review"
    assert classified.evidence_quote == message.user_message[:120]


def test_llm_fallback_includes_error_reason() -> None:
    message = FeedbackMessage(
        message_id="msg_test",
        user_message="Ich verstehe die Übung nicht und brauche Hilfe.",
    )

    classified = _llm_fallback_classification(
        message,
        ValueError("OPENAI_API_KEY is missing."),
    )

    assert classified.abstain_reason is not None
    assert "ValueError" in classified.abstain_reason
    assert "OPENAI_API_KEY is missing" in classified.abstain_reason


def test_normalize_llm_payload_unwraps_nested_classification() -> None:
    message = FeedbackMessage(
        message_id="msg_test",
        user_message="Gibt es was fürs Knie ohne Geräte?",
    )
    payload = {
        "message_id": "msg_test",
        "classification": {
            "labels": ["content_request"],
            "body_region": "knee",
            "equipment": "none",
            "request_theme": "Knee exercises without equipment",
            "safety_flag": False,
            "evidence_quote": "Knie ohne Geräte",
            "summary": "User requests knee exercises without equipment.",
            "confidence": 0.9,
            "routing": "process",
        },
    }

    normalized = _normalize_llm_payload(payload, message)

    assert normalized["message_id"] == "msg_test"
    assert normalized["user_message"] == message.user_message
    assert normalized["labels"] == ["content_request"]
    assert normalized["body_region"] == "knee"
    assert normalized["equipment"] == "none"
    assert normalized["request_theme"] == "knee exercises without equipment"


def test_normalize_llm_payload_drops_invalid_metadata_values() -> None:
    message = FeedbackMessage(
        message_id="msg_test",
        user_message="Bitte mehr Übungen um einen Handstand zu lernen.",
    )
    payload = {
        "labels": ["content_request"],
        "body_region": "general",
        "therapy_goal": "Handstand lernen",
        "difficulty_requested": "advanced",
        "equipment": "unknown",
        "safety_flag": False,
        "evidence_quote": "Handstand zu lernen",
        "summary": "User requests handstand exercises.",
        "confidence": 0.9,
        "routing": "process",
    }

    normalized = _normalize_llm_payload(payload, message)

    assert normalized["body_region"] == "general"
    assert normalized["therapy_goal"] is None
    assert normalized["difficulty_requested"] == "advanced"
    assert normalized["equipment"] is None
