from src.metadata import (
    RequestMetadata,
    _normalize_metadata_payload,
    enrich_request_metadata,
)
from src.schemas import ClassifiedFeedback


def _classified(**overrides) -> ClassifiedFeedback:
    data = {
        "message_id": "msg_test",
        "user_message": "Bitte mehr leichtere Knieübungen ohne Geräte.",
        "labels": ["content_request"],
        "safety_flag": False,
        "evidence_quote": "Bitte mehr leichtere Knieübungen ohne Geräte.",
        "summary": "User requests easier knee exercises without equipment.",
        "confidence": 0.9,
        "routing": "process",
        "classifier_source": "ml",
    }
    data.update(overrides)
    return ClassifiedFeedback(**data)


def test_enrich_request_metadata_rules_fills_missing_fields() -> None:
    classified = enrich_request_metadata(
        _classified(),
        mode="rules",
        classifier_mode="ml",
        matcher_mode="embeddings",
    )

    assert classified.body_region == "knee"
    assert classified.equipment == "none"
    assert classified.difficulty_requested == "beginner"
    assert classified.request_theme == "knie leichter"
    assert classified.metadata_source == "rules"


def test_enrich_request_metadata_skips_non_content_requests() -> None:
    classified = enrich_request_metadata(
        _classified(labels=["praise"]),
        mode="rules",
        classifier_mode="ml",
        matcher_mode="embeddings",
    )

    assert classified.body_region is None
    assert classified.metadata_source is None


def test_normalize_metadata_payload_keeps_only_allowed_values() -> None:
    metadata = _normalize_metadata_payload(
        {
            "body_region": "KNEE",
            "therapy_goal": "strength",
            "difficulty_requested": "easy",
            "equipment": "NONE",
            "position": "standing",
            "request_theme": "  Plantar Fascia / Foot Arch  ",
        }
    )

    assert isinstance(metadata, RequestMetadata)
    assert metadata.body_region == "knee"
    assert metadata.therapy_goal == "strength"
    assert metadata.difficulty_requested is None
    assert metadata.equipment == "none"
    assert metadata.position == "standing"
    assert metadata.request_theme == "plantar fascia / foot arch"


def test_normalize_metadata_payload_accepts_jaw_body_region() -> None:
    metadata = _normalize_metadata_payload(
        {
            "body_region": "JAW",
            "request_theme": "jaw relaxation",
        }
    )

    assert metadata.body_region == "jaw"
    assert metadata.request_theme == "jaw relaxation"
