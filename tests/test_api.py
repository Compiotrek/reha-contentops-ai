import pytest
from fastapi import HTTPException

from src.api import ProcessFeedbackRequest, health, process_feedback_endpoint


def test_health_endpoint() -> None:
    assert health() == {"status": "ok", "service": "reha-contentops-ai"}


def test_process_feedback_mock_placeholder() -> None:
    payload = process_feedback_endpoint(
        ProcessFeedbackRequest(
            message_id="msg_demo_001",
            user_message="Die Knieübungen sind zu schwer. Ich hätte gern leichtere Varianten ohne Geräte.",
            classifier="mock",
            matcher="placeholder",
        )
    )

    assert payload["message"]["message_id"] == "msg_demo_001"
    assert payload["classifier_mode"] == "mock"
    assert payload["matcher_mode"] == "placeholder"
    assert payload["routing"]
    assert payload["top_matches"]
    assert payload["match_status"]
    assert payload["decision_reason"]
    assert payload["final_action"]


def test_invalid_classifier_returns_400() -> None:
    with pytest.raises(HTTPException) as exc_info:
        process_feedback_endpoint(
            ProcessFeedbackRequest(
                message_id="msg_demo_001",
                user_message="Bitte mehr Knieübungen.",
                classifier="bad",
                matcher="placeholder",
            )
        )

    assert exc_info.value.status_code == 400
    assert "Invalid classifier" in exc_info.value.detail


def test_invalid_matcher_returns_400() -> None:
    with pytest.raises(HTTPException) as exc_info:
        process_feedback_endpoint(
            ProcessFeedbackRequest(
                message_id="msg_demo_001",
                user_message="Bitte mehr Knieübungen.",
                classifier="mock",
                matcher="bad",
            )
        )

    assert exc_info.value.status_code == 400
    assert "Invalid matcher" in exc_info.value.detail
