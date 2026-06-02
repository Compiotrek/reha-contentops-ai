import pytest
from fastapi import HTTPException

from src import api as api_module
from src.api import ProcessFeedbackRequest, health, process_feedback_endpoint
from src.schemas import ClassifiedFeedback, DecisionResult, FeedbackMessage, ProcessedFeedback


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


def test_process_feedback_accepts_ml_classifier(tmp_path, monkeypatch) -> None:
    _patch_model_and_processing(tmp_path, monkeypatch)

    payload = process_feedback_endpoint(
        ProcessFeedbackRequest(
            message_id="msg_demo_001",
            user_message="Bitte mehr Knieübungen.",
            classifier="ml",
            matcher="placeholder",
        )
    )

    assert payload["classifier_mode"] == "ml"


def test_process_feedback_accepts_hybrid_classifier(tmp_path, monkeypatch) -> None:
    _patch_model_and_processing(tmp_path, monkeypatch)

    payload = process_feedback_endpoint(
        ProcessFeedbackRequest(
            message_id="msg_demo_001",
            user_message="Bitte mehr Knieübungen.",
            classifier="hybrid",
            matcher="placeholder",
        )
    )

    assert payload["classifier_mode"] == "hybrid"


def _patch_model_and_processing(tmp_path, monkeypatch) -> None:
    model_path = tmp_path / "ml_classifier.joblib"
    model_path.write_text("placeholder", encoding="utf-8")
    monkeypatch.setattr(api_module, "CLASSIFIER_MODEL_PATH", model_path)

    def fake_process_single_feedback(
        feedback_message,
        exercises,
        classifier_mode,
        matcher_mode,
    ):
        classification = ClassifiedFeedback(
            message_id=feedback_message.message_id,
            user_message=feedback_message.user_message,
            labels=["content_request"],
            safety_flag=False,
            evidence_quote=feedback_message.user_message,
            summary="Fake processed item.",
            confidence=0.9,
            routing="process",
        )
        decision = DecisionResult(
            routing="process",
            match_status="track_only",
            reason="Fake decision.",
            final_action="track_request",
        )
        return ProcessedFeedback(
            message=FeedbackMessage(
                message_id=feedback_message.message_id,
                user_message=feedback_message.user_message,
            ),
            classification=classification,
            matches=[],
            decision=decision,
            classifier_mode=classifier_mode,
            matcher_mode=matcher_mode,
            routing="process",
            top_matches=[],
            match_status="track_only",
            decision_reason="Fake decision.",
            final_action="track_request",
        )

    monkeypatch.setattr(api_module, "process_single_feedback", fake_process_single_feedback)
