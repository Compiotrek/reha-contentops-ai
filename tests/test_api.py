import pytest
from fastapi import HTTPException

from src import api as api_module
from src.api import (
    ProcessFeedbackBatchRequest,
    ProcessFeedbackRequest,
    health,
    process_feedback_batch_endpoint,
    process_feedback_endpoint,
)
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


def test_process_feedback_batch_accepts_json_messages(monkeypatch) -> None:
    captured = _patch_batch_processing(monkeypatch)

    payload = process_feedback_batch_endpoint(
        ProcessFeedbackBatchRequest(
            messages=[
                {
                    "message_id": "batch_001",
                    "user_message": "Bitte mehr Knieübungen.",
                },
                {
                    "message_id": "batch_002",
                    "user_message": "Das Video lädt nicht.",
                },
            ],
            classifier="mock",
            matcher="placeholder",
            metadata_extractor="rules",
            write_output_files=False,
        )
    )

    assert payload["processed_count"] == 2
    assert payload["exercise_records_loaded"] == len(api_module.EXERCISES)
    assert payload["outputs_written"] is False
    assert payload["content_gap_alerts"] == []
    assert [item["message"]["message_id"] for item in payload["items"]] == [
        "batch_001",
        "batch_002",
    ]
    assert captured["exercise_count"] == len(api_module.EXERCISES)


def test_process_feedback_batch_accepts_csv_text(monkeypatch) -> None:
    _patch_batch_processing(monkeypatch)
    csv_text = (
        "message_id,user_message,source\n"
        "csv_001,Bitte mehr Rückenübungen.,slack\n"
        "csv_002,Video bleibt schwarz,slack\n"
    )

    payload = process_feedback_batch_endpoint(
        ProcessFeedbackBatchRequest(
            csv_text=csv_text,
            classifier="mock",
            matcher="placeholder",
            metadata_extractor="rules",
            write_output_files=False,
        )
    )

    assert payload["processed_count"] == 2
    assert [item["message"]["message_id"] for item in payload["items"]] == [
        "csv_001",
        "csv_002",
    ]


def test_process_feedback_batch_rejects_invalid_csv_text() -> None:
    with pytest.raises(HTTPException) as exc_info:
        process_feedback_batch_endpoint(
            ProcessFeedbackBatchRequest(
                csv_text="id,text\nbad_001,Missing required columns\n",
                classifier="mock",
                matcher="placeholder",
            )
        )

    assert exc_info.value.status_code == 400
    assert "message_id" in exc_info.value.detail


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
        metadata_extractor="auto",
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


def _patch_batch_processing(monkeypatch) -> dict:
    captured = {}

    def fake_process_feedback_messages(
        feedback_messages,
        classifier,
        matcher,
        hybrid_threshold=None,
        metadata_extractor="auto",
        exercises=None,
    ):
        captured["exercise_count"] = len(exercises or [])
        items = []
        for feedback_message in feedback_messages:
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
                final_action="aggregate_content_request",
            )
            items.append(
                ProcessedFeedback(
                    message=FeedbackMessage(
                        message_id=feedback_message.message_id,
                        user_message=feedback_message.user_message,
                    ),
                    classification=classification,
                    matches=[],
                    decision=decision,
                    classifier_mode=classifier,
                    matcher_mode=matcher,
                    routing="process",
                    top_matches=[],
                    match_status="track_only",
                    decision_reason="Fake decision.",
                    final_action="aggregate_content_request",
                )
            )
        return items

    monkeypatch.setattr(
        api_module,
        "process_feedback_messages",
        fake_process_feedback_messages,
    )
    return captured
