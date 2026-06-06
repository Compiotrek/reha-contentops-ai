from src.evaluate_pipeline import _evaluation_rows, _gap_evaluation_rows
from src.schemas import (
    ClassifiedFeedback,
    DecisionResult,
    FeedbackMessage,
    ProcessedFeedback,
)


def test_evaluation_rows_compare_expected_outcomes() -> None:
    item = _processed_item(
        message_id="msg_001",
        labels=["content_request"],
        match_status="track_only",
        final_action="aggregate_content_request",
        review_required=False,
    )
    expected = {
        "msg_001": {
            "expected_labels": "content_request",
            "expected_match_status": "track_only",
            "expected_final_action": "aggregate_content_request",
            "expected_review_required": "false",
            "expected_gap_cluster": "demo_gap",
        }
    }

    rows = _evaluation_rows([item], expected)

    assert rows[0]["label_hit"] is True
    assert rows[0]["match_status_hit"] is True
    assert rows[0]["final_action_hit"] is True
    assert rows[0]["review_required_hit"] is True


def test_gap_evaluation_rows_measure_expected_cluster_overlap() -> None:
    items = [
        _processed_item(
            message_id=f"gap_{index}",
            user_message=f"Ich will Atemübungen Nummer {index}",
            labels=["content_request"],
            match_status="track_only",
            final_action="aggregate_content_request",
            review_required=False,
        )
        for index in range(5)
    ]
    expected = {
        item.message.message_id: {
            "expected_gap_cluster": "breathing",
        }
        for item in items
    }

    rows = _gap_evaluation_rows(items, expected)

    assert rows[0]["expected_gap_cluster"] == "breathing"
    assert rows[0]["expected_count"] == 5
    assert rows[0]["matched_expected_messages"] == 5
    assert rows[0]["gap_hit"] is True


def _processed_item(
    message_id: str,
    labels: list[str],
    match_status: str,
    final_action: str,
    review_required: bool,
    user_message: str = "Test message.",
) -> ProcessedFeedback:
    message = FeedbackMessage(message_id=message_id, user_message=user_message)
    classification = ClassifiedFeedback(
        message_id=message_id,
        user_message=user_message,
        labels=labels,
        safety_flag=False,
        evidence_quote=user_message,
        summary=user_message,
        confidence=0.9,
        routing="process",
    )
    decision = DecisionResult(
        routing="process",
        match_status=match_status,
        reason="Test decision.",
        final_action=final_action,
        review_required=review_required,
        priority="none",
    )
    return ProcessedFeedback(
        message=message,
        classification=classification,
        matches=[],
        decision=decision,
        classifier_mode="mock",
        matcher_mode="placeholder",
        routing=decision.routing,
        top_matches=[],
        match_status=decision.match_status,
        decision_reason=decision.reason,
        final_action=decision.final_action,
        review_required=decision.review_required,
        priority=decision.priority,
    )
