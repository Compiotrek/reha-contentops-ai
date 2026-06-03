import csv

from src import process as process_module
from src.process import write_outputs
from src.schemas import (
    ClassifiedFeedback,
    DecisionResult,
    FeedbackMessage,
    MatchResult,
    ProcessedFeedback,
)


def test_review_queue_excludes_items_without_required_review(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(process_module, "OUTPUT_DIR", tmp_path)
    items = [
        _processed_item(
            "msg_review",
            DecisionResult(
                routing="needs_review",
                match_status="needs_review",
                reason="Needs review.",
                final_action="route_to_human_review",
                review_required=True,
                priority="medium",
            ),
        ),
        _processed_item(
            "msg_log",
            DecisionResult(
                routing="process",
                match_status="log_only",
                reason="Aggregate only.",
                final_action="aggregate_praise",
                review_required=False,
                priority="none",
            ),
        ),
    ]

    write_outputs(items, classifier="mock", matcher="placeholder")

    with (tmp_path / "review_queue.csv").open(encoding="utf-8", newline="") as file:
        rows = list(csv.DictReader(file))

    assert [row["message_id"] for row in rows] == ["msg_review"]
    assert rows[0]["review_required"] == "True"
    assert rows[0]["priority"] == "medium"


def test_content_ops_decisions_csv_includes_every_processed_item(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setattr(process_module, "OUTPUT_DIR", tmp_path)
    items = [
        _processed_item(
            "msg_review",
            DecisionResult(
                routing="needs_review",
                match_status="needs_review",
                reason="Needs review.",
                final_action="route_to_human_review",
                review_required=True,
                priority="medium",
            ),
        ),
        _processed_item(
            "msg_log",
            DecisionResult(
                routing="process",
                match_status="log_only",
                reason="Aggregate only.",
                final_action="aggregate_praise",
                review_required=False,
                priority="none",
            ),
        ),
    ]

    write_outputs(items, classifier="mock", matcher="placeholder")

    with (tmp_path / "content_ops_decisions.csv").open(
        encoding="utf-8", newline=""
    ) as file:
        rows = list(csv.DictReader(file))

    assert [row["message_id"] for row in rows] == ["msg_review", "msg_log"]
    assert rows[0]["review_required"] == "true"
    assert rows[1]["review_required"] == "false"


def test_content_ops_decisions_csv_includes_top_match_fields(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setattr(process_module, "OUTPUT_DIR", tmp_path)
    match = MatchResult(
        exercise_id="ex_001",
        title="Demo Knee Stability",
        score=0.82,
        vector_similarity=0.8,
        metadata_fit_score=0.75,
        final_score=0.82,
        metadata_mismatches=[
            "equipment mismatch: requested none, exercise requires miniband"
        ],
    )
    items = [
        _processed_item(
            "msg_match",
            DecisionResult(
                routing="process",
                match_status="possible_duplicate",
                reason="Similar content with metadata mismatch.",
                final_action="aggregate_possible_duplicate",
                review_required=False,
                priority="none",
            ),
            matches=[match],
        )
    ]

    write_outputs(items, classifier="mock", matcher="placeholder")

    with (tmp_path / "content_ops_decisions.csv").open(
        encoding="utf-8", newline=""
    ) as file:
        rows = list(csv.DictReader(file))

    assert rows[0]["top_match_exercise_id"] == "ex_001"
    assert rows[0]["top_match_title"] == "Demo Knee Stability"
    assert rows[0]["top_match_score"] == "0.82"
    assert rows[0]["top_match_vector_similarity"] == "0.8"
    assert rows[0]["top_match_metadata_fit_score"] == "0.75"
    assert rows[0]["top_match_final_score"] == "0.82"
    assert (
        rows[0]["top_match_metadata_mismatches"]
        == "equipment mismatch: requested none, exercise requires miniband"
    )


def _processed_item(
    message_id: str,
    decision: DecisionResult,
    matches: list[MatchResult] | None = None,
) -> ProcessedFeedback:
    message = FeedbackMessage(message_id=message_id, user_message="Test message.")
    classification = ClassifiedFeedback(
        message_id=message_id,
        user_message=message.user_message,
        labels=["praise"],
        safety_flag=False,
        evidence_quote=message.user_message,
        summary="Test classification.",
        confidence=0.9,
        routing=decision.routing,
    )
    top_matches = matches or []
    return ProcessedFeedback(
        message=message,
        classification=classification,
        matches=top_matches,
        decision=decision,
        classifier_mode="mock",
        matcher_mode="placeholder",
        routing=decision.routing,
        top_matches=top_matches,
        match_status=decision.match_status,
        decision_reason=decision.reason,
        final_action=decision.final_action,
        review_required=decision.review_required,
        priority=decision.priority,
    )
