import csv

from src import output as output_module
from src.output import write_outputs
from src.schemas import (
    ClassifiedFeedback,
    DecisionResult,
    FeedbackMessage,
    MatchResult,
    ProcessedFeedback,
)


def test_review_queue_excludes_items_without_required_review(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(output_module, "OUTPUT_DIR", tmp_path)
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
    assert rows[0]["review_required"] == "true"
    assert rows[0]["priority"] == "medium"


def test_review_queue_includes_request_metadata(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(output_module, "OUTPUT_DIR", tmp_path)
    item = _processed_item(
        "msg_review",
        DecisionResult(
            routing="needs_review",
            match_status="needs_review",
            reason="Needs review.",
            final_action="route_to_human_review",
            review_required=True,
            priority="medium",
        ),
    )
    item.classification.body_region = "knee"
    item.classification.therapy_goal = "strength"
    item.classification.difficulty_requested = "beginner"
    item.classification.equipment = "none"
    item.classification.position = "standing"
    item.classification.request_theme = "knee beginner no equipment"

    write_outputs([item], classifier="mock", matcher="placeholder")

    with (tmp_path / "review_queue.csv").open(encoding="utf-8", newline="") as file:
        rows = list(csv.DictReader(file))

    assert rows[0]["body_region"] == "knee"
    assert rows[0]["therapy_goal"] == "strength"
    assert rows[0]["difficulty_requested"] == "beginner"
    assert rows[0]["equipment"] == "none"
    assert rows[0]["position"] == "standing"
    assert rows[0]["request_theme"] == "knee beginner no equipment"


def test_content_ops_decisions_csv_includes_every_processed_item(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setattr(output_module, "OUTPUT_DIR", tmp_path)
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
    assert "abstain_reason" in rows[0]


def test_content_ops_decisions_csv_includes_top_match_fields(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setattr(output_module, "OUTPUT_DIR", tmp_path)
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


def test_content_ops_decisions_csv_includes_abstain_reason(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setattr(output_module, "OUTPUT_DIR", tmp_path)
    item = _processed_item(
        "msg_abstain",
        DecisionResult(
            routing="needs_review",
            match_status="needs_review",
            reason="Needs review.",
            final_action="route_to_human_review",
            review_required=True,
            priority="medium",
        ),
    )
    item.classification.abstain_reason = (
        "ML classifier abstained; LLM fallback failed: APIConnectionError"
    )

    write_outputs([item], classifier="hybrid", matcher="embeddings")

    with (tmp_path / "content_ops_decisions.csv").open(
        encoding="utf-8", newline=""
    ) as file:
        rows = list(csv.DictReader(file))

    assert rows[0]["abstain_reason"] == (
        "ML classifier abstained; LLM fallback failed: APIConnectionError"
    )


def test_write_outputs_creates_content_gap_alert_csv(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(output_module, "OUTPUT_DIR", tmp_path)
    items = [
        _processed_item(
            f"gap_{index}",
            DecisionResult(
                routing="process",
                match_status="track_only",
                reason="Track repeated request.",
                final_action="aggregate_content_request",
                review_required=False,
                priority="none",
            ),
            labels=["content_request"],
            user_message=f"Ich will Handstand Übung Nummer {index}",
        )
        for index in range(5)
    ]

    write_outputs(items, classifier="hybrid", matcher="embeddings")

    with (tmp_path / "content_gap_alerts.csv").open(
        encoding="utf-8", newline=""
    ) as file:
        rows = list(csv.DictReader(file))

    assert len(rows) == 1
    assert rows[0]["request_count"] == "5"
    assert rows[0]["recommended_action"] == "review_content_gap"
    assert "gap_0" in rows[0]["message_ids"]


def _processed_item(
    message_id: str,
    decision: DecisionResult,
    matches: list[MatchResult] | None = None,
    labels: list[str] | None = None,
    user_message: str = "Test message.",
) -> ProcessedFeedback:
    message = FeedbackMessage(message_id=message_id, user_message=user_message)
    classification = ClassifiedFeedback(
        message_id=message_id,
        user_message=message.user_message,
        labels=labels or ["praise"],
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
