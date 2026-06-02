from src.report import generate_daily_report
from src.schemas import (
    ClassifiedFeedback,
    DecisionResult,
    FeedbackMessage,
    ProcessedFeedback,
)


def test_daily_report_note_for_mock_classifier(tmp_path) -> None:
    output_path = tmp_path / "report.md"

    generate_daily_report([], str(output_path), classifier="mock")

    assert (
        "This report uses deterministic counts from processed mock-classified data."
        in output_path.read_text(encoding="utf-8")
    )


def test_daily_report_note_for_llm_classifier(tmp_path) -> None:
    output_path = tmp_path / "report.md"

    generate_daily_report([], str(output_path), classifier="llm")

    assert (
        "This report uses deterministic counts from processed LLM-classified data."
        in output_path.read_text(encoding="utf-8")
    )


def test_daily_report_hybrid_overview_counts_llm_processed(tmp_path) -> None:
    output_path = tmp_path / "report.md"
    processed_items = [
        _processed_item("msg_001", classifier_source="hybrid_llm"),
        _processed_item("msg_002", classifier_source="hybrid_ml"),
        _processed_item("msg_003", classifier_source="ml_abstain"),
    ]

    generate_daily_report(processed_items, str(output_path), classifier="hybrid")

    report = output_path.read_text(encoding="utf-8")
    assert "- hybrid_llm_processed: 1" in report
    assert "- hybrid_ml_processed: 1" in report
    assert "- ml_abstained: 1" in report


def _processed_item(message_id: str, classifier_source: str) -> ProcessedFeedback:
    message = FeedbackMessage(message_id=message_id, user_message="Bitte mehr Übungen.")
    classification = ClassifiedFeedback(
        message_id=message_id,
        user_message=message.user_message,
        labels=["content_request"],
        safety_flag=False,
        evidence_quote=message.user_message,
        summary="Test classification.",
        confidence=0.9,
        routing="process",
        classifier_source=classifier_source,
    )
    decision = DecisionResult(
        routing="process",
        match_status="track_only",
        reason="Test decision.",
        final_action="track_request",
    )
    return ProcessedFeedback(
        message=message,
        classification=classification,
        matches=[],
        decision=decision,
        classifier_mode="hybrid",
        matcher_mode="placeholder",
        routing="process",
        top_matches=[],
        match_status="track_only",
        decision_reason="Test decision.",
        final_action="track_request",
    )
