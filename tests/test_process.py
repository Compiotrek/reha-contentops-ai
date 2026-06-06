from src.output import format_single_message_summary
from src.process import process_feedback, process_feedback_message, process_single_feedback
from src.schemas import ClassifiedFeedback, FeedbackMessage


def test_process_feedback_with_mock_classifier_and_limit() -> None:
    processed_items = process_feedback(classifier="mock", limit=5)

    assert len(processed_items) == 5
    assert all(item.classification.message_id for item in processed_items)
    assert all(item.decision.routing for item in processed_items)
    assert all(len(item.matches) == 3 for item in processed_items)


def test_process_feedback_accepts_custom_input_csv(tmp_path) -> None:
    input_csv = tmp_path / "messages.csv"
    input_csv.write_text(
        "message_id,timestamp,user_message,source\n"
        "msg_custom,2026-06-03 09:00:00,Ich will handstandübungen,slack\n",
        encoding="utf-8",
    )

    processed_items = process_feedback(
        classifier="mock",
        matcher="placeholder",
        input_csv=input_csv,
    )

    assert len(processed_items) == 1
    assert processed_items[0].message.message_id == "msg_custom"
    assert "content_request" in processed_items[0].classification.labels


def test_process_feedback_message_processes_single_cli_message() -> None:
    processed = process_feedback_message(
        user_message="Ich will handstandübungen.",
        message_id="msg_cli",
        classifier="mock",
        matcher="placeholder",
    )

    assert processed.message.message_id == "msg_cli"
    assert processed.message.user_message == "Ich will handstandübungen."
    assert "content_request" in processed.classification.labels
    assert processed.matcher_mode == "placeholder"


def test_single_message_summary_is_concise_without_scores() -> None:
    processed = process_feedback_message(
        user_message="Bitte mehr Knieübungen ohne Geräte.",
        message_id="msg_cli",
        classifier="mock",
        matcher="placeholder",
    )

    summary = format_single_message_summary(processed)

    assert "Labels: content_request" in summary
    assert "Match status: existing_content" in summary
    assert "Review required:" in summary
    assert "Top match:" in summary
    assert "0." not in summary
    assert '"message"' not in summary


def test_process_single_backfills_metadata_for_placeholder_matching() -> None:
    message = FeedbackMessage(
        message_id="msg_test",
        user_message="Bitte mehr leichtere Knieübungen ohne Geräte.",
    )
    captured = {}

    def classifier(feedback_message: FeedbackMessage) -> ClassifiedFeedback:
        return ClassifiedFeedback(
            message_id=feedback_message.message_id,
            user_message=feedback_message.user_message,
            labels=["content_request"],
            safety_flag=False,
            evidence_quote=feedback_message.user_message,
            summary="ML classifier omitted request metadata.",
            confidence=0.9,
            routing="process",
            classifier_source="ml",
        )

    def matcher(classified: ClassifiedFeedback, exercises):
        captured["body_region"] = classified.body_region
        captured["equipment"] = classified.equipment
        captured["difficulty_requested"] = classified.difficulty_requested
        return []

    process_single_feedback(
        feedback_message=message,
        exercises=[],
        classifier_mode="ml",
        matcher_mode="placeholder",
        classifier=classifier,
        matcher_fn=matcher,
    )

    assert captured["body_region"] == "knee"
    assert captured["equipment"] == "none"
    assert captured["difficulty_requested"] == "beginner"


def test_process_single_auto_leaves_embedding_metadata_empty_without_api_key(
    monkeypatch,
) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    message = FeedbackMessage(
        message_id="msg_test",
        user_message="Bitte mehr leichtere Knieübungen ohne Geräte.",
    )
    captured = {}

    def classifier(feedback_message: FeedbackMessage) -> ClassifiedFeedback:
        return ClassifiedFeedback(
            message_id=feedback_message.message_id,
            user_message=feedback_message.user_message,
            labels=["content_request"],
            safety_flag=False,
            evidence_quote=feedback_message.user_message,
            summary="Classifier intentionally omitted request metadata.",
            confidence=0.9,
            routing="process",
            classifier_source="hybrid_rule_request",
        )

    def matcher(classified: ClassifiedFeedback, exercises):
        captured["body_region"] = classified.body_region
        captured["equipment"] = classified.equipment
        captured["difficulty_requested"] = classified.difficulty_requested
        return []

    process_single_feedback(
        feedback_message=message,
        exercises=[],
        classifier_mode="hybrid",
        matcher_mode="embeddings",
        classifier=classifier,
        matcher_fn=matcher,
    )

    assert captured["body_region"] is None
    assert captured["equipment"] is None
    assert captured["difficulty_requested"] is None


def test_process_single_can_use_llm_metadata_extractor_for_ml_embeddings(
    monkeypatch,
) -> None:
    message = FeedbackMessage(
        message_id="msg_test",
        user_message="Bitte mehr leichtere Knieübungen ohne Geräte.",
    )
    captured = {}

    def classifier(feedback_message: FeedbackMessage) -> ClassifiedFeedback:
        return ClassifiedFeedback(
            message_id=feedback_message.message_id,
            user_message=feedback_message.user_message,
            labels=["content_request"],
            safety_flag=False,
            evidence_quote=feedback_message.user_message,
            summary="ML classifier accepted the request label.",
            confidence=0.9,
            routing="process",
            classifier_source="ml",
        )

    def fake_metadata_extractor(classified: ClassifiedFeedback) -> ClassifiedFeedback:
        classified.body_region = "knee"
        classified.equipment = "none"
        classified.difficulty_requested = "beginner"
        classified.metadata_source = "llm_metadata"
        return classified

    def matcher(classified: ClassifiedFeedback, exercises):
        captured["body_region"] = classified.body_region
        captured["equipment"] = classified.equipment
        captured["difficulty_requested"] = classified.difficulty_requested
        captured["metadata_source"] = classified.metadata_source
        return []

    monkeypatch.setattr(
        "src.metadata.extract_request_metadata_llm",
        fake_metadata_extractor,
    )

    process_single_feedback(
        feedback_message=message,
        exercises=[],
        classifier_mode="ml",
        matcher_mode="embeddings",
        classifier=classifier,
        matcher_fn=matcher,
        metadata_extractor="llm",
    )

    assert captured["body_region"] == "knee"
    assert captured["equipment"] == "none"
    assert captured["difficulty_requested"] == "beginner"
    assert captured["metadata_source"] == "llm_metadata"
