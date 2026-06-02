from src.process import process_feedback


def test_process_feedback_with_mock_classifier_and_limit() -> None:
    processed_items = process_feedback(classifier="mock", limit=5)

    assert len(processed_items) == 5
    assert all(item.classification.message_id for item in processed_items)
    assert all(item.decision.routing for item in processed_items)
    assert all(len(item.matches) == 3 for item in processed_items)
