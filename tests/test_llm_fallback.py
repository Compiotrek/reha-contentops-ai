from src.classify import _llm_fallback_classification
from src.schemas import ClassifiedFeedback, FeedbackMessage


def test_llm_fallback_classification_object_is_valid() -> None:
    message = FeedbackMessage(
        message_id="msg_test",
        user_message="Ich verstehe die Übung nicht und brauche Hilfe.",
    )

    classified = _llm_fallback_classification(message)

    assert isinstance(classified, ClassifiedFeedback)
    assert classified.message_id == "msg_test"
    assert classified.labels == ["unclear"]
    assert classified.safety_flag is False
    assert classified.confidence == 0.0
    assert classified.routing == "needs_review"
    assert classified.evidence_quote == message.user_message[:120]
