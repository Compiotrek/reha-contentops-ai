from src.decide import apply_routing
from src.schemas import ClassifiedFeedback


def _classified(safety_flag: bool, confidence: float) -> ClassifiedFeedback:
    return ClassifiedFeedback(
        message_id="msg_test",
        user_message="Test feedback",
        labels=["safety_signal"] if safety_flag else ["content_request"],
        safety_flag=safety_flag,
        evidence_quote="Test feedback",
        summary="Test classification",
        confidence=confidence,
        routing="needs_review",
    )


def test_safety_flag_routes_to_safety_review() -> None:
    classified = _classified(safety_flag=True, confidence=0.9)

    assert apply_routing(classified) == "safety_review"


def test_low_confidence_routes_to_needs_review() -> None:
    classified = _classified(safety_flag=False, confidence=0.5)

    assert apply_routing(classified) == "needs_review"


def test_high_confidence_without_safety_routes_to_process() -> None:
    classified = _classified(safety_flag=False, confidence=0.9)

    assert apply_routing(classified) == "process"
