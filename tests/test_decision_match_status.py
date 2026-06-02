from src.decide import decide_match_status
from src.schemas import ClassifiedFeedback, MatchResult


def _classified(**overrides) -> ClassifiedFeedback:
    data = {
        "message_id": "msg_test",
        "user_message": "Bitte mehr Knieübungen ohne Geräte.",
        "labels": ["content_request"],
        "safety_flag": False,
        "evidence_quote": "mehr Knieübungen ohne Geräte",
        "summary": "User requests knee exercises without equipment.",
        "confidence": 0.9,
        "routing": "process",
    }
    data.update(overrides)
    return ClassifiedFeedback(**data)


def _match(**overrides) -> MatchResult:
    data = {
        "exercise_id": "ex_test",
        "title": "Knee Exercise",
        "score": 0.9,
        "final_score": 0.9,
        "vector_similarity": 0.9,
        "metadata_fit_score": 1.0,
        "metadata_mismatches": [],
        "reasons": ["no key metadata mismatches"],
    }
    data.update(overrides)
    return MatchResult(**data)


def test_strong_score_without_mismatch_is_existing_content() -> None:
    decision = decide_match_status(_classified(), [_match(final_score=0.9)])

    assert decision.match_status == "existing_content"


def test_high_score_with_equipment_mismatch_is_possible_duplicate() -> None:
    decision = decide_match_status(
        _classified(),
        [
            _match(
                final_score=0.8,
                metadata_mismatches=[
                    "equipment mismatch: requested none, exercise requires miniband"
                ],
            )
        ],
    )

    assert decision.match_status == "possible_duplicate"


def test_no_content_request_is_log_only() -> None:
    decision = decide_match_status(_classified(labels=["praise"]), [_match()])

    assert decision.match_status == "log_only"


def test_safety_routing_is_safety_review() -> None:
    decision = decide_match_status(
        _classified(labels=["safety_signal"], safety_flag=True),
        [_match()],
    )

    assert decision.match_status == "safety_review"


def test_low_confidence_is_needs_review() -> None:
    decision = decide_match_status(_classified(confidence=0.5), [_match()])

    assert decision.match_status == "needs_review"
