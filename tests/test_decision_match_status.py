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
        "vector_similarity": None,
        "metadata_fit_score": None,
        "metadata_mismatches": [],
        "reasons": ["no key metadata mismatches"],
    }
    data.update(overrides)
    return MatchResult(**data)


def test_strong_score_without_mismatch_is_existing_content() -> None:
    decision = decide_match_status(_classified(), [_match(final_score=0.9)])

    assert decision.match_status == "existing_content"
    assert decision.review_required is False
    assert decision.final_action == "link_existing_content"


def test_metadata_match_without_mismatch_is_existing_content() -> None:
    decision = decide_match_status(_classified(), [_match(final_score=0.7)])

    assert decision.match_status == "existing_content"
    assert decision.review_required is False
    assert decision.final_action == "link_existing_content"


def test_weak_match_without_mismatch_stays_track_only() -> None:
    decision = decide_match_status(_classified(), [_match(final_score=0.45)])

    assert decision.match_status == "track_only"
    assert decision.review_required is False


def test_embedding_match_without_mismatch_is_existing_content() -> None:
    decision = decide_match_status(
        _classified(),
        [
            _match(
                final_score=0.59,
                vector_similarity=0.59,
            )
        ],
    )

    assert decision.match_status == "existing_content"
    assert decision.final_action == "link_existing_content"
    assert decision.review_required is False


def test_embedding_match_below_similarity_threshold_stays_track_only() -> None:
    decision = decide_match_status(
        _classified(),
        [
            _match(
                final_score=0.57,
                vector_similarity=0.57,
            )
        ],
    )

    assert decision.match_status == "track_only"
    assert decision.review_required is False


def test_weak_embedding_match_with_mismatch_stays_track_only() -> None:
    decision = decide_match_status(
        _classified(),
        [
            _match(
                final_score=0.56,
                vector_similarity=0.6,
                metadata_fit_score=0.55,
                metadata_mismatches=[
                    "equipment mismatch: requested none, exercise requires miniband"
                ],
            )
        ],
    )

    assert decision.match_status == "track_only"
    assert decision.review_required is False


def test_embedding_match_with_strong_metadata_support_can_be_possible_duplicate() -> None:
    decision = decide_match_status(
        _classified(),
        [
            _match(
                final_score=0.66,
                vector_similarity=0.62,
                metadata_fit_score=0.7,
                metadata_mismatches=[
                    "equipment mismatch: requested none, exercise requires miniband"
                ],
            )
        ],
    )

    assert decision.match_status == "possible_duplicate"
    assert decision.review_required is True


def test_embedding_topic_mismatch_stays_track_only_even_with_metadata_support() -> None:
    decision = decide_match_status(
        _classified(),
        [
            _match(
                final_score=0.66,
                vector_similarity=0.62,
                metadata_fit_score=0.8,
                metadata_mismatches=[
                    "topic mismatch: request topic not represented (handstand)",
                    "difficulty mismatch: requested beginner, exercise is intermediate",
                ],
            )
        ],
    )

    assert decision.match_status == "track_only"
    assert decision.review_required is False


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
    assert decision.review_required is True
    assert decision.priority == "medium"


def test_possible_duplicate_without_key_review_condition_is_aggregated() -> None:
    decision = decide_match_status(
        _classified(),
        [
            _match(
                final_score=0.9,
                metadata_mismatches=[
                    "equipment mismatch: requested none, exercise requires miniband"
                ],
            )
        ],
    )

    assert decision.match_status == "possible_duplicate"
    assert decision.review_required is False
    assert decision.final_action == "aggregate_possible_duplicate"


def test_no_content_request_is_log_only() -> None:
    decision = decide_match_status(_classified(labels=["praise"]), [_match()])

    assert decision.match_status == "log_only"
    assert decision.review_required is False
    assert decision.final_action == "aggregate_praise"


def test_safety_routing_is_safety_review() -> None:
    decision = decide_match_status(
        _classified(labels=["safety_signal"], safety_flag=True),
        [_match()],
    )

    assert decision.match_status == "safety_review"
    assert decision.review_required is True
    assert decision.priority == "high"


def test_low_confidence_is_needs_review() -> None:
    decision = decide_match_status(_classified(confidence=0.5), [_match()])

    assert decision.match_status == "needs_review"
    assert decision.review_required is True
    assert decision.priority == "medium"


def test_criticism_without_content_request_is_aggregated() -> None:
    decision = decide_match_status(_classified(labels=["criticism"]), [_match()])

    assert decision.match_status == "log_only"
    assert decision.final_action == "aggregate_criticism"
    assert decision.review_required is False


def test_bug_routes_to_support_without_review() -> None:
    decision = decide_match_status(
        _classified(labels=["bug_or_access_problem"]),
        [_match()],
    )

    assert decision.match_status == "support_queue"
    assert decision.final_action == "route_to_support"
    assert decision.review_required is False
    assert decision.priority == "medium"


def test_track_only_does_not_require_review() -> None:
    decision = decide_match_status(_classified(), [])

    assert decision.match_status == "track_only"
    assert decision.final_action == "aggregate_content_request"
    assert decision.review_required is False


def test_metadata_issue_is_low_priority_review() -> None:
    decision = decide_match_status(_classified(labels=["metadata_issue"]), [_match()])

    assert decision.match_status == "metadata_issue"
    assert decision.review_required is True
    assert decision.priority == "low"
