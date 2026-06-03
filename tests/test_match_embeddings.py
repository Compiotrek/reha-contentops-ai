from src.match import (
    classified_feedback_to_query_text,
    cosine_similarity,
    detect_metadata_mismatches,
    exercise_to_search_text,
    hybrid_score,
    match_exercises_placeholder,
    metadata_fit_score,
)
from src.schemas import ClassifiedFeedback, Exercise


def _exercise(**overrides) -> Exercise:
    data = {
        "exercise_id": "ex_test",
        "title": "Seated Knee Extension",
        "body_region": "knee",
        "therapy_goal": "strength",
        "difficulty": "beginner",
        "equipment": "chair",
        "position": "seated",
        "description": "Gentle knee extension from a seated position.",
        "tags": "knee;strength;beginner;chair",
        "review_status": "approved",
    }
    data.update(overrides)
    return Exercise(**data)


def _classified(**overrides) -> ClassifiedFeedback:
    data = {
        "message_id": "msg_test",
        "user_message": "Ich hätte gern leichtere Knieübungen ohne Geräte.",
        "labels": ["content_request"],
        "body_region": "knee",
        "therapy_goal": "strength",
        "difficulty_requested": "beginner",
        "equipment": "none",
        "position": "seated",
        "sentiment": "neutral",
        "safety_flag": False,
        "evidence_quote": "leichtere Knieübungen ohne Geräte",
        "summary": "User requests beginner knee exercises without equipment.",
        "confidence": 0.9,
        "routing": "process",
    }
    data.update(overrides)
    return ClassifiedFeedback(**data)


def test_cosine_similarity() -> None:
    assert cosine_similarity([1, 0], [1, 0]) == 1.0
    assert cosine_similarity([1, 0], [0, 1]) == 0.0
    assert cosine_similarity([0, 0], [1, 0]) == 0.0


def test_exercise_to_search_text_includes_core_metadata() -> None:
    search_text = exercise_to_search_text(_exercise())

    assert "title: Seated Knee Extension" in search_text
    assert "body_region: knee" in search_text
    assert "equipment: chair" in search_text
    assert "review_status: approved" in search_text


def test_classified_feedback_to_query_text_includes_request_fields() -> None:
    query_text = classified_feedback_to_query_text(_classified())

    assert "User requests beginner knee exercises without equipment." in query_text
    assert "Body region: knee." in query_text
    assert "Equipment: none." in query_text
    assert "Evidence: leichtere Knieübungen ohne Geräte." in query_text


def test_detect_metadata_mismatches() -> None:
    mismatches = detect_metadata_mismatches(_classified(), _exercise())

    assert "equipment mismatch: requested none, exercise requires chair" in mismatches


def test_hybrid_score_calculation() -> None:
    fit_score = metadata_fit_score(["equipment mismatch: requested none, exercise requires chair"])

    assert fit_score == 0.75
    assert hybrid_score(vector_similarity=0.8, fit_score=fit_score) == 0.788


def test_placeholder_matcher_prefers_fewer_mismatches_on_tie() -> None:
    classified = _classified(
        body_region="general",
        difficulty_requested="beginner",
        equipment=None,
        position="standing",
    )
    chair_exercise = _exercise(
        exercise_id="ex_chair",
        title="Chair General Demo",
        body_region="general",
        difficulty="beginner",
        position="sitting",
    )
    standing_exercise = _exercise(
        exercise_id="ex_standing",
        title="Standing General Demo",
        body_region="general",
        difficulty="beginner",
        position="standing",
        equipment="none",
    )

    matches = match_exercises_placeholder(
        classified, [chair_exercise, standing_exercise]
    )

    assert matches[0].exercise_id == "ex_standing"


def test_placeholder_matcher_score_does_not_exceed_one_with_position_match() -> None:
    classified = _classified(equipment="chair", position="sitting")
    exercise = _exercise(position="sitting")

    matches = match_exercises_placeholder(classified, [exercise])

    assert matches[0].score <= 1.0
    assert matches[0].final_score <= 1.0
