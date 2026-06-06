from src.match import (
    classified_feedback_to_query_text,
    cosine_similarity,
    detect_metadata_mismatches,
    detect_topic_mismatches,
    exercise_to_search_text,
    hybrid_score,
    match_exercises_embeddings,
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


def test_detect_topic_mismatches_for_new_request_topic() -> None:
    classified = _classified(
        user_message="Ich suche eine Handstand Vorbereitung an der Wand.",
        body_region="shoulder",
        therapy_goal="strength",
        difficulty_requested="beginner",
        equipment="wall",
        position="standing",
    )
    exercise = _exercise(
        title="Shoulder Stability At Wall",
        body_region="shoulder",
        therapy_goal="stability",
        difficulty="intermediate",
        equipment="wall",
        position="standing",
        description="Wall-based shoulder stability demo content.",
        tags="shoulder;wall;stability",
    )

    mismatches = detect_topic_mismatches(classified, exercise)

    assert any(mismatch.startswith("topic mismatch") for mismatch in mismatches)
    assert "handstand" in mismatches[0]


def test_detect_topic_mismatches_ignores_metadata_covered_terms() -> None:
    classified = _classified(
        user_message="Ich brauche Knieübungen ohne Geräte im Liegen.",
        body_region="knee",
        therapy_goal=None,
        difficulty_requested=None,
        equipment="none",
        position="lying",
    )
    exercise = _exercise(
        title="Knee No Equipment Lying Demo",
        body_region="knee",
        therapy_goal="stability",
        difficulty="beginner",
        equipment="none",
        position="lying",
        description="Lying knee demo without equipment.",
        tags="knee;lying;none",
    )

    assert detect_topic_mismatches(classified, exercise) == []


def test_hybrid_score_calculation() -> None:
    fit_score = metadata_fit_score(["equipment mismatch: requested none, exercise requires chair"])

    assert fit_score == 0.75
    assert hybrid_score(vector_similarity=0.8, fit_score=fit_score) == 0.782


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


def test_placeholder_matcher_skips_non_content_requests() -> None:
    classified = _classified(labels=["praise"])

    matches = match_exercises_placeholder(classified, [_exercise()])

    assert matches == []


def test_embedding_matcher_uses_metadata_support_as_guard(monkeypatch) -> None:
    classified = _classified(
        body_region="knee",
        therapy_goal=None,
        difficulty_requested=None,
        equipment="none",
        position=None,
    )
    exact_metadata = _exercise(
        exercise_id="ex_exact",
        title="Knee No Equipment",
        body_region="knee",
        equipment="none",
    )
    higher_vector_wrong_metadata = _exercise(
        exercise_id="ex_wrong",
        title="Generic High Vector",
        body_region="general",
        equipment="chair",
    )

    monkeypatch.setattr("src.match.get_embedding", lambda text: [1.0, 0.0])
    monkeypatch.setattr(
        "src.match._load_or_create_exercise_embeddings",
        lambda exercises: [
            {"exercise_id": "ex_exact", "embedding": [0.52, 0.854]},
            {"exercise_id": "ex_wrong", "embedding": [0.56, 0.828]},
        ],
    )

    matches = match_exercises_embeddings(
        classified,
        [exact_metadata, higher_vector_wrong_metadata],
    )

    assert matches[0].exercise_id == "ex_exact"
    assert matches[0].vector_similarity == 0.52
    assert matches[0].metadata_fit_score == 0.7
    assert matches[0].final_score > matches[1].final_score


def test_embedding_hybrid_score_does_not_over_accept_weak_metadata() -> None:
    assert hybrid_score(vector_similarity=0.595, fit_score=0.25) == 0.474


def test_embedding_matcher_marks_uncovered_request_topic(monkeypatch) -> None:
    classified = _classified(
        user_message="Ich suche eine Handstand Vorbereitung an der Wand.",
        body_region="shoulder",
        therapy_goal="strength",
        difficulty_requested="beginner",
        equipment="wall",
        position="standing",
    )
    shoulder_wall = _exercise(
        exercise_id="ex_wall",
        title="Shoulder Stability At Wall",
        body_region="shoulder",
        therapy_goal="stability",
        difficulty="intermediate",
        equipment="wall",
        position="standing",
        description="Wall-based shoulder stability demo content.",
        tags="shoulder;wall;stability",
    )

    monkeypatch.setattr("src.match.get_embedding", lambda text: [1.0, 0.0])
    monkeypatch.setattr(
        "src.match._load_or_create_exercise_embeddings",
        lambda exercises: [{"exercise_id": "ex_wall", "embedding": [0.64, 0.768]}],
    )

    matches = match_exercises_embeddings(classified, [shoulder_wall])

    assert matches[0].metadata_fit_score == 0.8
    assert any(
        mismatch.startswith("topic mismatch")
        for mismatch in matches[0].metadata_mismatches
    )
