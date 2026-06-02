from src.schemas import ClassifiedFeedback, Exercise, MatchResult


def match_exercises_placeholder(
    classified: ClassifiedFeedback, exercises: list[Exercise]
) -> list[MatchResult]:
    """Return simple metadata matches.

    This intentionally avoids embeddings or vector search. It is a deterministic
    placeholder for later semantic retrieval.
    """
    scored = [_score_exercise(classified, exercise) for exercise in exercises]
    scored.sort(key=lambda item: item.score, reverse=True)
    return scored[:3]


def _score_exercise(classified: ClassifiedFeedback, exercise: Exercise) -> MatchResult:
    score = 0.0
    reasons: list[str] = []

    if classified.body_region and classified.body_region == exercise.body_region:
        score += 0.5
        reasons.append("body_region")
    if classified.equipment and classified.equipment == exercise.equipment:
        score += 0.25
        reasons.append("equipment")
    if classified.difficulty_requested and classified.difficulty_requested == exercise.difficulty:
        score += 0.25
        reasons.append("difficulty")

    return MatchResult(
        exercise_id=exercise.exercise_id,
        title=exercise.title,
        score=round(score, 3),
        reasons=reasons,
    )
