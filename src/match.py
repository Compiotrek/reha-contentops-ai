import hashlib
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
from dotenv import load_dotenv

from src.schemas import ClassifiedFeedback, Exercise, MatchResult


ROOT = Path(__file__).resolve().parents[1]
EMBEDDING_CACHE_PATH = ROOT / "outputs" / "exercise_embeddings.json"
DEFAULT_EMBEDDING_MODEL = "text-embedding-3-small"


def match_exercises_placeholder(
    classified: ClassifiedFeedback, exercises: list[Exercise]
) -> list[MatchResult]:
    """Return simple metadata matches.

    This intentionally avoids embeddings or vector search. It is a deterministic
    placeholder for later semantic retrieval.
    """
    scored = [_score_exercise(classified, exercise) for exercise in exercises]
    scored.sort(key=_placeholder_sort_key, reverse=True)
    return scored[:3]


def exercise_to_search_text(exercise: Exercise) -> str:
    tags = exercise.tags.replace(";", ", ")
    return (
        f"title: {exercise.title};\n"
        f"body_region: {exercise.body_region};\n"
        f"indication: {exercise.indication};\n"
        f"therapy_goal: {exercise.therapy_goal};\n"
        f"difficulty: {exercise.difficulty};\n"
        f"equipment: {exercise.equipment};\n"
        f"position: {exercise.position};\n"
        f"description: {exercise.description};\n"
        f"tags: {tags};\n"
        f"contraindication_note: {exercise.contraindication_note};\n"
        f"review_status: {exercise.review_status}"
    )


def classified_feedback_to_query_text(classified: ClassifiedFeedback) -> str:
    parts = [
        classified.summary,
        f"Labels: {', '.join(classified.labels)}.",
    ]
    if classified.body_region:
        parts.append(f"Body region: {classified.body_region}.")
    if classified.therapy_goal:
        parts.append(f"Therapy goal: {classified.therapy_goal}.")
    if classified.difficulty_requested:
        parts.append(f"Difficulty: {classified.difficulty_requested}.")
    if classified.equipment:
        parts.append(f"Equipment: {classified.equipment}.")
    if classified.position:
        parts.append(f"Position: {classified.position}.")
    if classified.evidence_quote:
        parts.append(f"Evidence: {classified.evidence_quote}.")
    return " ".join(parts)


def get_embedding(text: str) -> list[float]:
    return get_embeddings([text])[0]


def get_embeddings(texts: list[str]) -> list[list[float]]:
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "OPENAI_API_KEY is required for --matcher embeddings. "
            "Use --matcher placeholder for local runs without OpenAI credentials."
        )

    from openai import OpenAI

    model = os.getenv("OPENAI_EMBEDDING_MODEL") or DEFAULT_EMBEDDING_MODEL
    client = OpenAI(api_key=api_key)
    response = client.embeddings.create(model=model, input=texts)
    return [item.embedding for item in response.data]


def cosine_similarity(vec_a, vec_b) -> float:
    array_a = np.asarray(vec_a, dtype=float)
    array_b = np.asarray(vec_b, dtype=float)
    norm_a = np.linalg.norm(array_a)
    norm_b = np.linalg.norm(array_b)
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return float(np.dot(array_a, array_b) / (norm_a * norm_b))


def detect_metadata_mismatches(
    classified: ClassifiedFeedback, exercise: Exercise
) -> list[str]:
    mismatches: list[str] = []

    if classified.body_region and classified.body_region != exercise.body_region:
        mismatches.append(
            f"body_region mismatch: requested {classified.body_region}, "
            f"exercise is {exercise.body_region}"
        )
    if classified.equipment and classified.equipment != exercise.equipment:
        mismatches.append(
            f"equipment mismatch: requested {classified.equipment}, "
            f"exercise requires {exercise.equipment}"
        )
    if (
        classified.difficulty_requested
        and classified.difficulty_requested != exercise.difficulty
    ):
        mismatches.append(
            f"difficulty mismatch: requested {classified.difficulty_requested}, "
            f"exercise is {exercise.difficulty}"
        )
    if classified.position and classified.position != exercise.position:
        mismatches.append(
            f"position mismatch: requested {classified.position}, "
            f"exercise position is {exercise.position}"
        )
    if classified.therapy_goal and classified.therapy_goal != exercise.therapy_goal:
        mismatches.append(
            f"therapy_goal mismatch: requested {classified.therapy_goal}, "
            f"exercise goal is {exercise.therapy_goal}"
        )

    return mismatches


def metadata_fit_score(metadata_mismatches: list[str]) -> float:
    key_mismatch_count = sum(
        1
        for mismatch in metadata_mismatches
        if mismatch.startswith(("body_region", "equipment", "difficulty", "position"))
    )
    return max(0.0, min(1.0, 1.0 - (0.25 * key_mismatch_count)))


def hybrid_score(vector_similarity: float, fit_score: float) -> float:
    return round((0.75 * vector_similarity) + (0.25 * fit_score), 3)


def match_exercises_embeddings(
    classified: ClassifiedFeedback,
    exercises: list[Exercise],
    top_k: int = 3,
) -> list[MatchResult]:
    if "content_request" not in classified.labels:
        return []

    approved_exercises = [
        exercise for exercise in exercises if exercise.review_status == "approved"
    ]
    if not approved_exercises:
        return []

    query_text = classified_feedback_to_query_text(classified)
    exercise_embeddings = _load_or_create_exercise_embeddings(approved_exercises)
    query_embedding = get_embedding(query_text)

    exercise_by_id = {exercise.exercise_id: exercise for exercise in approved_exercises}
    results: list[MatchResult] = []
    for cached_item in exercise_embeddings:
        exercise = exercise_by_id.get(cached_item["exercise_id"])
        if exercise is None:
            continue
        vector_similarity = round(
            cosine_similarity(query_embedding, cached_item["embedding"]),
            3,
        )
        mismatches = detect_metadata_mismatches(classified, exercise)
        fit_score = metadata_fit_score(mismatches)
        final_score = hybrid_score(vector_similarity, fit_score)
        reasons = _embedding_match_reasons(vector_similarity, fit_score, mismatches)

        results.append(
            MatchResult(
                exercise_id=exercise.exercise_id,
                title=exercise.title,
                score=final_score,
                reasons=reasons,
                vector_similarity=vector_similarity,
                metadata_fit_score=fit_score,
                final_score=final_score,
                metadata_mismatches=mismatches,
                reason="; ".join(reasons),
            )
        )

    results.sort(key=lambda item: item.final_score or item.score, reverse=True)
    return results[:top_k]


def _score_exercise(classified: ClassifiedFeedback, exercise: Exercise) -> MatchResult:
    score = 0.0
    reasons: list[str] = []

    if classified.body_region and classified.body_region == exercise.body_region:
        score += 0.45
        reasons.append("body_region")
    if classified.equipment and classified.equipment == exercise.equipment:
        score += 0.25
        reasons.append("equipment")
    if classified.difficulty_requested and classified.difficulty_requested == exercise.difficulty:
        score += 0.2
        reasons.append("difficulty")
    if classified.position and classified.position == exercise.position:
        score += 0.1
        reasons.append("position")
    score = min(score, 1.0)

    return MatchResult(
        exercise_id=exercise.exercise_id,
        title=exercise.title,
        score=round(score, 3),
        reasons=reasons,
        final_score=round(score, 3),
        metadata_mismatches=detect_metadata_mismatches(classified, exercise),
        reason=", ".join(reasons) if reasons else "No metadata fields matched.",
    )


def _placeholder_sort_key(match_result: MatchResult) -> tuple[float, int, int]:
    return (
        match_result.score,
        -len(match_result.metadata_mismatches),
        len(match_result.reasons),
    )


def _load_or_create_exercise_embeddings(exercises: list[Exercise]) -> list[dict[str, Any]]:
    load_dotenv()
    model = os.getenv("OPENAI_EMBEDDING_MODEL") or DEFAULT_EMBEDDING_MODEL
    cache = _load_embedding_cache()
    cached_items = cache.get("items", []) if cache.get("embedding_model") == model else []
    cache_lookup = {
        (item.get("exercise_id"), item.get("text_hash")): item for item in cached_items
    }

    reused_items: list[dict[str, Any]] = []
    missing_exercises: list[tuple[Exercise, str, str]] = []
    for exercise in exercises:
        search_text = exercise_to_search_text(exercise)
        text_hash = _text_hash(search_text)
        cached_item = cache_lookup.get((exercise.exercise_id, text_hash))
        if cached_item:
            reused_items.append(cached_item)
        else:
            missing_exercises.append((exercise, search_text, text_hash))

    new_items: list[dict[str, Any]] = []
    if missing_exercises:
        embeddings = get_embeddings([item[1] for item in missing_exercises])
        for (exercise, search_text, text_hash), embedding in zip(
            missing_exercises, embeddings
        ):
            new_items.append(
                {
                    "exercise_id": exercise.exercise_id,
                    "text_hash": text_hash,
                    "search_text": search_text,
                    "embedding": embedding,
                }
            )

    items_by_id = {
        item["exercise_id"]: item for item in [*reused_items, *new_items]
    }
    ordered_items = [
        items_by_id[exercise.exercise_id]
        for exercise in exercises
        if exercise.exercise_id in items_by_id
    ]
    _write_embedding_cache({"embedding_model": model, "items": ordered_items})
    return ordered_items


def _load_embedding_cache() -> dict[str, Any]:
    if not EMBEDDING_CACHE_PATH.exists():
        return {"embedding_model": None, "items": []}
    try:
        return json.loads(EMBEDDING_CACHE_PATH.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {"embedding_model": None, "items": []}


def _write_embedding_cache(cache: dict[str, Any]) -> None:
    EMBEDDING_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    EMBEDDING_CACHE_PATH.write_text(
        json.dumps(cache, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _text_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _embedding_match_reasons(
    vector_similarity: float,
    fit_score: float,
    mismatches: list[str],
) -> list[str]:
    reasons = [
        f"vector similarity {vector_similarity}",
        f"metadata fit {fit_score}",
    ]
    if mismatches:
        reasons.extend(mismatches)
    else:
        reasons.append("no key metadata mismatches")
    return reasons
