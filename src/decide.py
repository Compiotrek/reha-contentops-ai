from src.schemas import ClassifiedFeedback, DecisionResult, MatchResult


STRONG_EXISTING_CONTENT_THRESHOLD = 0.88
SIMILAR_CONTENT_THRESHOLD = 0.75


def apply_routing(classified: ClassifiedFeedback) -> str:
    if classified.safety_flag:
        return "safety_review"
    if classified.confidence < 0.75:
        return "needs_review"
    return "process"


def decide_match_status(
    classified: ClassifiedFeedback, match_results: list[MatchResult]
) -> DecisionResult:
    routing = apply_routing(classified)

    if routing == "safety_review":
        return DecisionResult(
            routing=routing,
            match_status="safety_review",
            reason="Safety flag requires human review.",
            final_action="route_to_safety_review",
        )
    if routing == "needs_review":
        return DecisionResult(
            routing=routing,
            match_status="needs_review",
            reason="Low confidence classification requires review.",
            final_action="route_to_human_review",
        )
    if "content_request" not in classified.labels:
        return DecisionResult(
            routing=routing,
            match_status="log_only",
            reason="Feedback is not a content request.",
            final_action="log_feedback",
        )
    if not match_results:
        return DecisionResult(
            routing=routing,
            match_status="track_only",
            reason=(
                "No approved exercise match met the similarity threshold. "
                "TODO: repeated similar requests should later become potential_content_gap."
            ),
            final_action="track_request",
        )

    top_match = match_results[0]
    top_score = _match_score(top_match)
    has_key_mismatches = bool(top_match.metadata_mismatches)

    if top_score >= STRONG_EXISTING_CONTENT_THRESHOLD and not has_key_mismatches:
        return DecisionResult(
            routing=routing,
            match_status="existing_content",
            reason="Top approved exercise match is strong and has no key metadata mismatches.",
            final_action="link_existing_content_for_review",
        )
    if top_score >= SIMILAR_CONTENT_THRESHOLD and has_key_mismatches:
        return DecisionResult(
            routing=routing,
            match_status="possible_duplicate",
            reason="Top approved exercise match is similar but has key metadata mismatches.",
            final_action="review_possible_duplicate",
        )
    if top_score >= SIMILAR_CONTENT_THRESHOLD and "metadata_issue" in classified.labels:
        return DecisionResult(
            routing=routing,
            match_status="metadata_issue",
            reason="Similar approved exercise found and feedback indicates metadata may be incomplete.",
            final_action="review_metadata",
        )
    if top_score < SIMILAR_CONTENT_THRESHOLD:
        return DecisionResult(
            routing=routing,
            match_status="track_only",
            reason=(
                "No approved exercise match met the similarity threshold. "
                "TODO: repeated similar requests should later become potential_content_gap."
            ),
            final_action="track_request",
        )

    return DecisionResult(
        routing=routing,
        match_status="needs_review",
        reason="Content request has an ambiguous match and needs human review.",
        final_action="route_to_human_review",
    )


def _match_score(match_result: MatchResult) -> float:
    if match_result.final_score is not None:
        return match_result.final_score
    return match_result.score
