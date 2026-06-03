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
    labels = set(classified.labels)

    if routing == "safety_review":
        return DecisionResult(
            routing=routing,
            match_status="safety_review",
            reason="Safety flag requires human review.",
            final_action="route_to_safety_review",
            review_required=True,
            priority="high",
        )
    if labels == {"praise"}:
        return DecisionResult(
            routing=routing,
            match_status="log_only",
            reason="Clear praise is aggregated without human review.",
            final_action="aggregate_praise",
            review_required=False,
            priority="none",
        )
    if labels == {"criticism"}:
        return DecisionResult(
            routing=routing,
            match_status="log_only",
            reason="Criticism without a content request is aggregated without human review.",
            final_action="aggregate_criticism",
            review_required=False,
            priority="none",
        )
    if "bug_or_access_problem" in labels:
        return DecisionResult(
            routing=routing,
            match_status="support_queue",
            reason="Technical or access issue should be routed to support.",
            final_action="route_to_support",
            review_required=False,
            priority="medium",
        )
    if "metadata_issue" in labels:
        return DecisionResult(
            routing=routing,
            match_status="metadata_issue",
            reason="Metadata or discoverability issue needs low-priority content operations review.",
            final_action="review_metadata",
            review_required=True,
            priority="low",
        )
    if routing == "needs_review":
        return DecisionResult(
            routing=routing,
            match_status="needs_review",
            reason="Low confidence classification requires review.",
            final_action="route_to_human_review",
            review_required=True,
            priority="medium",
        )
    if "content_request" not in labels:
        return DecisionResult(
            routing=routing,
            match_status="log_only",
            reason="Feedback is not a content request.",
            final_action="log_feedback",
            review_required=False,
            priority="none",
        )
    if not match_results:
        return DecisionResult(
            routing=routing,
            match_status="track_only",
            reason=(
                "No approved exercise match met the similarity threshold. "
                "TODO: repeated similar requests should later become potential_content_gap."
            ),
            final_action="aggregate_content_request",
            review_required=False,
            priority="none",
        )

    top_match = match_results[0]
    top_score = _match_score(top_match)
    has_key_mismatches = bool(top_match.metadata_mismatches)

    if top_score >= STRONG_EXISTING_CONTENT_THRESHOLD and not has_key_mismatches:
        return DecisionResult(
            routing=routing,
            match_status="existing_content",
            reason="Top approved exercise match is strong and has no key metadata mismatches.",
            final_action="link_existing_content",
            review_required=False,
            priority="none",
        )
    if top_score >= SIMILAR_CONTENT_THRESHOLD and has_key_mismatches:
        review_required = _possible_duplicate_requires_review(
            classified, top_match, top_score
        )
        return DecisionResult(
            routing=routing,
            match_status="possible_duplicate",
            reason=_possible_duplicate_reason(review_required),
            final_action=(
                "review_possible_duplicate"
                if review_required
                else "aggregate_possible_duplicate"
            ),
            review_required=review_required,
            priority="medium" if review_required else "none",
        )
    if top_score < SIMILAR_CONTENT_THRESHOLD:
        return DecisionResult(
            routing=routing,
            match_status="track_only",
            reason=(
                "No approved exercise match met the similarity threshold. "
                "TODO: repeated similar requests should later become potential_content_gap."
            ),
            final_action="aggregate_content_request",
            review_required=False,
            priority="none",
        )

    return DecisionResult(
        routing=routing,
        match_status="track_only",
        reason="Content request is tracked without immediate human review.",
        final_action="aggregate_content_request",
        review_required=False,
        priority="none",
    )


def _match_score(match_result: MatchResult) -> float:
    if match_result.final_score is not None:
        return match_result.final_score
    return match_result.score


def _possible_duplicate_requires_review(
    classified: ClassifiedFeedback, top_match: MatchResult, top_score: float
) -> bool:
    return (
        0.70 <= top_score <= 0.82
        and _has_equipment_or_difficulty_mismatch(top_match)
    ) or classified.confidence < 0.75


def _has_equipment_or_difficulty_mismatch(match_result: MatchResult) -> bool:
    return any(
        mismatch.startswith(("equipment mismatch", "difficulty mismatch"))
        for mismatch in match_result.metadata_mismatches
    )


def _possible_duplicate_reason(review_required: bool) -> str:
    if review_required:
        return (
            "Similar approved content has an equipment or difficulty mismatch "
            "and needs targeted review."
        )
    return "Similar approved content is aggregated as a possible duplicate without review."
