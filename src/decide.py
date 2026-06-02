from src.schemas import ClassifiedFeedback, DecisionResult, MatchResult


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
        )
    if routing == "needs_review":
        return DecisionResult(
            routing=routing,
            match_status="needs_review",
            reason="Low confidence classification requires review.",
        )
    if "content_request" not in classified.labels:
        return DecisionResult(
            routing=routing,
            match_status="log_only",
            reason="Feedback is not a content request.",
        )
    return DecisionResult(
        routing=routing,
        match_status="needs_review",
        reason="Content request needs review before content operations action.",
    )
