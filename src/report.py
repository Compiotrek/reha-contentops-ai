from collections import Counter
from pathlib import Path

from src.schemas import ProcessedFeedback


def generate_daily_report(
    processed_items: list[ProcessedFeedback],
    output_path: str,
    classifier: str | None = None,
    matcher: str | None = None,
) -> None:
    """Generate a deterministic Markdown report from processed pipeline data."""
    total_messages = len(processed_items)
    label_counts: Counter[str] = Counter()
    routing_counts: Counter[str] = Counter()
    match_status_counts: Counter[str] = Counter()
    classifier_source_counts: Counter[str] = Counter()
    final_action_counts: Counter[str] = Counter()
    priority_counts: Counter[str] = Counter()

    for item in processed_items:
        label_counts.update(item.classification.labels)
        routing_counts.update([item.decision.routing])
        match_status_counts.update([item.decision.match_status])
        final_action_counts.update([item.decision.final_action])
        priority_counts.update([item.decision.priority])
        if item.classification.classifier_source:
            classifier_source_counts.update([item.classification.classifier_source])

    content_requests = [
        item.message.message_id
        for item in processed_items
        if "content_request" in item.classification.labels
    ]
    safety_reviews = [
        item.message.message_id
        for item in processed_items
        if item.decision.routing == "safety_review"
    ]
    needs_review = [
        item.message.message_id
        for item in processed_items
        if item.decision.review_required
    ]
    review_required_count = sum(
        1 for item in processed_items if item.decision.review_required
    )
    no_review_count = total_messages - review_required_count

    lines = [
        "# Daily Content Ops Report",
        "",
        _report_note(classifier),
        "",
        f"- Classifier mode: {classifier or 'unknown'}",
        f"- Matcher mode: {matcher or 'unknown'}",
        f"- Total messages: {total_messages}",
        f"- review_required_count: {review_required_count}",
        f"- no_review_count: {no_review_count}",
        f"- existing_content: {match_status_counts['existing_content']}",
        f"- possible_duplicate: {match_status_counts['possible_duplicate']}",
        f"- metadata_issue: {match_status_counts['metadata_issue']}",
        f"- track_only: {match_status_counts['track_only']}",
        f"- safety_review: {match_status_counts['safety_review']}",
        f"- needs_review: {match_status_counts['needs_review']}",
        "- content_ops_decisions.csv contains all processed decisions.",
        "- review_queue.csv contains only review-required items.",
        *_hybrid_overview_lines(classifier, classifier_source_counts),
        "",
        "## Count by Label",
        *_format_counter(label_counts),
        "",
        "## Count by Routing",
        *_format_counter(routing_counts),
        "",
        "## Count by Match Status",
        *_format_counter(match_status_counts),
        "",
        "## Count by Final Action",
        *_format_counter(final_action_counts),
        "",
        "## Count by Priority",
        *_format_counter(priority_counts),
        "",
        "## Count by Classifier Source",
        *_format_counter(classifier_source_counts),
        "",
        "## Content Requests",
        *_format_ids(content_requests),
        "",
        "## Safety Reviews",
        *_format_ids(safety_reviews),
        "",
        "## Needs Review",
        *_format_ids(needs_review),
        "",
        "## Example Message IDs",
        *_format_ids([item.message.message_id for item in processed_items[:5]]),
        "",
    ]

    Path(output_path).write_text("\n".join(lines), encoding="utf-8")


def _format_counter(counter: Counter[str]) -> list[str]:
    if not counter:
        return ["- None: 0"]
    return [f"- {key}: {counter[key]}" for key in sorted(counter)]


def _report_note(classifier: str | None) -> str:
    if classifier == "mock":
        return "This report uses deterministic counts from processed mock-classified data."
    if classifier == "llm":
        return "This report uses deterministic counts from processed LLM-classified data."
    return "This report uses deterministic counts from processed pipeline data."


def _hybrid_overview_lines(
    classifier: str | None, classifier_source_counts: Counter[str]
) -> list[str]:
    if classifier != "hybrid":
        return []
    return [
        f"- hybrid_llm_processed: {classifier_source_counts['hybrid_llm']}",
        f"- hybrid_ml_processed: {classifier_source_counts['hybrid_ml']}",
        f"- ml_abstained: {classifier_source_counts['ml_abstain']}",
        f"- hybrid_rule_safety: {classifier_source_counts['hybrid_rule_safety']}",
        f"- hybrid_rule_bug: {classifier_source_counts['hybrid_rule_bug']}",
        f"- hybrid_rule_praise: {classifier_source_counts['hybrid_rule_praise']}",
    ]


def _format_ids(message_ids: list[str]) -> list[str]:
    if not message_ids:
        return ["- None"]
    return [f"- {message_id}" for message_id in message_ids]
