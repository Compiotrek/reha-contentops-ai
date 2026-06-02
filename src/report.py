from collections import Counter
from pathlib import Path

from src.schemas import ProcessedFeedback


def generate_daily_report(
    processed_items: list[ProcessedFeedback],
    output_path: str,
    classifier: str | None = None,
) -> None:
    """Generate a deterministic Markdown report from processed pipeline data."""
    total_messages = len(processed_items)
    label_counts: Counter[str] = Counter()
    routing_counts: Counter[str] = Counter()
    match_status_counts: Counter[str] = Counter()

    for item in processed_items:
        label_counts.update(item.classification.labels)
        routing_counts.update([item.decision.routing])
        match_status_counts.update([item.decision.match_status])

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
        if item.decision.routing == "needs_review"
    ]

    lines = [
        "# Daily Content Ops Report",
        "",
        _report_note(classifier),
        "",
        f"- Total messages: {total_messages}",
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


def _format_ids(message_ids: list[str]) -> list[str]:
    if not message_ids:
        return ["- None"]
    return [f"- {message_id}" for message_id in message_ids]
