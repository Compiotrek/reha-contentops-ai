import json
from pathlib import Path

import pandas as pd

from src.content_gap import ContentGapAlert, detect_content_gap_alerts
from src.report import generate_daily_report
from src.schemas import ProcessedFeedback


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "outputs"


def write_outputs(
    processed_items: list[ProcessedFeedback],
    classifier: str | None = None,
    matcher: str | None = None,
) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    processed_path = OUTPUT_DIR / "processed_feedback.json"
    processed_payload = [_model_dump(item) for item in processed_items]
    processed_path.write_text(
        json.dumps(processed_payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    decision_columns = [
        "message_id",
        "user_message",
        "labels",
        "body_region",
        "therapy_goal",
        "difficulty_requested",
        "equipment",
        "request_theme",
        "metadata_source",
        "safety_flag",
        "classifier_mode",
        "classifier_source",
        "matcher_mode",
        "routing",
        "match_status",
        "final_action",
        "review_required",
        "priority",
        "decision_reason",
        "top_match_exercise_id",
        "top_match_title",
        "top_match_score",
        "top_match_vector_similarity",
        "top_match_metadata_fit_score",
        "top_match_final_score",
        "top_match_metadata_mismatches",
        "confidence",
        "abstain_reason",
        "evidence_quote",
    ]
    decision_rows = [_content_ops_decision_row(item) for item in processed_items]
    pd.DataFrame(decision_rows, columns=decision_columns).to_csv(
        OUTPUT_DIR / "content_ops_decisions.csv",
        index=False,
        encoding="utf-8",
    )

    review_columns = [
        "message_id",
        "user_message",
        "labels",
        "body_region",
        "therapy_goal",
        "difficulty_requested",
        "equipment",
        "position",
        "request_theme",
        "metadata_source",
        "routing",
        "match_status",
        "final_action",
        "review_required",
        "priority",
        "top_match_title",
        "top_match_score",
        "metadata_mismatches",
        "decision_reason",
    ]
    review_rows = []
    for item in processed_items:
        if _include_in_review_queue(item):
            top_match = item.top_matches[0] if item.top_matches else None
            review_rows.append(
                {
                    "message_id": item.message.message_id,
                    "user_message": item.message.user_message,
                    "labels": ";".join(item.classification.labels),
                    "body_region": item.classification.body_region or "",
                    "therapy_goal": item.classification.therapy_goal or "",
                    "difficulty_requested": (
                        item.classification.difficulty_requested or ""
                    ),
                    "equipment": item.classification.equipment or "",
                    "request_theme": item.classification.request_theme or "",
                    "metadata_source": item.classification.metadata_source or "",
                    "position": item.classification.position or "",
                    "routing": item.decision.routing,
                    "match_status": item.decision.match_status,
                    "final_action": item.decision.final_action,
                    "review_required": _csv_bool(item.decision.review_required),
                    "priority": item.decision.priority,
                    "top_match_title": top_match.title if top_match else "",
                    "top_match_score": _match_score(top_match) if top_match else "",
                    "metadata_mismatches": (
                        "; ".join(top_match.metadata_mismatches) if top_match else ""
                    ),
                    "decision_reason": item.decision.reason,
                }
            )

    pd.DataFrame(review_rows, columns=review_columns).to_csv(
        OUTPUT_DIR / "review_queue.csv", index=False
    )

    content_gap_columns = [
        "alert_id",
        "request_count",
        "theme",
        "message_ids",
        "example_messages",
        "body_regions",
        "therapy_goals",
        "difficulty_requested",
        "equipment",
        "recommended_action",
    ]
    content_gap_rows = [
        _content_gap_alert_row(alert)
        for alert in detect_content_gap_alerts(processed_items)
    ]
    pd.DataFrame(content_gap_rows, columns=content_gap_columns).to_csv(
        OUTPUT_DIR / "content_gap_alerts.csv",
        index=False,
        encoding="utf-8",
    )
    generate_daily_report(
        processed_items,
        str(OUTPUT_DIR / "daily_report.md"),
        classifier=classifier,
        matcher=matcher,
    )


def format_single_message_summary(processed_item: ProcessedFeedback) -> str:
    classification = processed_item.classification
    decision = processed_item.decision
    top_match = processed_item.top_matches[0] if processed_item.top_matches else None
    lines = [
        f"Message: {processed_item.message.message_id}",
        f"Labels: {_format_list(classification.labels)}",
        f"Routing: {decision.routing}",
        f"Match status: {decision.match_status}",
        f"Action: {decision.final_action}",
        f"Review required: {_yes_no(decision.review_required)}",
    ]
    metadata = _format_metadata(classification)
    if metadata:
        lines.append(f"Metadata: {metadata}")
    if top_match and _show_top_match(decision.match_status):
        lines.append(f"Top match: {top_match.title} ({top_match.exercise_id})")
    if (
        top_match
        and _show_top_match(decision.match_status)
        and top_match.metadata_mismatches
    ):
        lines.append(f"Mismatches: {_format_list(top_match.metadata_mismatches)}")
    lines.append(f"Reason: {decision.reason}")
    lines.append(f"Outputs: {OUTPUT_DIR}")
    return "\n".join(lines)


def _model_dump(item: ProcessedFeedback) -> dict:
    if hasattr(item, "model_dump"):
        return item.model_dump()
    return item.dict()


def _include_in_review_queue(item: ProcessedFeedback) -> bool:
    return item.decision.review_required


def _content_ops_decision_row(item: ProcessedFeedback) -> dict:
    classification = item.classification
    decision = item.decision
    top_match = item.top_matches[0] if item.top_matches else None

    return {
        "message_id": item.message.message_id,
        "user_message": item.message.user_message,
        "labels": ";".join(classification.labels),
        "body_region": classification.body_region or "",
        "therapy_goal": classification.therapy_goal or "",
        "difficulty_requested": classification.difficulty_requested or "",
        "equipment": classification.equipment or "",
        "request_theme": classification.request_theme or "",
        "metadata_source": classification.metadata_source or "",
        "safety_flag": _csv_bool(classification.safety_flag),
        "classifier_mode": item.classifier_mode,
        "classifier_source": classification.classifier_source or "",
        "matcher_mode": item.matcher_mode,
        "routing": decision.routing,
        "match_status": decision.match_status,
        "final_action": decision.final_action,
        "review_required": _csv_bool(decision.review_required),
        "priority": decision.priority,
        "decision_reason": decision.reason,
        "top_match_exercise_id": top_match.exercise_id if top_match else "",
        "top_match_title": top_match.title if top_match else "",
        "top_match_score": _match_score(top_match) if top_match else "",
        "top_match_vector_similarity": (
            top_match.vector_similarity if top_match else ""
        ),
        "top_match_metadata_fit_score": (
            top_match.metadata_fit_score if top_match else ""
        ),
        "top_match_final_score": top_match.final_score if top_match else "",
        "top_match_metadata_mismatches": (
            "; ".join(top_match.metadata_mismatches) if top_match else ""
        ),
        "confidence": classification.confidence,
        "abstain_reason": classification.abstain_reason or "",
        "evidence_quote": classification.evidence_quote,
    }


def _content_gap_alert_row(alert: ContentGapAlert) -> dict:
    return {
        "alert_id": alert.alert_id,
        "request_count": alert.request_count,
        "theme": alert.theme,
        "message_ids": ";".join(alert.message_ids),
        "example_messages": " | ".join(alert.example_messages),
        "body_regions": ";".join(alert.body_regions),
        "therapy_goals": ";".join(alert.therapy_goals),
        "difficulty_requested": ";".join(alert.difficulty_requested),
        "equipment": ";".join(alert.equipment),
        "recommended_action": alert.recommended_action,
    }


def _csv_bool(value: bool) -> str:
    return "true" if value else "false"


def _match_score(match_result) -> float:
    if match_result.final_score is not None:
        return match_result.final_score
    return match_result.score


def _format_metadata(classification) -> str:
    fields = [
        ("body_region", classification.body_region),
        ("therapy_goal", classification.therapy_goal),
        ("difficulty", classification.difficulty_requested),
        ("equipment", classification.equipment),
        ("position", classification.position),
        ("theme", classification.request_theme),
    ]
    return ", ".join(f"{name}={value}" for name, value in fields if value)


def _show_top_match(match_status: str) -> bool:
    return match_status in {"existing_content", "possible_duplicate", "needs_review"}


def _format_list(values: list[str]) -> str:
    return ", ".join(values) if values else "none"


def _yes_no(value: bool) -> str:
    return "yes" if value else "no"
