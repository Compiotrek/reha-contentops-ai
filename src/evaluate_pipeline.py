import argparse
from pathlib import Path

import pandas as pd

from src.classify import ML_AUTO_ACCEPT_CONFIDENCE
from src.content_gap import detect_content_gap_alerts
from src.process import process_feedback
from src.schemas import ProcessedFeedback


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
OUTPUT_DIR = ROOT / "outputs"
DEFAULT_INPUT_CSV = DATA_DIR / "heldout_eval_messages.csv"
DEFAULT_EXPECTED_CSV = DATA_DIR / "heldout_expected_outcomes.csv"
DEFAULT_REPORT_PATH = OUTPUT_DIR / "pipeline_evaluation.md"
DEFAULT_DETAILS_PATH = OUTPUT_DIR / "pipeline_evaluation_details.csv"


def evaluate_pipeline(
    classifier: str = "mock",
    matcher: str = "placeholder",
    input_csv: str | Path = DEFAULT_INPUT_CSV,
    expected_csv: str | Path = DEFAULT_EXPECTED_CSV,
    metadata_extractor: str = "auto",
    hybrid_threshold: float = ML_AUTO_ACCEPT_CONFIDENCE,
    report_path: str | Path = DEFAULT_REPORT_PATH,
    details_path: str | Path = DEFAULT_DETAILS_PATH,
) -> str:
    processed_items = process_feedback(
        classifier=classifier,
        matcher=matcher,
        input_csv=input_csv,
        hybrid_threshold=hybrid_threshold,
        metadata_extractor=metadata_extractor,
    )
    expected = _load_expected(expected_csv)
    detail_rows = _evaluation_rows(processed_items, expected)
    gap_rows = _gap_evaluation_rows(processed_items, expected)
    report = _build_report(
        classifier=classifier,
        matcher=matcher,
        processed_items=processed_items,
        detail_rows=detail_rows,
        gap_rows=gap_rows,
    )

    report_path = Path(report_path)
    details_path = Path(details_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    details_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report, encoding="utf-8")
    pd.DataFrame(detail_rows).to_csv(details_path, index=False, encoding="utf-8")
    return report


def _load_expected(path: str | Path) -> dict[str, dict]:
    rows = pd.read_csv(path).fillna("").to_dict(orient="records")
    return {row["message_id"]: row for row in rows}


def _evaluation_rows(
    processed_items: list[ProcessedFeedback],
    expected: dict[str, dict],
) -> list[dict]:
    rows = []
    for item in processed_items:
        expected_row = expected.get(item.message.message_id, {})
        expected_labels = _parse_expected_list(expected_row.get("expected_labels", ""))
        actual_labels = set(item.classification.labels)
        label_hit = all(label in actual_labels for label in expected_labels)
        expected_review_required = _parse_bool(
            expected_row.get("expected_review_required", "")
        )
        rows.append(
            {
                "message_id": item.message.message_id,
                "user_message": item.message.user_message,
                "expected_labels": ";".join(expected_labels),
                "actual_labels": ";".join(item.classification.labels),
                "label_hit": label_hit,
                "expected_match_status": expected_row.get("expected_match_status", ""),
                "actual_match_status": item.decision.match_status,
                "match_status_hit": (
                    item.decision.match_status
                    == expected_row.get("expected_match_status", "")
                ),
                "expected_final_action": expected_row.get("expected_final_action", ""),
                "actual_final_action": item.decision.final_action,
                "final_action_hit": (
                    item.decision.final_action
                    == expected_row.get("expected_final_action", "")
                ),
                "expected_review_required": expected_review_required,
                "actual_review_required": item.decision.review_required,
                "review_required_hit": (
                    item.decision.review_required == expected_review_required
                ),
                "expected_gap_cluster": expected_row.get("expected_gap_cluster", ""),
                "body_region": item.classification.body_region or "",
                "therapy_goal": item.classification.therapy_goal or "",
                "difficulty_requested": item.classification.difficulty_requested or "",
                "equipment": item.classification.equipment or "",
                "position": item.classification.position or "",
                "request_theme": item.classification.request_theme or "",
                "metadata_source": item.classification.metadata_source or "",
                "classifier_source": item.classification.classifier_source or "",
                "top_match_title": item.top_matches[0].title if item.top_matches else "",
                "decision_reason": item.decision.reason,
            }
        )
    return rows


def _gap_evaluation_rows(
    processed_items: list[ProcessedFeedback],
    expected: dict[str, dict],
) -> list[dict]:
    alerts = detect_content_gap_alerts(processed_items)
    actual_alert_sets = [set(alert.message_ids) for alert in alerts]
    expected_clusters: dict[str, set[str]] = {}
    for message_id, row in expected.items():
        cluster = row.get("expected_gap_cluster", "")
        if cluster:
            expected_clusters.setdefault(cluster, set()).add(message_id)

    rows = []
    for cluster, expected_message_ids in sorted(expected_clusters.items()):
        best_overlap = 0
        best_actual_size = 0
        for actual_message_ids in actual_alert_sets:
            overlap = len(expected_message_ids.intersection(actual_message_ids))
            if overlap > best_overlap:
                best_overlap = overlap
                best_actual_size = len(actual_message_ids)
        expected_count = len(expected_message_ids)
        recall = best_overlap / expected_count if expected_count else 0.0
        precision = best_overlap / best_actual_size if best_actual_size else 0.0
        rows.append(
            {
                "expected_gap_cluster": cluster,
                "expected_count": expected_count,
                "best_actual_alert_size": best_actual_size,
                "matched_expected_messages": best_overlap,
                "gap_recall": recall,
                "gap_precision": precision,
                "gap_hit": recall == 1.0 and precision >= 0.8,
            }
        )
    return rows


def _build_report(
    classifier: str,
    matcher: str,
    processed_items: list[ProcessedFeedback],
    detail_rows: list[dict],
    gap_rows: list[dict],
) -> str:
    total = len(detail_rows)
    label_accuracy = _mean(row["label_hit"] for row in detail_rows)
    match_accuracy = _mean(row["match_status_hit"] for row in detail_rows)
    action_accuracy = _mean(row["final_action_hit"] for row in detail_rows)
    review_accuracy = _mean(row["review_required_hit"] for row in detail_rows)
    gap_hits = sum(1 for row in gap_rows if row["gap_hit"])
    gap_total = len(gap_rows)

    lines = [
        "# Pipeline Evaluation",
        "",
        "This is a held-out evaluation report. It should be used to inspect generalization, not to tune the same examples.",
        "",
        f"- Classifier mode: {classifier}",
        f"- Matcher mode: {matcher}",
        f"- Evaluation messages: {total}",
        f"- Label coverage accuracy: {label_accuracy:.3f}",
        f"- Match status accuracy: {match_accuracy:.3f}",
        f"- Final action accuracy: {action_accuracy:.3f}",
        f"- Review-required accuracy: {review_accuracy:.3f}",
        f"- Expected gap clusters hit: {gap_hits}/{gap_total}",
        "",
        "## Gap Cluster Evaluation",
    ]
    if gap_rows:
        for row in gap_rows:
            lines.append(
                "- "
                f"{row['expected_gap_cluster']}: "
                f"recall {row['gap_recall']:.3f}, "
                f"precision {row['gap_precision']:.3f}, "
                f"matched {row['matched_expected_messages']}/{row['expected_count']}"
            )
    else:
        lines.append("- None")

    misses = [
        row
        for row in detail_rows
        if not (
            row["label_hit"]
            and row["match_status_hit"]
            and row["final_action_hit"]
            and row["review_required_hit"]
        )
    ]
    lines.extend(["", "## Misses"])
    if misses:
        for row in misses[:20]:
            lines.append(
                "- "
                f"{row['message_id']}: expected "
                f"{row['expected_match_status']}/{row['expected_final_action']} "
                f"but got {row['actual_match_status']}/{row['actual_final_action']} "
                f"labels={row['actual_labels']}"
            )
    else:
        lines.append("- None")

    lines.extend(
        [
            "",
            "## Actual Alerts",
        ]
    )
    for alert in detect_content_gap_alerts(processed_items):
        lines.append(
            f"- {alert.alert_id}: {alert.theme} ({alert.request_count} requests) "
            f"ids={';'.join(alert.message_ids)}"
        )
    if lines[-1] == "## Actual Alerts":
        lines.append("- None")

    return "\n".join(lines) + "\n"


def _parse_expected_list(value: str) -> list[str]:
    return [part.strip() for part in str(value).split(";") if part.strip()]


def _parse_bool(value) -> bool:
    return str(value).strip().casefold() in {"true", "1", "yes", "y"}


def _mean(values) -> float:
    values = list(values)
    if not values:
        return 0.0
    return sum(bool(value) for value in values) / len(values)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate pipeline outcomes.")
    parser.add_argument("--classifier", choices=["mock", "llm", "ml", "hybrid"], default="mock")
    parser.add_argument("--matcher", choices=["placeholder", "embeddings"], default="placeholder")
    parser.add_argument("--metadata-extractor", choices=["auto", "none", "rules", "llm"], default="auto")
    parser.add_argument("--input-csv", default=str(DEFAULT_INPUT_CSV))
    parser.add_argument("--expected-csv", default=str(DEFAULT_EXPECTED_CSV))
    parser.add_argument("--hybrid-threshold", type=float, default=ML_AUTO_ACCEPT_CONFIDENCE)
    parser.add_argument("--report-path", default=str(DEFAULT_REPORT_PATH))
    parser.add_argument("--details-path", default=str(DEFAULT_DETAILS_PATH))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = evaluate_pipeline(
        classifier=args.classifier,
        matcher=args.matcher,
        input_csv=args.input_csv,
        expected_csv=args.expected_csv,
        metadata_extractor=args.metadata_extractor,
        hybrid_threshold=args.hybrid_threshold,
        report_path=args.report_path,
        details_path=args.details_path,
    )
    print(report)
    print(f"Wrote evaluation report to {args.report_path}")
    print(f"Wrote evaluation details to {args.details_path}")


if __name__ == "__main__":
    main()
