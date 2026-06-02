import argparse
import json
from pathlib import Path

import pandas as pd

from src.classify import classify_feedback_llm, classify_feedback_mock
from src.decide import apply_routing, decide_match_status
from src.load_data import load_exercises, load_feedback_messages
from src.match import match_exercises_embeddings, match_exercises_placeholder
from src.report import generate_daily_report
from src.schemas import Exercise, FeedbackMessage, ProcessedFeedback


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
OUTPUT_DIR = ROOT / "outputs"


def process_feedback(
    classifier: str = "mock",
    matcher: str = "placeholder",
    limit: int | None = None,
) -> list[ProcessedFeedback]:
    feedback_messages = load_feedback_messages(str(DATA_DIR / "feedback_messages.csv"))
    if limit is not None:
        feedback_messages = feedback_messages[:limit]
    exercises = load_exercises(str(DATA_DIR / "exercises.csv"))
    classify = _get_classifier(classifier)
    match = _get_matcher(matcher)

    return [
        process_single_feedback(
            feedback_message=message,
            exercises=exercises,
            classifier_mode=classifier,
            matcher_mode=matcher,
            classifier=classify,
            matcher_fn=match,
        )
        for message in feedback_messages
    ]


def process_single_feedback(
    feedback_message: FeedbackMessage,
    exercises: list[Exercise],
    classifier_mode: str,
    matcher_mode: str,
    classifier=None,
    matcher_fn=None,
) -> ProcessedFeedback:
    classify = classifier or _get_classifier(classifier_mode)
    match = matcher_fn or _get_matcher(matcher_mode)

    classified = classify(feedback_message)
    classified.routing = apply_routing(classified)
    matches = match(classified, exercises)
    decision = decide_match_status(classified, matches)

    return ProcessedFeedback(
        message=feedback_message,
        classification=classified,
        matches=matches,
        decision=decision,
        classifier_mode=classifier_mode,
        matcher_mode=matcher_mode,
        routing=decision.routing,
        top_matches=matches,
        match_status=decision.match_status,
        decision_reason=decision.reason,
        final_action=decision.final_action,
    )


def _get_classifier(classifier: str):
    if classifier == "mock":
        return classify_feedback_mock
    if classifier == "llm":
        return classify_feedback_llm
    raise ValueError(f"Unsupported classifier: {classifier}")


def _get_matcher(matcher: str):
    if matcher == "placeholder":
        return match_exercises_placeholder
    if matcher == "embeddings":
        return match_exercises_embeddings
    raise ValueError(f"Unsupported matcher: {matcher}")


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

    review_rows = []
    for item in processed_items:
        if _include_in_review_queue(item):
            top_match = item.top_matches[0] if item.top_matches else None
            review_rows.append(
                {
                    "message_id": item.message.message_id,
                    "user_message": item.message.user_message,
                    "labels": ";".join(item.classification.labels),
                    "routing": item.decision.routing,
                    "match_status": item.decision.match_status,
                    "final_action": item.decision.final_action,
                    "top_match_title": top_match.title if top_match else "",
                    "top_match_score": _match_score(top_match) if top_match else "",
                    "metadata_mismatches": (
                        "; ".join(top_match.metadata_mismatches) if top_match else ""
                    ),
                    "decision_reason": item.decision.reason,
                }
            )

    pd.DataFrame(review_rows).to_csv(OUTPUT_DIR / "review_queue.csv", index=False)
    generate_daily_report(
        processed_items,
        str(OUTPUT_DIR / "daily_report.md"),
        classifier=classifier,
        matcher=matcher,
    )


def _model_dump(item: ProcessedFeedback) -> dict:
    if hasattr(item, "model_dump"):
        return item.model_dump()
    return item.dict()


def _include_in_review_queue(item: ProcessedFeedback) -> bool:
    review_actions = {
        "route_to_safety_review",
        "route_to_human_review",
        "link_existing_content_for_review",
        "review_possible_duplicate",
        "review_metadata",
    }
    return (
        item.decision.routing in {"needs_review", "safety_review"}
        or item.decision.final_action in review_actions
    )


def _match_score(match_result) -> float:
    if match_result.final_score is not None:
        return match_result.final_score
    return match_result.score


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the Reha ContentOps AI pipeline.")
    parser.add_argument(
        "--classifier",
        choices=["mock", "llm"],
        default="mock",
        help="Classifier backend to use. Defaults to mock.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional number of feedback messages to process.",
    )
    parser.add_argument(
        "--matcher",
        choices=["placeholder", "embeddings"],
        default="placeholder",
        help="Matcher backend to use. Defaults to placeholder.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    processed_items = process_feedback(
        classifier=args.classifier,
        matcher=args.matcher,
        limit=args.limit,
    )
    write_outputs(processed_items, classifier=args.classifier, matcher=args.matcher)
    print(
        f"Processed {len(processed_items)} feedback messages with "
        f"{args.classifier} classifier and {args.matcher} matcher."
    )
    print(f"Wrote outputs to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
