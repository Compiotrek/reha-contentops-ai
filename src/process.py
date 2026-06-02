import argparse
import json
from pathlib import Path

import pandas as pd

from src.classify import classify_feedback_llm, classify_feedback_mock
from src.decide import apply_routing, decide_match_status
from src.load_data import load_exercises, load_feedback_messages
from src.match import match_exercises_placeholder
from src.report import generate_daily_report
from src.schemas import ProcessedFeedback


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
OUTPUT_DIR = ROOT / "outputs"


def process_feedback(
    classifier: str = "mock", limit: int | None = None
) -> list[ProcessedFeedback]:
    feedback_messages = load_feedback_messages(str(DATA_DIR / "feedback_messages.csv"))
    if limit is not None:
        feedback_messages = feedback_messages[:limit]
    exercises = load_exercises(str(DATA_DIR / "exercises.csv"))
    classify = _get_classifier(classifier)

    processed_items: list[ProcessedFeedback] = []
    for message in feedback_messages:
        classified = classify(message)
        classified.routing = apply_routing(classified)
        matches = match_exercises_placeholder(classified, exercises)
        decision = decide_match_status(classified, matches)

        processed_items.append(
            ProcessedFeedback(
                message=message,
                classification=classified,
                matches=matches,
                decision=decision,
            )
        )

    return processed_items


def _get_classifier(classifier: str):
    if classifier == "mock":
        return classify_feedback_mock
    if classifier == "llm":
        return classify_feedback_llm
    raise ValueError(f"Unsupported classifier: {classifier}")


def write_outputs(
    processed_items: list[ProcessedFeedback], classifier: str | None = None
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
        if item.decision.routing in {"needs_review", "safety_review"}:
            review_rows.append(
                {
                    "message_id": item.message.message_id,
                    "routing": item.decision.routing,
                    "match_status": item.decision.match_status,
                    "labels": ";".join(item.classification.labels),
                    "safety_flag": item.classification.safety_flag,
                    "confidence": item.classification.confidence,
                    "summary": item.classification.summary,
                }
            )

    pd.DataFrame(review_rows).to_csv(OUTPUT_DIR / "review_queue.csv", index=False)
    generate_daily_report(
        processed_items,
        str(OUTPUT_DIR / "daily_report.md"),
        classifier=classifier,
    )


def _model_dump(item: ProcessedFeedback) -> dict:
    if hasattr(item, "model_dump"):
        return item.model_dump()
    return item.dict()


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
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    processed_items = process_feedback(classifier=args.classifier, limit=args.limit)
    write_outputs(processed_items, classifier=args.classifier)
    print(f"Processed {len(processed_items)} feedback messages with {args.classifier} classifier.")
    print(f"Wrote outputs to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
