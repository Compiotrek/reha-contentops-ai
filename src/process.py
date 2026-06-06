import argparse
from pathlib import Path

from src.classify import (
    ML_AUTO_ACCEPT_CONFIDENCE,
    classify_feedback_hybrid,
    classify_feedback_llm,
    classify_feedback_ml,
    classify_feedback_mock,
)
from src.decide import apply_routing, decide_match_status
from src.load_data import load_exercises, load_feedback_messages
from src.match import match_exercises_embeddings, match_exercises_placeholder
from src.metadata import enrich_request_metadata
from src.output import OUTPUT_DIR, format_single_message_summary, write_outputs
from src.schemas import Exercise, FeedbackMessage, ProcessedFeedback


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"


def process_feedback(
    classifier: str = "mock",
    matcher: str = "placeholder",
    limit: int | None = None,
    hybrid_threshold: float = ML_AUTO_ACCEPT_CONFIDENCE,
    input_csv: str | Path | None = None,
    metadata_extractor: str = "auto",
) -> list[ProcessedFeedback]:
    feedback_path = Path(input_csv) if input_csv else DATA_DIR / "feedback_messages.csv"
    feedback_messages = load_feedback_messages(str(feedback_path))
    if limit is not None:
        feedback_messages = feedback_messages[:limit]
    exercises = load_exercises(str(DATA_DIR / "exercises.csv"))
    classify = _get_classifier(classifier, hybrid_threshold=hybrid_threshold)
    match = _get_matcher(matcher)

    return [
        process_single_feedback(
            feedback_message=message,
            exercises=exercises,
            classifier_mode=classifier,
            matcher_mode=matcher,
            classifier=classify,
            matcher_fn=match,
            metadata_extractor=metadata_extractor,
        )
        for message in feedback_messages
    ]


def process_feedback_messages(
    feedback_messages: list[FeedbackMessage],
    classifier: str = "mock",
    matcher: str = "placeholder",
    hybrid_threshold: float | None = ML_AUTO_ACCEPT_CONFIDENCE,
    metadata_extractor: str = "auto",
    exercises: list[Exercise] | None = None,
) -> list[ProcessedFeedback]:
    exercise_records = exercises or load_exercises(str(DATA_DIR / "exercises.csv"))
    classify = _get_classifier(
        classifier,
        hybrid_threshold=(
            hybrid_threshold
            if hybrid_threshold is not None
            else ML_AUTO_ACCEPT_CONFIDENCE
        ),
    )
    match = _get_matcher(matcher)

    return [
        process_single_feedback(
            feedback_message=message,
            exercises=exercise_records,
            classifier_mode=classifier,
            matcher_mode=matcher,
            classifier=classify,
            matcher_fn=match,
            metadata_extractor=metadata_extractor,
        )
        for message in feedback_messages
    ]


def process_feedback_message(
    user_message: str,
    message_id: str = "cli_message",
    classifier: str = "mock",
    matcher: str = "placeholder",
    hybrid_threshold: float = ML_AUTO_ACCEPT_CONFIDENCE,
    metadata_extractor: str = "auto",
) -> ProcessedFeedback:
    exercises = load_exercises(str(DATA_DIR / "exercises.csv"))
    return process_single_feedback(
        feedback_message=FeedbackMessage(
            message_id=message_id,
            user_message=user_message,
        ),
        exercises=exercises,
        classifier_mode=classifier,
        matcher_mode=matcher,
        classifier=_get_classifier(classifier, hybrid_threshold=hybrid_threshold),
        matcher_fn=_get_matcher(matcher),
        metadata_extractor=metadata_extractor,
    )


def process_single_feedback(
    feedback_message: FeedbackMessage,
    exercises: list[Exercise],
    classifier_mode: str,
    matcher_mode: str,
    classifier=None,
    matcher_fn=None,
    metadata_extractor: str = "auto",
) -> ProcessedFeedback:
    classify = classifier or _get_classifier(classifier_mode)
    match = matcher_fn or _get_matcher(matcher_mode)

    classified = classify(feedback_message)
    classified = enrich_request_metadata(
        classified,
        mode=metadata_extractor,
        classifier_mode=classifier_mode,
        matcher_mode=matcher_mode,
    )
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
        review_required=decision.review_required,
        priority=decision.priority,
    )


def _get_classifier(
    classifier: str, hybrid_threshold: float = ML_AUTO_ACCEPT_CONFIDENCE
):
    if classifier == "mock":
        return classify_feedback_mock
    if classifier == "llm":
        return classify_feedback_llm
    if classifier == "ml":
        return classify_feedback_ml
    if classifier == "hybrid":
        return lambda message: classify_feedback_hybrid(
            message, threshold=hybrid_threshold
        )
    raise ValueError(f"Unsupported classifier: {classifier}")


def _get_matcher(matcher: str):
    if matcher == "placeholder":
        return match_exercises_placeholder
    if matcher == "embeddings":
        return match_exercises_embeddings
    raise ValueError(f"Unsupported matcher: {matcher}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the Reha ContentOps AI pipeline.")
    parser.add_argument(
        "--classifier",
        choices=["mock", "llm", "ml", "hybrid"],
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
        "--input-csv",
        default=None,
        help="Optional feedback CSV path. Defaults to data/feedback_messages.csv.",
    )
    parser.add_argument(
        "--message",
        default=None,
        help="Optional single feedback message to process instead of the CSV dataset.",
    )
    parser.add_argument(
        "--message-id",
        default="cli_message",
        help="Message ID to use with --message. Defaults to cli_message.",
    )
    parser.add_argument(
        "--matcher",
        choices=["placeholder", "embeddings"],
        default="placeholder",
        help="Matcher backend to use. Defaults to placeholder.",
    )
    parser.add_argument(
        "--hybrid-threshold",
        type=float,
        default=ML_AUTO_ACCEPT_CONFIDENCE,
        help="Auto-accept confidence threshold for hybrid ML predictions.",
    )
    parser.add_argument(
        "--metadata-extractor",
        choices=["auto", "none", "rules", "llm"],
        default="auto",
        help=(
            "Metadata enrichment after classification. Auto uses LLM when an "
            "API key is available, otherwise rules for mock/placeholder demos."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.message:
        processed_item = process_feedback_message(
            user_message=args.message,
            message_id=args.message_id,
            classifier=args.classifier,
            matcher=args.matcher,
            hybrid_threshold=args.hybrid_threshold,
            metadata_extractor=args.metadata_extractor,
        )
        write_outputs([processed_item], classifier=args.classifier, matcher=args.matcher)
        print(format_single_message_summary(processed_item))
        return

    processed_items = process_feedback(
        classifier=args.classifier,
        matcher=args.matcher,
        limit=args.limit,
        hybrid_threshold=args.hybrid_threshold,
        input_csv=args.input_csv,
        metadata_extractor=args.metadata_extractor,
    )
    write_outputs(processed_items, classifier=args.classifier, matcher=args.matcher)
    print(
        f"Processed {len(processed_items)} feedback messages with "
        f"{args.classifier} classifier and {args.matcher} matcher."
    )
    print(f"Wrote outputs to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
