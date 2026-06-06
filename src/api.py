import os
from csv import DictReader
from io import StringIO
from typing import Any

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.classify import CLASSIFIER_MODEL_PATH
from src.content_gap import detect_content_gap_alerts
from src.load_data import load_exercises
from src.output import OUTPUT_DIR, write_outputs
from src.process import DATA_DIR, process_feedback_messages, process_single_feedback
from src.schemas import Exercise, FeedbackMessage, ProcessedFeedback


ALLOWED_CLASSIFIERS = {"mock", "llm", "ml", "hybrid"}
ALLOWED_MATCHERS = {"placeholder", "embeddings"}
ALLOWED_METADATA_EXTRACTORS = {"auto", "none", "rules", "llm"}

app = FastAPI(title="Reha ContentOps AI")
EXERCISES: list[Exercise] = load_exercises(str(DATA_DIR / "exercises.csv"))


class ProcessFeedbackRequest(BaseModel):
    message_id: str
    user_message: str
    classifier: str = "mock"
    matcher: str = "placeholder"
    metadata_extractor: str = "auto"


class BatchFeedbackItem(BaseModel):
    message_id: str
    user_message: str


class ProcessFeedbackBatchRequest(BaseModel):
    messages: list[BatchFeedbackItem] | None = None
    csv_text: str | None = None
    classifier: str = "mock"
    matcher: str = "placeholder"
    metadata_extractor: str = "auto"
    hybrid_threshold: float | None = None
    write_output_files: bool = True
    limit: int | None = Field(default=None, ge=1)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "service": "reha-contentops-ai"}


@app.post("/process-feedback")
def process_feedback_endpoint(request: ProcessFeedbackRequest) -> dict[str, Any]:
    _validate_modes(request.classifier, request.matcher, request.metadata_extractor)
    _validate_runtime_requirements(
        request.classifier, request.matcher, request.metadata_extractor
    )

    feedback_message = FeedbackMessage(
        message_id=request.message_id,
        user_message=request.user_message,
    )
    processed = process_single_feedback(
        feedback_message=feedback_message,
        exercises=EXERCISES,
        classifier_mode=request.classifier,
        matcher_mode=request.matcher,
        metadata_extractor=request.metadata_extractor,
    )
    return _model_dump(processed)


@app.post("/process-feedback-batch")
def process_feedback_batch_endpoint(
    request: ProcessFeedbackBatchRequest,
) -> dict[str, Any]:
    _validate_modes(request.classifier, request.matcher, request.metadata_extractor)
    _validate_runtime_requirements(
        request.classifier, request.matcher, request.metadata_extractor
    )
    feedback_messages = _batch_messages_from_request(request)
    if request.limit is not None:
        feedback_messages = feedback_messages[: request.limit]

    processed_items = process_feedback_messages(
        feedback_messages=feedback_messages,
        classifier=request.classifier,
        matcher=request.matcher,
        hybrid_threshold=request.hybrid_threshold
        if request.hybrid_threshold is not None
        else None,
        metadata_extractor=request.metadata_extractor,
        exercises=EXERCISES,
    )
    if request.write_output_files:
        write_outputs(
            processed_items,
            classifier=request.classifier,
            matcher=request.matcher,
        )
    content_gap_alerts = detect_content_gap_alerts(processed_items)

    return {
        "processed_count": len(processed_items),
        "classifier": request.classifier,
        "matcher": request.matcher,
        "metadata_extractor": request.metadata_extractor,
        "exercise_records_loaded": len(EXERCISES),
        "outputs_written": request.write_output_files,
        "output_dir": str(OUTPUT_DIR) if request.write_output_files else None,
        "content_gap_alerts": [
            _content_gap_alert_dump(alert) for alert in content_gap_alerts
        ],
        "items": [_model_dump(item) for item in processed_items],
    }


def _validate_modes(
    classifier: str, matcher: str, metadata_extractor: str = "auto"
) -> None:
    if classifier not in ALLOWED_CLASSIFIERS:
        raise HTTPException(
            status_code=400,
            detail="Invalid classifier. Allowed values: mock, llm, ml, hybrid.",
        )
    if matcher not in ALLOWED_MATCHERS:
        raise HTTPException(
            status_code=400,
            detail="Invalid matcher. Allowed values: placeholder, embeddings.",
        )
    if metadata_extractor not in ALLOWED_METADATA_EXTRACTORS:
        raise HTTPException(
            status_code=400,
            detail="Invalid metadata_extractor. Allowed values: auto, none, rules, llm.",
        )


def _validate_runtime_requirements(
    classifier: str, matcher: str, metadata_extractor: str = "auto"
) -> None:
    if classifier in {"ml", "hybrid"} and not CLASSIFIER_MODEL_PATH.exists():
        raise HTTPException(
            status_code=400,
            detail="ML classifier model not found. Run python -m src.train_ml_classifier first.",
        )
    if (
        classifier != "llm"
        and matcher != "embeddings"
        and metadata_extractor != "llm"
    ):
        return
    load_dotenv()
    if os.getenv("OPENAI_API_KEY"):
        return
    if classifier == "llm":
        raise HTTPException(
            status_code=400,
            detail="OPENAI_API_KEY is required for classifier='llm'. Use classifier='mock' for local runs without OpenAI credentials.",
        )
    if metadata_extractor == "llm":
        raise HTTPException(
            status_code=400,
            detail="OPENAI_API_KEY is required for metadata_extractor='llm'. Use metadata_extractor='rules' for local runs without OpenAI credentials.",
        )
    raise HTTPException(
        status_code=400,
        detail="OPENAI_API_KEY is required for matcher='embeddings'. Use matcher='placeholder' for local runs without OpenAI credentials.",
    )


def _batch_messages_from_request(
    request: ProcessFeedbackBatchRequest,
) -> list[FeedbackMessage]:
    if request.messages and request.csv_text:
        raise HTTPException(
            status_code=400,
            detail="Send either messages or csv_text, not both.",
        )
    if request.messages:
        return [
            FeedbackMessage(
                message_id=item.message_id,
                user_message=item.user_message,
            )
            for item in request.messages
        ]
    if request.csv_text:
        return _parse_feedback_csv_text(request.csv_text)
    raise HTTPException(
        status_code=400,
        detail="Batch request must include messages or csv_text.",
    )


def _parse_feedback_csv_text(csv_text: str) -> list[FeedbackMessage]:
    rows = list(DictReader(StringIO(csv_text)))
    if not rows:
        raise HTTPException(status_code=400, detail="csv_text did not contain rows.")
    required_columns = {"message_id", "user_message"}
    missing_columns = required_columns - set(rows[0])
    if missing_columns:
        raise HTTPException(
            status_code=400,
            detail=(
                "csv_text must contain columns: message_id, user_message. "
                f"Missing: {', '.join(sorted(missing_columns))}."
            ),
        )
    return [
        FeedbackMessage(
            message_id=str(row["message_id"]),
            user_message=str(row["user_message"]),
        )
        for row in rows
    ]


def _model_dump(processed: ProcessedFeedback) -> dict[str, Any]:
    if hasattr(processed, "model_dump"):
        return processed.model_dump()
    return processed.dict()


def _content_gap_alert_dump(alert) -> dict[str, Any]:
    return {
        "alert_id": alert.alert_id,
        "request_count": alert.request_count,
        "theme": alert.theme,
        "message_ids": alert.message_ids,
        "example_messages": alert.example_messages,
        "body_regions": alert.body_regions,
        "therapy_goals": alert.therapy_goals,
        "difficulty_requested": alert.difficulty_requested,
        "equipment": alert.equipment,
        "recommended_action": alert.recommended_action,
    }
