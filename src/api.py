import os
from typing import Any

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from src.load_data import load_exercises
from src.process import DATA_DIR, process_single_feedback
from src.schemas import Exercise, FeedbackMessage, ProcessedFeedback


ALLOWED_CLASSIFIERS = {"mock", "llm"}
ALLOWED_MATCHERS = {"placeholder", "embeddings"}

app = FastAPI(title="Reha ContentOps AI")
EXERCISES: list[Exercise] = load_exercises(str(DATA_DIR / "exercises.csv"))


class ProcessFeedbackRequest(BaseModel):
    message_id: str
    user_message: str
    classifier: str = "mock"
    matcher: str = "placeholder"


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "service": "reha-contentops-ai"}


@app.post("/process-feedback")
def process_feedback_endpoint(request: ProcessFeedbackRequest) -> dict[str, Any]:
    _validate_modes(request.classifier, request.matcher)
    _validate_openai_requirements(request.classifier, request.matcher)

    feedback_message = FeedbackMessage(
        message_id=request.message_id,
        user_message=request.user_message,
    )
    processed = process_single_feedback(
        feedback_message=feedback_message,
        exercises=EXERCISES,
        classifier_mode=request.classifier,
        matcher_mode=request.matcher,
    )
    return _model_dump(processed)


def _validate_modes(classifier: str, matcher: str) -> None:
    if classifier not in ALLOWED_CLASSIFIERS:
        raise HTTPException(
            status_code=400,
            detail="Invalid classifier. Allowed values: mock, llm.",
        )
    if matcher not in ALLOWED_MATCHERS:
        raise HTTPException(
            status_code=400,
            detail="Invalid matcher. Allowed values: placeholder, embeddings.",
        )


def _validate_openai_requirements(classifier: str, matcher: str) -> None:
    if classifier != "llm" and matcher != "embeddings":
        return
    load_dotenv()
    if os.getenv("OPENAI_API_KEY"):
        return
    if classifier == "llm":
        raise HTTPException(
            status_code=400,
            detail="OPENAI_API_KEY is required for classifier='llm'. Use classifier='mock' for local runs without OpenAI credentials.",
        )
    raise HTTPException(
        status_code=400,
        detail="OPENAI_API_KEY is required for matcher='embeddings'. Use matcher='placeholder' for local runs without OpenAI credentials.",
    )


def _model_dump(processed: ProcessedFeedback) -> dict[str, Any]:
    if hasattr(processed, "model_dump"):
        return processed.model_dump()
    return processed.dict()
