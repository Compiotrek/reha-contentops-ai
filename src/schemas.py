from typing import Literal

from pydantic import BaseModel, Field


AllowedLabel = Literal[
    "content_request",
    "criticism",
    "praise",
    "safety_signal",
    "bug_or_access_problem",
    "metadata_issue",
    "unclear",
    "other",
]

RoutingValue = Literal["process", "needs_review", "safety_review", "ignore"]


class FeedbackMessage(BaseModel):
    message_id: str
    user_message: str


class Exercise(BaseModel):
    exercise_id: str
    title: str
    body_region: str
    indication: str = ""
    therapy_goal: str
    difficulty: str
    equipment: str
    position: str
    description: str
    tags: str
    contraindication_note: str = ""
    review_status: str


class ClassifiedFeedback(BaseModel):
    message_id: str
    user_message: str
    labels: list[AllowedLabel]
    body_region: str | None = None
    therapy_goal: str | None = None
    difficulty_requested: str | None = None
    equipment: str | None = None
    position: str | None = None
    sentiment: str | None = None
    safety_flag: bool = False
    evidence_quote: str
    summary: str
    confidence: float = Field(ge=0.0, le=1.0)
    routing: RoutingValue
    classifier_source: str | None = None
    abstained: bool | None = None
    abstain_reason: str | None = None
    safety_probability: float | None = None
    top2_margin: float | None = None


class MatchResult(BaseModel):
    exercise_id: str
    title: str
    score: float
    reasons: list[str] = Field(default_factory=list)
    vector_similarity: float | None = None
    metadata_fit_score: float | None = None
    final_score: float | None = None
    metadata_mismatches: list[str] = Field(default_factory=list)
    reason: str | None = None


class DecisionResult(BaseModel):
    routing: RoutingValue
    match_status: str
    reason: str
    final_action: str


class ProcessedFeedback(BaseModel):
    message: FeedbackMessage
    classification: ClassifiedFeedback
    matches: list[MatchResult]
    decision: DecisionResult
    classifier_mode: str = "mock"
    matcher_mode: str = "placeholder"
    routing: str
    top_matches: list[MatchResult] = Field(default_factory=list)
    match_status: str
    decision_reason: str
    final_action: str
