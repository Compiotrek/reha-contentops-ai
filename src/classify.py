import joblib

from src.llm_classifier import (
    _llm_fallback_classification as _llm_fallback_classification,
    _normalize_llm_payload as _normalize_llm_payload,
    classify_feedback_llm,
)
from src.ml_classifier import (
    CLASSIFIER_MODEL_PATH,
    MEANINGFUL_LABELS as MEANINGFUL_LABELS,
    ML_AUTO_ACCEPT_CONFIDENCE,
    ML_AUTO_ACCEPT_MARGIN as ML_AUTO_ACCEPT_MARGIN,
    ML_SAFETY_THRESHOLD as ML_SAFETY_THRESHOLD,
    MLClassifierModelNotFoundError,
    _classify_feedback_ml_with_artifact,
    _predict_label_probabilities as _predict_label_probabilities,
    _predict_safety_probability as _predict_safety_probability,
    _top2_margin as _top2_margin,
    ml_auto_accepted,
)
from src.rules import (
    classify_feedback_mock as classify_feedback_mock,
    classify_with_rule_gates,
    enrich_classification_metadata as enrich_classification_metadata,
)
from src.schemas import ClassifiedFeedback, FeedbackMessage


def classify_feedback_ml(message: FeedbackMessage) -> ClassifiedFeedback:
    if not CLASSIFIER_MODEL_PATH.exists():
        raise MLClassifierModelNotFoundError(
            "ML classifier model not found. Run python -m src.train_ml_classifier first."
        )
    return _classify_feedback_ml_with_artifact(message, joblib.load(CLASSIFIER_MODEL_PATH))


def classify_feedback_hybrid(
    message: FeedbackMessage, threshold: float = ML_AUTO_ACCEPT_CONFIDENCE
) -> ClassifiedFeedback:
    gated_result = classify_with_rule_gates(message)
    if gated_result:
        return gated_result

    ml_result = classify_feedback_ml(message)
    if ml_result.routing == "safety_review":
        ml_result.classifier_source = "hybrid_ml"
        return ml_result
    if ml_auto_accepted(ml_result, threshold):
        ml_result.classifier_source = "hybrid_ml"
        return ml_result

    llm_result = classify_feedback_llm(message)
    if llm_result.abstained is False:
        llm_result.classifier_source = "hybrid_llm"
        return llm_result

    ml_result.routing = "needs_review"
    ml_result.classifier_source = "ml_abstain"
    ml_result.abstained = True
    ml_result.abstain_reason = (
        "ML classifier abstained; LLM fallback failed: "
        f"{llm_result.abstain_reason or 'unknown error'}"
    )
    return ml_result
