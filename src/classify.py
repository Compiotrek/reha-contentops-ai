from src.schemas import ClassifiedFeedback, FeedbackMessage


SAFETY_KEYWORDS = ["pain", "schmerzen", "schwindelig", "unsicher", "op darf"]
REQUEST_KEYWORDS = ["mehr", "hätte gern", "bitte", "ich brauche"]
CRITICISM_KEYWORDS = ["schwer", "zu lang", "nervt", "problem"]
PRAISE_KEYWORDS = ["super", "gut", "verständlich"]
BUG_KEYWORDS = ["startbutton", "reagiert nicht", "app"]
METADATA_KEYWORDS = ["tag", "steht", "passt nicht", "anfänger"]


def classify_feedback_mock(message: FeedbackMessage) -> ClassifiedFeedback:
    """Classify feedback with deterministic keyword rules.

    This is a mock implementation intended to be replaced by an LLM-backed
    classifier later. It does not provide medical advice.
    """
    text = message.user_message
    normalized = text.casefold()

    labels: list[str] = []
    safety_flag = _contains_any(normalized, SAFETY_KEYWORDS)
    if safety_flag:
        labels.append("safety_signal")
    if _contains_any(normalized, REQUEST_KEYWORDS):
        labels.append("content_request")
    if _contains_any(normalized, CRITICISM_KEYWORDS):
        labels.append("criticism")
    if _contains_any(normalized, PRAISE_KEYWORDS):
        labels.append("praise")
    if _contains_any(normalized, BUG_KEYWORDS):
        labels.append("bug_or_access_problem")
    if _contains_any(normalized, METADATA_KEYWORDS):
        labels.append("metadata_issue")

    if not labels:
        labels = ["unclear"]

    confidence = _estimate_confidence(labels, safety_flag)
    return ClassifiedFeedback(
        message_id=message.message_id,
        user_message=message.user_message,
        labels=labels,
        body_region=_detect_body_region(normalized),
        therapy_goal=_detect_therapy_goal(normalized),
        difficulty_requested=_detect_difficulty(normalized),
        equipment=_detect_equipment(normalized),
        position=_detect_position(normalized),
        sentiment=_detect_sentiment(labels),
        safety_flag=safety_flag,
        evidence_quote=text[:160],
        summary=_build_summary(labels, message.message_id),
        confidence=confidence,
        routing="needs_review",
    )


def _contains_any(text: str, keywords: list[str]) -> bool:
    return any(keyword in text for keyword in keywords)


def _detect_body_region(text: str) -> str | None:
    if "knie" in text:
        return "knee"
    if "rücken" in text:
        return "back"
    if "schulter" in text:
        return "shoulder"
    if "hüfte" in text:
        return "general"
    return None


def _detect_therapy_goal(text: str) -> str | None:
    if "mobilisation" in text or "beweglichkeit" in text:
        return "mobility"
    if "balance" in text or "stabil" in text:
        return "stability"
    if "kräft" in text or "stärke" in text or "schwerere" in text:
        return "strength"
    return None


def _detect_difficulty(text: str) -> str | None:
    if "leichter" in text or "leichtere" in text or "anfänger" in text or "senioren" in text:
        return "beginner"
    if "schwerere" in text or "fortgeschritten" in text:
        return "advanced"
    if "schwer" in text:
        return "beginner"
    return None


def _detect_equipment(text: str) -> str | None:
    if "ohne geräte" in text:
        return "none"
    if "theraband" in text:
        return "theraband"
    if "miniband" in text:
        return "miniband"
    if "stuhl" in text or "sitzen" in text:
        return "chair"
    return None


def _detect_position(text: str) -> str | None:
    if "sitzen" in text or "stuhl" in text:
        return "seated"
    if "liegen" in text:
        return "supine"
    return None


def _detect_sentiment(labels: list[str]) -> str | None:
    if "safety_signal" in labels:
        return "concerned"
    if "praise" in labels and "criticism" not in labels:
        return "positive"
    if "criticism" in labels or "bug_or_access_problem" in labels:
        return "negative"
    return "neutral"


def _estimate_confidence(labels: list[str], safety_flag: bool) -> float:
    if safety_flag:
        return 0.9
    if labels == ["unclear"]:
        return 0.45
    if len(labels) > 2:
        return 0.72
    return 0.82


def _build_summary(labels: list[str], message_id: str) -> str:
    return f"{message_id} classified as {', '.join(labels)} by mock keyword rules."
