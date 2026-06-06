import re

from src.schemas import ClassifiedFeedback, FeedbackMessage


SAFETY_KEYWORDS = ["pain", "schmerzen", "schwindelig", "unsicher", "op darf"]
REQUEST_KEYWORDS = [
    "mehr",
    "hätte gern",
    "haette gern",
    "bitte",
    "ich brauche",
    "ich will",
    "ich möchte",
    "ich moechte",
    "suche",
    "gibt es",
    "könnt ihr",
    "koennt ihr",
]
CRITICISM_KEYWORDS = ["schwer", "zu lang", "nervt", "problem"]
PRAISE_KEYWORDS = ["super", "gut", "verständlich"]
BUG_KEYWORDS = ["startbutton", "reagiert nicht", "app"]
METADATA_KEYWORDS = ["tag", "steht", "passt nicht", "anfänger"]

RULE_GATE_SAFETY_PHRASES = [
    "starke schmerzen",
    "schmerzen bei der übung",
    "schmerz bei der übung",
    "mir wird schwindelig",
    "schwindelig",
    "wurde schlimmer",
    "verschlimmert",
    "nach meiner op",
    "nach der op",
    "darf ich",
    "unsicher ob ich",
]
RULE_GATE_BUG_PHRASES = [
    "video lädt nicht",
    "ton fehlt",
    "app stürzt ab",
    "lässt sich nicht öffnen",
    "link funktioniert nicht",
    "login funktioniert nicht",
]
RULE_GATE_PRAISE_PHRASES = [
    "sehr verständlich",
    "super erklärt",
    "hat mir gut geholfen",
    "gute erklärung",
    "kurze übungen gefallen mir",
]
RULE_GATE_PRAISE_BLOCKERS = [
    "bitte",
    "ich brauche",
    "ich will",
    "ich möchte",
    "ich moechte",
    "hätte gern",
    "haette gern",
    "mehr",
    "suche",
    "gibt es",
    "schmerzen",
    "schwindelig",
    "unsicher",
    "darf ich",
]
RULE_GATE_REQUEST_PHRASES = [
    "ich will",
    "ich möchte",
    "ich moechte",
    "ich brauche",
    "ich suche",
    "suche",
    "gibt es",
    "hätte gern",
    "haette gern",
    "könnt ihr",
    "koennt ihr",
]


def classify_feedback_mock(message: FeedbackMessage) -> ClassifiedFeedback:
    """Classify feedback with deterministic keyword rules."""
    text = message.user_message
    normalized = text.casefold()
    body_region = _detect_body_region(normalized)
    therapy_goal = _detect_therapy_goal(normalized)
    difficulty_requested = _detect_difficulty(normalized)
    equipment = _detect_equipment(normalized)
    position = _detect_position(normalized)
    metadata_source = (
        "mock_rules"
        if any((body_region, therapy_goal, difficulty_requested, equipment, position))
        else None
    )

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
        body_region=body_region,
        therapy_goal=therapy_goal,
        difficulty_requested=difficulty_requested,
        equipment=equipment,
        position=position,
        metadata_source=metadata_source,
        sentiment=_detect_sentiment(labels),
        safety_flag=safety_flag,
        evidence_quote=text[:160],
        summary=_build_summary(labels, message.message_id),
        confidence=confidence,
        routing="needs_review",
        classifier_source="mock",
        abstained=False,
    )


def enrich_classification_metadata(
    classified: ClassifiedFeedback,
) -> ClassifiedFeedback:
    """Backfill deterministic request metadata for rule-based modes."""
    normalized = classified.user_message.casefold()
    if classified.body_region is None:
        classified.body_region = _detect_body_region(normalized)
    if classified.therapy_goal is None:
        classified.therapy_goal = _detect_therapy_goal(normalized)
    if classified.difficulty_requested is None:
        classified.difficulty_requested = _detect_difficulty(normalized)
    if classified.equipment is None:
        classified.equipment = _detect_equipment(normalized)
    if classified.position is None:
        classified.position = _detect_position(normalized)
    return classified


def classify_with_rule_gates(message: FeedbackMessage) -> ClassifiedFeedback | None:
    normalized = message.user_message.casefold()
    if _contains_any(normalized, RULE_GATE_SAFETY_PHRASES):
        return ClassifiedFeedback(
            message_id=message.message_id,
            user_message=message.user_message,
            labels=["safety_signal"],
            sentiment="negative",
            safety_flag=True,
            evidence_quote=message.user_message[:120],
            summary="Rule gate detected a safety-related signal.",
            confidence=0.99,
            routing="safety_review",
            classifier_source="hybrid_rule_safety",
            abstained=False,
        )

    if _contains_any(normalized, RULE_GATE_BUG_PHRASES):
        return ClassifiedFeedback(
            message_id=message.message_id,
            user_message=message.user_message,
            labels=["bug_or_access_problem"],
            sentiment="negative",
            safety_flag=False,
            evidence_quote=message.user_message[:120],
            summary="Rule gate detected a technical or access issue.",
            confidence=0.95,
            routing="process",
            classifier_source="hybrid_rule_bug",
            abstained=False,
        )

    if _contains_any(normalized, RULE_GATE_PRAISE_PHRASES) and not _contains_any(
        normalized, RULE_GATE_PRAISE_BLOCKERS
    ):
        return ClassifiedFeedback(
            message_id=message.message_id,
            user_message=message.user_message,
            labels=["praise"],
            sentiment="positive",
            safety_flag=False,
            evidence_quote=message.user_message[:120],
            summary="Rule gate detected clear praise.",
            confidence=0.93,
            routing="process",
            classifier_source="hybrid_rule_praise",
            abstained=False,
        )

    if _contains_any(normalized, RULE_GATE_REQUEST_PHRASES):
        return ClassifiedFeedback(
            message_id=message.message_id,
            user_message=message.user_message,
            labels=["content_request"],
            sentiment="neutral",
            safety_flag=False,
            evidence_quote=message.user_message[:120],
            summary="Rule gate detected a content request phrase.",
            confidence=0.88,
            routing="process",
            classifier_source="hybrid_rule_request",
            abstained=False,
        )

    return None


def _contains_any(text: str, keywords: list[str]) -> bool:
    return any(keyword in text for keyword in keywords)


def _detect_body_region(text: str) -> str | None:
    if "knie" in text:
        return "knee"
    if "rücken" in text or "ruecken" in text:
        return "back"
    if "schulter" in text:
        return "shoulder"
    if "hüfte" in text or "huefte" in text or "hüft" in text or "hueft" in text:
        return "hip"
    if "sprunggelenk" in text or "knöchel" in text:
        return "ankle"
    if "nacken" in text or "nack" in text:
        return "neck"
    if "kiefer" in text or "zähnepressen" in text or "zaehnepressen" in text or "tmj" in text:
        return "jaw"
    if "allgemein" in text:
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
    if (
        "leichter" in text
        or "leichtere" in text
        or "anfänger" in text
        or "senioren" in text
        or "ältere leute" in text
        or "aeltere leute" in text
        or "einfache variante" in text
        or "leicht" in text
    ):
        return "beginner"
    if (
        "schwerere" in text
        or "fortgeschritten" in text
        or "anspruchsvoller" in text
        or "anspruchsvoll" in text
    ):
        return "advanced"
    if "schwer" in text:
        return "beginner"
    return None


def _detect_equipment(text: str) -> str | None:
    if "ohne geräte" in text:
        return "none"
    if "ohne equipment" in text:
        return "none"
    if "theraband" in text:
        return "theraband"
    if "miniband" in text:
        return "miniband"
    if "stuhl" in text:
        return "chair"
    if "wand" in text:
        return "wall"
    if "matte" in text:
        return "mat"
    if "handtuch" in text:
        return "towel"
    return None


def _detect_position(text: str) -> str | None:
    if "sitzen" in text or "stuhl" in text:
        return "sitting"
    if "liegen" in text:
        return "lying"
    if "knien" in text:
        return "kneeling"
    if "vierfüßler" in text or "vierfuessler" in text:
        return "all_fours"
    if "wand" in text or re.search(
        r"\b(?:im stand|im stehen|standposition|stehen|stehend)\b", text
    ):
        return "standing"
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
