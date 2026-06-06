from src.classify import classify_feedback_hybrid, classify_with_rule_gates
from src.schemas import ClassifiedFeedback, FeedbackMessage


def _message(text: str) -> FeedbackMessage:
    return FeedbackMessage(message_id="msg_test", user_message=text)


def test_safety_rule_gate_routes_to_safety_review() -> None:
    classified = classify_with_rule_gates(
        _message("Mir wird schwindelig bei der Übung.")
    )

    assert classified is not None
    assert classified.labels == ["safety_signal"]
    assert classified.safety_flag is True
    assert classified.routing == "safety_review"
    assert classified.classifier_source == "hybrid_rule_safety"


def test_hybrid_returns_safety_rule_without_llm(monkeypatch) -> None:
    def fail_if_called(message):
        raise AssertionError("LLM should not be called for rule-gated safety.")

    monkeypatch.setattr("src.classify.classify_feedback_llm", fail_if_called)

    classified = classify_feedback_hybrid(
        _message("Ich habe starke Schmerzen bei der Übung.")
    )

    assert classified.routing == "safety_review"
    assert classified.classifier_source == "hybrid_rule_safety"


def test_bug_rule_gate_detects_access_problem() -> None:
    classified = classify_with_rule_gates(_message("Das Video lädt nicht."))

    assert classified is not None
    assert classified.labels == ["bug_or_access_problem"]
    assert classified.routing == "process"
    assert classified.classifier_source == "hybrid_rule_bug"


def test_praise_rule_gate_detects_clear_praise() -> None:
    classified = classify_with_rule_gates(_message("Das Video war sehr verständlich."))

    assert classified is not None
    assert classified.labels == ["praise"]
    assert classified.routing == "process"
    assert classified.classifier_source == "hybrid_rule_praise"


def test_praise_rule_gate_does_not_trigger_when_request_word_present() -> None:
    classified = classify_with_rule_gates(
        _message("Das Video war sehr verständlich, bitte mehr davon.")
    )

    assert classified is None


def test_request_rule_gate_detects_will_request() -> None:
    classified = classify_with_rule_gates(_message("Ich will handstandübungen."))

    assert classified is not None
    assert classified.labels == ["content_request"]
    assert classified.routing == "process"
    assert classified.safety_flag is False
    assert classified.classifier_source == "hybrid_rule_request"


def test_safety_rule_gate_still_wins_over_request_phrase() -> None:
    classified = classify_with_rule_gates(
        _message("Ich will Übungen, aber mir wird schwindelig.")
    )

    assert classified is not None
    assert classified.labels == ["safety_signal"]
    assert classified.routing == "safety_review"


def test_normal_content_request_does_not_trigger_rule_gate() -> None:
    classified = classify_with_rule_gates(_message("Bitte mehr Knieübungen ohne Geräte."))

    assert classified is None


def test_hybrid_continues_existing_logic_when_rule_gate_returns_none(monkeypatch) -> None:
    ml_result = ClassifiedFeedback(
        message_id="msg_test",
        user_message="Bitte mehr Knieübungen ohne Geräte.",
        labels=["content_request"],
        safety_flag=False,
        evidence_quote="Bitte mehr Knieübungen ohne Geräte.",
        summary="ML accepted.",
        confidence=0.9,
        routing="process",
        classifier_source="ml",
        abstained=False,
        top2_margin=0.2,
    )
    monkeypatch.setattr("src.classify.classify_feedback_ml", lambda message: ml_result)

    classified = classify_feedback_hybrid(_message("Bitte mehr Knieübungen ohne Geräte."))

    assert classified.classifier_source == "hybrid_ml"


def test_hybrid_accepts_llm_only_when_not_abstained(monkeypatch) -> None:
    ml_result = ClassifiedFeedback(
        message_id="msg_test",
        user_message="Komische Nachricht ohne klares Muster.",
        labels=["unclear"],
        safety_flag=False,
        evidence_quote="Komische Nachricht ohne klares Muster.",
        summary="ML abstained.",
        confidence=0.4,
        routing="needs_review",
        classifier_source="ml_abstain",
        abstained=True,
        top2_margin=0.01,
    )
    llm_result = ClassifiedFeedback(
        message_id="msg_test",
        user_message="Komische Nachricht ohne klares Muster.",
        labels=["unclear"],
        safety_flag=False,
        evidence_quote="Komische Nachricht ohne klares Muster.",
        summary="LLM fallback.",
        confidence=0.0,
        routing="needs_review",
        classifier_source="llm",
        abstained=True,
        abstain_reason="LLM failed.",
    )
    monkeypatch.setattr("src.classify.classify_feedback_ml", lambda message: ml_result)
    monkeypatch.setattr("src.classify.classify_feedback_llm", lambda message: llm_result)

    classified = classify_feedback_hybrid(
        _message("Komische Nachricht ohne klares Muster.")
    )

    assert classified.classifier_source == "ml_abstain"
    assert "LLM fallback failed" in (classified.abstain_reason or "")
