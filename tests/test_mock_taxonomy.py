from src.classify import classify_feedback_mock
from src.schemas import FeedbackMessage


def test_mock_classifier_detects_new_dataset_taxonomy_values() -> None:
    classified = classify_feedback_mock(
        FeedbackMessage(
            message_id="msg_test",
            user_message="Ich suche Sprunggelenkübungen in einfacher Sprache und an der Wand.",
        )
    )

    assert classified.body_region == "ankle"
    assert classified.equipment == "wall"
    assert classified.position == "standing"


def test_mock_classifier_does_not_detect_widerstandsband_as_standing() -> None:
    classified = classify_feedback_mock(
        FeedbackMessage(
            message_id="msg_test",
            user_message="Ich brauche Übungen mit Widerstandsband.",
        )
    )

    assert classified.position != "standing"


def test_mock_classifier_does_not_detect_widerstand_as_standing() -> None:
    classified = classify_feedback_mock(
        FeedbackMessage(
            message_id="msg_test",
            user_message="Ich suche Übungen gegen Widerstand.",
        )
    )

    assert classified.position != "standing"


def test_mock_classifier_detects_im_stand_as_standing() -> None:
    classified = classify_feedback_mock(
        FeedbackMessage(
            message_id="msg_test",
            user_message="Ich möchte Übungen im Stand.",
        )
    )

    assert classified.position == "standing"


def test_mock_classifier_detects_stehend_as_standing() -> None:
    classified = classify_feedback_mock(
        FeedbackMessage(
            message_id="msg_test",
            user_message="Ich möchte Übungen stehend machen.",
        )
    )

    assert classified.position == "standing"
