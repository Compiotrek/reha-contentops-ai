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


def test_mock_classifier_detects_will_as_content_request() -> None:
    classified = classify_feedback_mock(
        FeedbackMessage(
            message_id="msg_test",
            user_message="Ich will handstandübungen.",
        )
    )

    assert classified.labels == ["content_request"]


def test_mock_classifier_detects_hip_stem() -> None:
    classified = classify_feedback_mock(
        FeedbackMessage(
            message_id="msg_test",
            user_message="Ich suche Hüftübungen mit Miniband.",
        )
    )

    assert classified.body_region == "hip"
    assert classified.equipment == "miniband"


def test_mock_classifier_detects_older_people_as_beginner() -> None:
    classified = classify_feedback_mock(
        FeedbackMessage(
            message_id="msg_test",
            user_message="Bitte mehr Übungen für ältere Leute im Sitzen.",
        )
    )

    assert classified.difficulty_requested == "beginner"
    assert classified.position == "sitting"


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
