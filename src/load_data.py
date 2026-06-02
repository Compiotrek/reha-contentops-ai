import pandas as pd

from src.schemas import Exercise, FeedbackMessage


def load_feedback_messages(path: str) -> list[FeedbackMessage]:
    """Load synthetic feedback messages from CSV."""
    rows = pd.read_csv(path).to_dict(orient="records")
    return [FeedbackMessage(**row) for row in rows]


def load_exercises(path: str) -> list[Exercise]:
    """Load the small exercise database from CSV."""
    rows = pd.read_csv(path).to_dict(orient="records")
    return [Exercise(**row) for row in rows]
