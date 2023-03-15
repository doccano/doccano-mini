import pandas as pd

from .tasks import TaskType


def select_example_df(task: TaskType) -> pd.DataFrame:
    if task == TaskType.TEXT_CLASSIFICATION.value:
        df = pd.DataFrame(
            [
                {"text": "That would be awesome!", "label": "positive"},
                {"text": "This is awful!", "label": "negative"},
                {"text": "Today is hot day.", "label": "neutral"},
            ],
            columns=["text", "label"],
        )
        return df
    raise ValueError(f"Task {task} is not supported.")
