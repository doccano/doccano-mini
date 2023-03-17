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
    elif task == TaskType.TASK_FREE.value:
        df = pd.DataFrame([{"Column 1": "", "Column 2": ""}], columns=["Column 1", "Column 2"])
        return df
    raise ValueError(f"Task {task} is not supported.")
