import pandas as pd


def make_classification_example() -> pd.DataFrame:
    df = pd.DataFrame(
        [
            {"text": "That would be awesome!", "label": "positive"},
            {"text": "This is awful!", "label": "negative"},
            {"text": "Today is hot day.", "label": "neutral"},
        ],
        columns=["text", "label"],
    )
    return df


def make_task_free_example() -> pd.DataFrame:
    df = pd.DataFrame([{"Column 1": "", "Column 2": ""}], columns=["Column 1", "Column 2"])
    return df
