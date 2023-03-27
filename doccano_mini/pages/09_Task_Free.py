from typing import Dict, List

import streamlit as st

from doccano_mini.layout import BasePage
from doccano_mini.prompts import make_task_free_prompt


class TaskFreePage(BasePage):
    @property
    def columns(self) -> List[str]:
        num_cols = st.number_input("Set the number of columns", min_value=2, max_value=10)
        columns = [st.text_input(f"Column {i + 1}:", value=f"column {i + 1}") for i in range(int(num_cols))]
        return columns

    def make_examples(self, columns: List[str]):
        df = self.load_examples("task_free.json")
        df = df.reindex(columns, axis="columns", fill_value="")
        edited_df = st.experimental_data_editor(df, num_rows="dynamic", width=1000)
        examples = edited_df.to_dict(orient="records")
        return examples

    def make_prompt(self, examples: List[Dict]):
        return make_task_free_prompt(examples)

    def prepare_inputs(self, columns: List[str]):
        return {column: st.text_area(label=f"Input for {column}:", value="", height=300) for column in columns[:-1]}


page = TaskFreePage(title="Task Free")
page.render()
