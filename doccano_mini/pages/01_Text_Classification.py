from typing import Dict, List

import streamlit as st

from doccano_mini.layout import BasePage
from doccano_mini.prompts import make_classification_prompt


class TextClassificationPage(BasePage):
    def make_examples(self, columns: List[str]):
        df = self.load_examples("text_classification.json")
        edited_df = st.experimental_data_editor(df, num_rows="dynamic", width=1000)
        examples = edited_df.to_dict(orient="records")
        return examples

    def make_prompt(self, examples: List[Dict]):
        return make_classification_prompt(examples)

    def prepare_inputs(self, columns: List[str]):
        return {"input": st.text_area(label="Please enter your text.", value="", height=300)}


page = TextClassificationPage(title="Text Classification")
page.render()
