from typing import Dict, List

import streamlit as st

from doccano_mini.layout import BasePage
from doccano_mini.prompts import make_classification_prompt


class TextClassificationPage(BasePage):
    example_path = "text_classification.json"

    def make_prompt(self, examples: List[Dict]):
        return make_classification_prompt(examples)

    def prepare_inputs(self, columns: List[str]):
        return {"input": st.text_area(label="Please enter your text.", value="", height=300)}


page = TextClassificationPage(title="Text Classification")
page.render()
