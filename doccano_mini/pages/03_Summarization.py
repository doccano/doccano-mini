from typing import Dict, List

import streamlit as st

from doccano_mini.layout import BasePage
from doccano_mini.prompts import make_summarization_prompt


class SummarizationPage(BasePage):
    example_path = "summarization.json"

    def make_prompt(self, examples: List[Dict]):
        return make_summarization_prompt(examples)

    def prepare_inputs(self, columns: List[str]):
        return {
            "passage": st.text_area(label="Passage.", value="", height=300),
        }


page = SummarizationPage(title="Summarization")
page.render()
