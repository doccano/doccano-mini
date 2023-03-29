from typing import Dict, List

import streamlit as st

from doccano_mini.layout import BasePage
from doccano_mini.prompts import make_paraphrase_prompt


class ParaphrasePage(BasePage):
    example_path = "paraphrase.json"

    def make_prompt(self, examples: List[Dict]):
        return make_paraphrase_prompt(examples)

    def prepare_inputs(self, columns: List[str]):
        return {
            "text": st.text_area(label="Text.", value="", height=300),
        }


page = ParaphrasePage(title="Paraphrase")
page.render()
