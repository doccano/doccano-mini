from typing import Dict, List

import streamlit as st

from doccano_mini.layout import BasePage
from doccano_mini.prompts import make_question_answering_prompt


class QuestionAnsweringPage(BasePage):
    example_path = "question_answering.json"

    def make_prompt(self, examples: List[Dict]):
        return make_question_answering_prompt(examples)

    def prepare_inputs(self, columns: List[str]):
        return {
            "context": st.text_area(label="Context.", value="", height=300),
            "question": st.text_input(label="Question.", value=""),
        }


page = QuestionAnsweringPage(title="Question Answering")
page.render()
