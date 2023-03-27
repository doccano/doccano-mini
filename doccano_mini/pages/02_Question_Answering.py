from typing import Dict, List

import streamlit as st

from doccano_mini.layout import BasePage
from doccano_mini.prompts import make_question_answering_prompt


class QuestionAnsweringPage(BasePage):
    def make_examples(self, columns: List[str]):
        df = self.load_examples("question_answering.json")
        edited_df = st.experimental_data_editor(df, num_rows="dynamic", width=1000)
        examples = edited_df.to_dict(orient="records")
        return examples

    def make_prompt(self, examples: List[Dict]):
        return make_question_answering_prompt(examples)

    def prepare_inputs(self, columns: List[str]):
        return {
            "context": st.text_area(label="Context.", value="", height=300),
            "question": st.text_area(label="Question.", value="", height=100),
        }


page = QuestionAnsweringPage(title="Question Answering")
page.render()
