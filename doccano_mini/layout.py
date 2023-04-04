from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, List

import pandas as pd
import streamlit as st
from langchain.chains import LLMChain
from langchain.prompts.few_shot import FewShotPromptTemplate

from doccano_mini.components import (
    display_download_button,
    openai_model_form,
    task_instruction_editor,
    usage,
)
from doccano_mini.utils import escape_markdown


class BasePage(ABC):
    example_path: str = ""

    def __init__(self, title: str) -> None:
        self.title = title

    @property
    def columns(self) -> List[str]:
        return []

    def load_examples(self, filename: str) -> pd.DataFrame:
        filepath = Path(__file__).parent.resolve().joinpath("examples", filename)
        return pd.read_json(filepath)

    def make_examples(self, columns: List[str]) -> List[Dict]:
        df = self.load_examples(self.example_path)
        edited_df = st.experimental_data_editor(df, num_rows="dynamic", width=1000)
        examples = edited_df.to_dict(orient="records")
        return examples

    @abstractmethod
    def make_prompt(self, examples: List[Dict]) -> FewShotPromptTemplate:
        raise NotImplementedError()

    @abstractmethod
    def prepare_inputs(self, columns: List[str]) -> Dict:
        raise NotImplementedError()

    def annotate(self, examples: List[Dict]) -> List[Dict]:
        return examples

    def render(self) -> None:
        st.title(self.title)
        st.header("Annotate your data")
        columns = self.columns
        examples = self.make_examples(columns)
        examples = self.annotate(examples)

        prompt = self.make_prompt(examples)
        prompt = task_instruction_editor(prompt)

        st.header("Test")
        col1, col2 = st.columns([3, 1])

        with col1:
            inputs = self.prepare_inputs(columns)

        with col2:
            llm = openai_model_form()

        with st.expander("See your prompt"):
            st.markdown(f"```\n{prompt.format(**inputs)}\n```")

        if llm is None:
            st.error("Enter your API key.")

        if st.button("Predict", disabled=llm is None):
            chain = LLMChain(llm=llm, prompt=prompt)  # type:ignore
            response = chain.run(**inputs)
            st.markdown(escape_markdown(response).replace("\n", "  \n"))

            chain.save("config.yaml")
            display_download_button()
        usage()
