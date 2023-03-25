from abc import ABC, abstractmethod
from typing import Dict, List

import streamlit as st
from langchain.chains import LLMChain
from langchain.prompts.few_shot import FewShotPromptTemplate

from doccano_mini.components import (
    display_download_button,
    openai_model_form,
    task_instruction_editor,
    usage,
)


class BasePage(ABC):
    def __init__(self, title: str) -> None:
        self.title = title

    @property
    def columns(self) -> List[str]:
        return []

    @abstractmethod
    def make_examples(self, columns: List[str]) -> List[Dict]:
        raise NotImplementedError()

    @abstractmethod
    def make_prompt(self, examples: List[Dict]) -> FewShotPromptTemplate:
        raise NotImplementedError()

    @abstractmethod
    def prepare_inputs(self, columns: List[str]) -> Dict:
        raise NotImplementedError()

    def render(self) -> None:
        st.title(self.title)
        st.header("Annotate your data")
        columns = self.columns
        examples = self.make_examples(columns)

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
            st.text(response)

            chain.save("config.yaml")
            display_download_button()
        usage()