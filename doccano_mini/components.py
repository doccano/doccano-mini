import os
from pathlib import Path
from typing import Optional

import streamlit as st
from langchain.llms import OpenAI
from langchain.prompts.few_shot import FewShotPromptTemplate
from langchain.schema import BaseLanguageModel


def display_download_button():
    st.header("Download a config file")
    with open("config.yaml", "r", encoding="utf-8") as f:
        st.download_button(
            label="Download",
            data=f,
            file_name="config.yaml",
        )


def usage():
    st.header("Usage")
    filepath = Path(__file__).parent.resolve() / "docs" / "usage.md"
    with filepath.open("r", encoding="utf-8") as f:
        st.markdown(f.read())


def task_instruction_editor(prompt: FewShotPromptTemplate) -> FewShotPromptTemplate:
    st.header("Edit instruction")
    with st.expander("See instruction"):
        prompt.prefix = st.text_area(label="Enter task instruction", value=prompt.prefix, height=200)
    return prompt


def openai_model_form() -> Optional[BaseLanguageModel]:
    # https://platform.openai.com/docs/models/gpt-3-5
    AVAILABLE_MODELS = (
        "gpt-3.5-turbo",
        "gpt-3.5-turbo-0301",
        "text-davinci-003",
        "text-davinci-002",
        "code-davinci-002",
    )
    api_key = st.text_input("API key", value=os.environ.get("OPENAI_API_KEY", ""), type="password")
    model_name = st.selectbox("Model", AVAILABLE_MODELS, index=2)
    temperature = st.slider("Temperature", min_value=0.0, max_value=1.0, value=0.7, step=0.01)
    top_p = st.slider("Top-p", min_value=0.0, max_value=1.0, value=1.0, step=0.01)
    if not api_key:
        return None
    return OpenAI(model_name=model_name, temperature=temperature, top_p=top_p, openai_api_key=api_key)  # type:ignore
