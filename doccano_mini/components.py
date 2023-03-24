import os

import streamlit as st
from langchain.llms import OpenAI
from langchain.schema import BaseLanguageModel

CODE = """from langchain.chains import load_chain

chain = load_chain("chain.yaml")
chain.run("YOUR TEXT")"""


def display_download_button():
    st.header("Download a config file")
    with open("config.yaml", "r", encoding="utf-8") as f:
        st.download_button(
            label="Download",
            data=f,
            file_name="config.yaml",
        )


def display_usage():
    st.header("Usage")
    st.code(CODE)


def openai_model_form() -> BaseLanguageModel:
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
    return OpenAI(model_name=model_name, temperature=temperature, top_p=top_p, openai_api_key=api_key)  # type:ignore
