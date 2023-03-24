import streamlit as st

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
