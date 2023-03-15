import streamlit as st
from langchain.chains import LLMChain
from langchain.llms import OpenAI

from doccano_mini.examples import select_example_df
from doccano_mini.prompts import select_prompt_maker
from doccano_mini.tasks import options

CODE = """from langchain.chains import load_chain

chain = load_chain("chain.yaml")
chain.run("YOUR TEXT")"""


def display_download_button():
    st.header("Download LangChain's config")
    with open("config.yaml", "r", encoding="utf-8") as f:
        st.download_button(
            label="Download",
            data=f,
            file_name="config.yaml",
        )


def main():
    st.title("doccano-mini")

    st.header("Select your task")
    task = st.selectbox("", options=options, label_visibility="collapsed")

    st.header("Annotate your data")
    df = select_example_df(task)
    edited_df = st.experimental_data_editor(df, num_rows="dynamic", width=1000)
    examples = edited_df.to_dict(orient="records")

    st.header("Test")
    text = st.text_area(label="Please enter your text.", value="")
    if st.button("Predict"):
        prompt = select_prompt_maker(task)(examples)
        llm = OpenAI()
        chain = LLMChain(llm=llm, prompt=prompt)
        response = chain.run(text)
        label = response.split(":")[1]
        st.text(label)

        chain.save("config.yaml")
        display_download_button()

    st.header("Usage")
    st.code(CODE)


if __name__ == "__main__":
    main()
