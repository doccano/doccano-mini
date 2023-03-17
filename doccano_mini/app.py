import streamlit as st
from langchain.chains import LLMChain
from langchain.llms import OpenAI

from doccano_mini.examples import select_example_df
from doccano_mini.prompts import select_prompt_maker
from doccano_mini.tasks import TaskType, options

CODE = """from langchain.chains import load_chain

chain = load_chain("chain.yaml")
chain.run("YOUR TEXT")"""


# https://platform.openai.com/docs/models/gpt-3-5
AVAILABLE_MODELS = (
    "gpt-3.5-turbo",
    "gpt-3.5-turbo-0301",
    "text-davinci-003",
    "text-davinci-002",
    "code-davinci-002",
)


def display_download_button():
    st.header("Download LangChain's config")
    with open("config.yaml", "r", encoding="utf-8") as f:
        st.download_button(
            label="Download",
            data=f,
            file_name="config.yaml",
        )


def task_classification(task: TaskType):
    st.header("Annotate your data")
    df = select_example_df(task)
    edited_df = st.experimental_data_editor(df, num_rows="dynamic", width=1000)
    examples = edited_df.to_dict(orient="records")

    # Create prompt
    prompt = select_prompt_maker(task)(examples)

    st.header("Optional: Edit instruction")
    with st.expander("See instruction"):
        instruction = st.text_area(label="Instruction", value=prompt.prefix, height=200)
        prompt.prefix = instruction

    st.header("Test")
    col1, col2 = st.columns([3, 1])
    text = col1.text_area(label="Please enter your text.", value="", height=300)

    # Use text-davinci-003 by default.
    model_name = col2.selectbox("Model", AVAILABLE_MODELS, index=2)
    temperature = col2.slider("Temperature", min_value=0.0, max_value=1.0, value=0.7, step=0.01)
    top_p = col2.slider("Top-p", min_value=0.0, max_value=1.0, value=1.0, step=0.01)

    with st.expander("See your prompt"):
        st.markdown(f"```\n{prompt.format(input=text)}\n```")

    if st.button("Predict"):
        llm = OpenAI(model_name=model_name, temperature=temperature, top_p=top_p)  # type:ignore
        chain = LLMChain(llm=llm, prompt=prompt)
        response = chain.run(text)
        label = response.split(":")[1]
        st.text(label)

        chain.save("config.yaml")
        display_download_button()


def task_free(task: TaskType):
    st.header("Annotate your data")

    num_cols = st.number_input("Set the number of columns", min_value=2, max_value=10)
    columns = [st.text_input(f"Column {i}:", value=f"column {i}") for i in range(1, int(num_cols) + 1)]

    df = select_example_df(task)
    df = df.reindex(columns, axis="columns", fill_value="")
    edited_df = st.experimental_data_editor(df, num_rows="dynamic", width=1000)
    examples = edited_df.to_dict(orient="records")

    prompt = select_prompt_maker(task)(examples)

    prompt.prefix = st.text_area(
        label="Enter task instruction",
        placeholder=f"Predict {columns[-1]} based on {', '.join(columns[:-1])}.",
        height=200,
    )

    inputs = {column: st.text_input(f"Input for {column}:") for column in columns[:-1]}

    st.markdown(f"Your prompt\n```\n{prompt.format(**inputs)}\n```")

    # Use text-davinci-003 by default.
    model_name = st.selectbox("Model", AVAILABLE_MODELS, index=2)
    temperature = st.slider("Temperature", min_value=0.0, max_value=1.0, value=0.7, step=0.01)
    top_p = st.slider("Top-p", min_value=0.0, max_value=1.0, value=1.0, step=0.01)
    if st.button("Predict"):
        llm = OpenAI(model_name=model_name, temperature=temperature, top_p=top_p)  # type:ignore
        chain = LLMChain(llm=llm, prompt=prompt)
        response = chain.run(**inputs)
        st.text(response)

        chain.save("config.yaml")
        display_download_button()


def main():
    st.set_page_config(page_title="doccano-mini", page_icon=":memo:")

    st.title("doccano-mini")

    st.header("Select your task")
    task = st.selectbox("", options=options, label_visibility="collapsed")

    if task == TaskType.TEXT_CLASSIFICATION.value:
        task_classification(task)
    else:
        task_free(task)

    st.header("Usage")
    st.code(CODE)


if __name__ == "__main__":
    main()
