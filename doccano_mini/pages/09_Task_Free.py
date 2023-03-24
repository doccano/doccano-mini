import streamlit as st
from langchain.chains import LLMChain

from doccano_mini.components import (
    display_download_button,
    display_usage,
    openai_model_form,
    task_instruction_editor,
)
from doccano_mini.examples import make_task_free_example
from doccano_mini.prompts import make_task_free_prompt

st.title("Task Free")
st.header("Annotate your data")
num_cols = st.number_input("Set the number of columns", min_value=2, max_value=10)
columns = [st.text_input(f"Column {i}:", value=f"column {i}") for i in range(1, int(num_cols) + 1)]

df = make_task_free_example()
df = df.reindex(columns, axis="columns", fill_value="")
edited_df = st.experimental_data_editor(df, num_rows="dynamic", width=1000)
examples = edited_df.to_dict(orient="records")

prompt = make_task_free_prompt(examples)
prompt = task_instruction_editor(prompt)

st.header("Test")
col1, col2 = st.columns([3, 1])
inputs = {column: col1.text_area(label=f"Input for {column}:", value="", height=300) for column in columns[:-1]}

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
display_usage()
