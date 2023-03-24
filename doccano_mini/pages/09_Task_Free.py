import os

import streamlit as st
from langchain.chains import LLMChain
from langchain.llms import OpenAI

from doccano_mini.components import display_download_button, display_usage
from doccano_mini.examples import make_task_free_example
from doccano_mini.models import AVAILABLE_MODELS
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

prompt.prefix = st.text_area(
    label="Enter task instruction",
    placeholder=f"Predict {columns[-1]} based on {', '.join(columns[:-1])}.",
    height=200,
)

inputs = {column: st.text_input(f"Input for {column}:") for column in columns[:-1]}

st.markdown(f"Your prompt\n```\n{prompt.format(**inputs)}\n```")

# Use text-davinci-003 by default.
api_key = st.text_input("Enter API key", value=os.environ.get("OPENAI_API_KEY", ""), type="password")
model_name = st.selectbox("Model", AVAILABLE_MODELS, index=2)
temperature = st.slider("Temperature", min_value=0.0, max_value=1.0, value=0.7, step=0.01)
top_p = st.slider("Top-p", min_value=0.0, max_value=1.0, value=1.0, step=0.01)
if st.button("Predict"):
    llm = OpenAI(model_name=model_name, temperature=temperature, top_p=top_p, openai_api_key=api_key)  # type:ignore
    chain = LLMChain(llm=llm, prompt=prompt)
    response = chain.run(**inputs)
    st.text(response)

    chain.save("config.yaml")
    display_download_button()
display_usage()
