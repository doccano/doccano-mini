import os

import streamlit as st
from langchain.chains import LLMChain
from langchain.llms import OpenAI

from doccano_mini.components import display_download_button, display_usage
from doccano_mini.examples import make_classification_example
from doccano_mini.models import AVAILABLE_MODELS
from doccano_mini.prompts import make_classification_prompt

st.title("Text Classification")
st.header("Annotate your data")
df = make_classification_example()
edited_df = st.experimental_data_editor(df, num_rows="dynamic", width=1000)
examples = edited_df.to_dict(orient="records")

# Create prompt
prompt = make_classification_prompt(examples)

st.header("Optional: Edit instruction")
with st.expander("See instruction"):
    instruction = st.text_area(label="Instruction", value=prompt.prefix, height=200)
    prompt.prefix = instruction

st.header("Test")
api_key = st.text_input("Enter API key", value=os.environ.get("OPENAI_API_KEY", ""), type="password")
col1, col2 = st.columns([3, 1])
text = col1.text_area(label="Please enter your text.", value="", height=300)

# Use text-davinci-003 by default.
model_name = col2.selectbox("Model", AVAILABLE_MODELS, index=2)
temperature = col2.slider("Temperature", min_value=0.0, max_value=1.0, value=0.7, step=0.01)
top_p = col2.slider("Top-p", min_value=0.0, max_value=1.0, value=1.0, step=0.01)

with st.expander("See your prompt"):
    st.markdown(f"```\n{prompt.format(input=text)}\n```")

if st.button("Predict"):
    llm = OpenAI(model_name=model_name, temperature=temperature, top_p=top_p, openai_api_key=api_key)  # type:ignore
    chain = LLMChain(llm=llm, prompt=prompt)
    response = chain.run(text)
    label = response.split(":")[1]
    st.text(label)

    chain.save("config.yaml")
    display_download_button()
display_usage()
