from collections import defaultdict

import pandas as pd
import streamlit as st
from datasets import load_dataset
from langchain.chains import LLMChain
from more_itertools import interleave_longest
from sklearn.metrics import classification_report

from doccano_mini.components import openai_model_form, task_instruction_editor
from doccano_mini.prompts import make_classification_prompt
from doccano_mini.utils import escape_markdown

AVAILABLE_DATASETS = ("imdb", "ag_news", "rotten_tomatoes")


@st.cache_resource
def prepare_dataset(dataset_id):
    # Loading dataset
    dataset = load_dataset(dataset_id, split="train")
    # Splitting dataset
    dataset = dataset.train_test_split(test_size=0.2, stratify_by_column="label", shuffle=True)

    # Preparing indices
    indices_by_label = defaultdict(list)
    for i, x in enumerate(dataset["train"]):
        indices_by_label[x["label"]].append(i)

    return dataset, list(interleave_longest(*indices_by_label.values()))


st.title("Text Classification Evaluation on 🤗 datasets")

st.header("Setup your data")

dataset_id = st.selectbox("Select a dataset", options=AVAILABLE_DATASETS)

dataset, train_indices = prepare_dataset(dataset_id)

train_dataset = dataset["train"]
validation_dataset = dataset["test"]

label_info = train_dataset.features["label"]
num_classes = label_info.num_classes
few_shot_example_size = int(
    st.number_input("Number of examples", min_value=num_classes, max_value=num_classes * 5, value=num_classes)
)

subset = []
for i in range(few_shot_example_size):
    example = train_dataset[train_indices[i]]
    subset.append({"text": example["text"], "label": label_info.int2str(example["label"])})


df = pd.DataFrame(subset)

st.write(df)

prompt = make_classification_prompt(df.to_dict("records"))
prompt = task_instruction_editor(prompt)


st.header("Test")
col1, col2 = st.columns([3, 1])

with col1:
    inputs = {"input": st.text_area(label="Please enter your text.", value="", height=300)}

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

st.subheader("Evaluation")

evaluation_size = int(st.number_input("Number of examples", min_value=5, max_value=validation_dataset.dataset_size))

if llm is None:
    st.error("Enter your API key.")

if st.button("Evaluate", disabled=llm is None):
    chain = LLMChain(llm=llm, prompt=prompt)  # type:ignore
    y_true = []
    y_pred = []
    for i in range(evaluation_size):
        example = validation_dataset[i]
        response = chain.run(input=example["text"])
        y_true.append(label_info.int2str(example["label"]))
        y_pred.append(response.split(":")[-1].strip())

    st.text(classification_report(y_true, y_pred, digits=3))
