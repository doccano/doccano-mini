from collections import defaultdict
from typing import Dict, List

import pandas as pd
import streamlit as st
from st_ner_annotate import st_ner_annotate

from doccano_mini.layout import BasePage
from doccano_mini.prompts import make_named_entity_recognition_prompt

if "step" not in st.session_state:
    st.session_state.step = 0


if "entities" not in st.session_state:
    st.session_state.entities = defaultdict(list)


def increment(total):
    st.session_state.step += 1
    if st.session_state.step >= total:
        st.session_state.step = 0


def decrement(total: int):
    st.session_state.step -= 1
    if st.session_state.step < 0:
        st.session_state.step = total - 1


def load_entities(text: str):
    entities = st.session_state.entities[text]
    return entities


def save_entities(text: str, entities: List[Dict]):
    st.session_state.entities[text] = entities


class NamedEntityRecognitionPage(BasePage):
    example_path = "named_entity_recognition.json"

    def __init__(self, title: str) -> None:
        super().__init__(title)
        self.types: List[str] = []

    def define_entity_types(self):
        st.subheader("Define entity types")
        default_types = pd.DataFrame([{"type": entity_type} for entity_type in ["ORG", "LOC", "PER"]])
        edited_df = st.experimental_data_editor(default_types, num_rows="dynamic", width=1000)
        types = edited_df["type"].values
        self.types = types
        return types

    def annotate(self, examples: List[Dict]) -> List[Dict]:
        if len(examples) == 0:
            return []

        types = self.define_entity_types()
        selected_type = st.selectbox("Select an entity type", types)

        col1, col2, _ = st.columns([1, 1, 8])
        col1.button("Prev", on_click=decrement, args=(len(examples),))
        col2.button("Next", on_click=increment, args=(len(examples),))

        text = examples[st.session_state.step]["text"]
        entities = load_entities(text)
        entities = st_ner_annotate(selected_type, text, entities, key=text)
        save_entities(text, entities)
        return examples

    def make_prompt(self, examples: List[Dict]):
        examples = [{**example, "entities": load_entities(example["text"])} for example in examples]
        return make_named_entity_recognition_prompt(examples, types=self.types)

    def prepare_inputs(self, columns: List[str]):
        return {"text": st.text_area(label="Please enter your text.", value="", height=300)}


page = NamedEntityRecognitionPage(title="Named Entity Recognition")
page.render()
