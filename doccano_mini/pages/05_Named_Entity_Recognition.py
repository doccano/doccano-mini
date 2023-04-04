from typing import Dict, List

import pandas as pd
import streamlit as st
from st_ner_annotate import st_ner_annotate

from doccano_mini.layout import BasePage
from doccano_mini.prompts import make_named_entity_recognition_prompt
from doccano_mini.storages.entity import EntitySessionStorage
from doccano_mini.storages.stepper import StepperSessionStorage


class NamedEntityRecognitionPage(BasePage):
    example_path = "named_entity_recognition.json"

    def __init__(self, title: str) -> None:
        super().__init__(title)
        self.types: List[str] = []
        self.entity_repository = EntitySessionStorage()
        self.stepper_repository = StepperSessionStorage()

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
        col1.button("Prev", on_click=self.stepper_repository.decrement, args=(len(examples),))
        col2.button("Next", on_click=self.stepper_repository.increment, args=(len(examples),))

        self.stepper_repository.fit(len(examples))
        step = self.stepper_repository.get_step()
        text = examples[step]["text"]
        entities = self.entity_repository.find_by_text(text)
        entities = st_ner_annotate(selected_type, text, entities, key=text)
        self.entity_repository.store_by_text(text, entities)
        return examples

    def make_prompt(self, examples: List[Dict]):
        examples = [
            {**example, "entities": self.entity_repository.find_by_text(example["text"])} for example in examples
        ]
        return make_named_entity_recognition_prompt(examples, types=self.types)

    def prepare_inputs(self, columns: List[str]):
        return {"text": st.text_area(label="Please enter your text.", value="", height=300)}


page = NamedEntityRecognitionPage(title="Named Entity Recognition")
page.render()
