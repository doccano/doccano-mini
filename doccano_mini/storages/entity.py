from collections import defaultdict
from typing import List

import streamlit as st

from doccano_mini.models.entity import Entity
from doccano_mini.storages.session_storage import SessionStorage


class EntitySessionStorage:
    def __init__(self) -> None:
        self.storage = SessionStorage(state=st.session_state)
        self.storage.init_state("entities", defaultdict(list))

    def find_by_text(self, text: str) -> List[Entity]:
        entities = self.storage.get_state("entities")
        return entities.get(text, [])

    def store_by_text(self, text: str, entities: List[Entity]) -> None:
        current_entities = self.storage.get_state("entities")
        current_entities[text] = entities
        self.storage.set_state("entities", current_entities)
