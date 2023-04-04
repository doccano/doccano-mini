import streamlit as st

from doccano_mini.models.stepper import Stepper
from doccano_mini.storages.session_storage import SessionStorage


class StepperSessionStorage:
    def __init__(self) -> None:
        self.storage = SessionStorage(state=st.session_state)
        self.storage.init_state("step", 0)

    def get_step(self) -> int:
        return self.storage.get_state("step")

    def fit(self, total: int) -> None:
        step = self.storage.get_state("step")
        stepper = Stepper(step)
        stepper.fit(total)
        self.storage.set_state("step", stepper.step)

    def increment(self, total: int) -> None:
        step = self.storage.get_state("step")
        stepper = Stepper(step)
        stepper.increment(total)
        self.storage.set_state("step", stepper.step)

    def decrement(self, total: int) -> None:
        step = self.storage.get_state("step")
        stepper = Stepper(step)
        stepper.decrement(total)
        self.storage.set_state("step", stepper.step)
