from typing import Any

from streamlit.runtime.state import SessionStateProxy


class SessionStorage:
    def __init__(self, state: SessionStateProxy) -> None:
        self.state = state

    def init_state(self, key: str, value: Any) -> None:
        if key not in self.state:
            self.state[key] = value

    def set_state(self, key: str, value: Any, *, do_init: bool = False) -> None:
        if do_init:
            self.init_state(key, value)

        self.state[key] = value

    def get_state(self, key: str) -> Any:
        return self.state.get(key, None)
