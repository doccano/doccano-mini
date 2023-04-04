class Stepper:
    def __init__(self, step=0):
        self._step = step

    @property
    def step(self) -> int:
        return self._step

    def fit(self, total: int):
        if self._step >= total:
            self._step = total - 1

    def at(self, step: int, total: int):
        if step >= total:
            raise ValueError(f"step must be less than {total}")
        if step < 0:
            raise ValueError("step must be greater than 0")
        self._step = step

    def increment(self, total: int):
        self._step += 1
        if self._step >= total:
            self._step = 0

    def decrement(self, total: int):
        self._step -= 1
        if self._step < 0:
            self._step = total - 1
