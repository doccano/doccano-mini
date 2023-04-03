class Stepper:
    def __init__(self, total: int):
        self._total = total
        self._step = 0

    @property
    def step(self) -> int:
        return self._step

    def at(self, step: int):
        if step >= self._total:
            raise ValueError(f"step must be less than {self._total}")
        if step < 0:
            raise ValueError("step must be greater than 0")
        self._step = step

    def increment(self):
        self._step += 1
        if self._step >= self._total:
            self._step = 0

    def decrement(self):
        self._step -= 1
        if self._step < 0:
            self._step = self._total - 1
