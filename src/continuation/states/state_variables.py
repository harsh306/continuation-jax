from abc import ABC, abstractmethod
from typing import Dict


class StateVariable:
    def __init__(self, state: list, counter: int):
        self._state = state
        self._counter = counter

    @property
    def state(self):
        return self._state

    @state.setter
    def state(self, state):
        self._state = state

    @property
    def counter(self):
        return self._counter

    @counter.setter
    def counter(self, counter):
        self._counter = counter

    def get_record(self) -> Dict:
        return {self._counter: self._state}


class StateWriter:
    def __init__(self, file_name: str):
        self.file = open(file_name, "a")

    def write(self, record):
        self.file.write(f"\n{record}")
