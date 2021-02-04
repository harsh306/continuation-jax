from abc import ABC, abstractmethod
from typing import Dict


class StateVariable:
    """To track the state of problem"""

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
        """Get the state as indexed by the continuation counter"""
        return {self._counter: self._state}


class StateWriter:
    """State Writer will write the state values to a file."""

    def __init__(self, file_name: str):
        """Create a file object"""
        self.file = open(file_name, "a")

    def write(self, record):
        """Write/Append the record to the file."""
        self.file.write(f"\n{record}")
