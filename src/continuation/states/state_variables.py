from abc import ABC, abstractmethod
from typing import Dict
import jsonlines
import json
import jax.numpy as np

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

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

class StateWriter:
    """State Writer will write the state values to a file."""

    def __init__(self, file_name: str):
        """Create a file object"""

        self.writer = jsonlines.Writer(open(file_name, mode="a", encoding='utf-8'), dumps=NumpyEncoder().encode)

    def write(self, record: dict):
        """Write/Append the record to the file."""
        self.writer.write(record)
