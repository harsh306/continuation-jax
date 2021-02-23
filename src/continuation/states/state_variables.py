from abc import ABC, abstractmethod
from typing import Dict
import jsonlines
import json
import jax.numpy as np
import glob, os


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
        self.file_name = file_name
        self._clear()
        self.writer = jsonlines.Writer(
            open(file_name, mode="a", encoding="utf-8"), dumps=NumpyEncoder().encode
        )

    def _clear(self):
        try:
            for f in glob.glob(self.file_name):
                os.remove(f)
            print(f"Clearing previous runs of {self.file_name}")
        except Exception as e:
            print(f"No previos runs. {e}")

    def write(self, record: list):
        """Write/Append the record to the file."""
        self.writer.write(record)
