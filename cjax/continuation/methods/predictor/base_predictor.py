from typing import Tuple
from abc import ABC, abstractmethod


class AbstractPredictor(ABC):
    """Abstract Predictor to be inherited by developer for any new predictor."""

    @abstractmethod
    def prediction_step(self):
        pass

    @abstractmethod
    def _assign_states(self):
        pass


class Predictor(AbstractPredictor):  # Todo state maintiner ?
    def __init__(self, concat_states):
        self._concat_states = concat_states
        self._state = None
        self._bparam = None

    @property
    def state(self):
        return self._state

    @state.setter
    def state(self, state):
        self._state = state

    @property
    def bparam(self):
        return self._bparam

    @bparam.setter
    def bparam(self, bparam):
        self._bparam = bparam

    def _assign_states(self):
        self._state = self._concat_states[0]
        self._bparam = self._concat_states[1]

    def prediction_step(self):
        pass
