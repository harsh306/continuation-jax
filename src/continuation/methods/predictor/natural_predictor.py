from typing import Tuple

from src.continuation.methods.predictor.base_predictor import Predictor
from jax.tree_util import tree_multimap, tree_map


class NaturalPredictor(Predictor):
    def __init__(self, concat_states, delta_s):
        super().__init__(concat_states)
        self.delta_s = delta_s

    def _assign_states(self) -> None:
        super()._assign_states()

    def prediction_step(self) -> Tuple:
        self._assign_states()
        self._bparam = tree_map(lambda a: a+self.delta_s, self._bparam)
        return self._state, self._bparam
