from cjax.continuation.methods.predictor.base_predictor import Predictor
from cjax.utils.math_trees import pytree_element_add


class NaturalPredictor(Predictor):
    """Natural Predictor only updates continuation parameter"""

    def __init__(self, concat_states, delta_s):
        super().__init__(concat_states)
        self.delta_s = delta_s

    def _assign_states(self) -> None:
        super()._assign_states()

    def prediction_step(self):
        """Given current state predict next state.
        Updates (state: problem parameters, bparam: continuation parameter) Tuple
        """
        self._assign_states()
        self._bparam = pytree_element_add(self._bparam, self.delta_s)
