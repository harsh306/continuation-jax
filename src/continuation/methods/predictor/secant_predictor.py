from src.continuation.methods.predictor.base_predictor import Predictor
from jax.tree_util import tree_multimap
from utils.math import pytree_sub, pytree_normalized


class SecantPredictor(Predictor):
    def __init__(self, concat_states, delta_s, omega):
        super().__init__(concat_states)
        self._prev_state = None
        self._prev_bparam = None
        self.delta_s = delta_s
        self.omega = omega
        self.secantvar_state = None
        self.secantvar_bparam = None
        self.secant_direction = dict()

    def _assign_states(self):
        self._state, self._bparam = self._concat_states[1]
        self._prev_state, self._prev_bparam = self._concat_states[0]

    def _compute_secant(self):
        """Secant computation for PyTree"""
        self.secant_direction.update(
            {"state": pytree_sub(self._state, self._prev_state)}
        )
        self.secant_direction.update(
            {"bparam": pytree_sub(self._bparam, self._prev_bparam)}
        )
        self.secant_direction = pytree_normalized(self.secant_direction)

    def prediction_step(self):
        """Given current state predict next state.
        Updates (state_guess: problem parameters, bparam_guess: continuation parameter)
        """
        self._assign_states()
        self._compute_secant()
        self._state = tree_multimap(
            lambda a, b: a + self.omega * b, self._state, self.secant_direction["state"]
        )
        self._bparam = tree_multimap(
            lambda a, b: a + self.omega * b,
            self._bparam,
            self.secant_direction["bparam"],
        )

    def get_secant_vector_concat(self):
        """Concatenated secant vector.

        Returns:
          [state_vector: problem parameters, bparam_vector: continuation parameter] list
        """
        return self.secant_direction

    # TODO: norm of state
    def get_secant_concat(self):
        """Concatenated secant guess/point.

        Returns:
          [state_guess: problem parameters, bparam_guess: continuation parameter] list
        """
        concat = dict()
        concat.update({"state": self._state})
        concat.update({"bparam": self._bparam})
        return concat
