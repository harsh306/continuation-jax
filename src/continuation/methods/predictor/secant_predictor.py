from typing import Tuple
from src.continuation.methods.predictor.base_predictor import Predictor
from jax import numpy as np
from jax.tree_util import tree_multimap, tree_map
from utils.math import pytree_sub
from jax.experimental.optimizers import l2_norm


class SecantPredictor(Predictor):
    def __init__(self, concat_states, delta_s, omega):
        super().__init__(concat_states)
        self._prev_state = None
        self._prev_bparam = None
        self.delta_s = delta_s
        self.omega = omega
        self.secantvar_state = None
        self.secantvar_bparam = None

    def assign_states(self):
        self._state, self._bparam = self._concat_states[1]
        self._prev_state, self._prev_bparam = self._concat_states[0]

    def _compute_secant(self):  # TODO: make only one vector
        """Secant computation for PyTree"""
        self.secantvar_state = pytree_sub(self._state, self._prev_state)
        self.secantvar_bparam = pytree_sub(self._bparam, self._prev_bparam)
        states_norm = (
            l2_norm(self.secantvar_state) + l2_norm(self.secantvar_bparam) + 1e-4
        )
        self.secantvar_state = tree_map(lambda a: a / states_norm, self.secantvar_state)
        self.secantvar_bparam = tree_map(
            lambda a: a / states_norm, self.secantvar_bparam
        )
        del states_norm

    def prediction_step(self) -> Tuple:
        """Given current state predict next state.

        Returns:
          (state_guess: problem parameters, bparam_guess: continuation parameter) Tuple
        """
        self.assign_states()
        self._compute_secant()
        self._state = tree_multimap(
            lambda a, b: a + self.omega * b, self._state, self.secantvar_state
        )
        self._bparam = tree_multimap(
            lambda a, b: a + self.omega * b, self._bparam, self.secantvar_bparam
        )

        # self._state = [z + self.omega * k for (z, k) in zip(self._state, self.secantvar_state)]
        # self._bparam = [z + self.omega * k for (z, k) in zip(self._bparam, self.secantvar_bparam)]
        return self._state, self._bparam

    def get_secant_vector_concat(self):
        """Concatenated secant vector.

        Returns:
          [state_vector: problem parameters, bparam_vector: continuation parameter] list
        """
        concat = []
        concat.extend(self.secantvar_state)
        concat.extend(self.secantvar_bparam)
        return concat

    # TODO: norm of state
    def get_secant_concat(self):
        """Concatenated secant guess/point.

        Returns:
          [state_guess: problem parameters, bparam_guess: continuation parameter] list
        """
        concat = []
        concat.extend(self._state)
        concat.extend(self._bparam)
        return concat
