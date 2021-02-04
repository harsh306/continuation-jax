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

    def _compute_secant(self):
        """
        Operation w1 - w1'/ delta_s
        :param params_list: list of only one param ex- w1
        :param delta_s: normalization
        :return: secant vector of param
        """
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

        # norm = 1e-5
        # secantvar_state = []
        # for (i, j) in zip(self._state, self._prev_state):
        #     vec = i - j
        #     secantvar_state.append(vec)
        #     norm += np.linalg.norm(vec)
        # secantvar_bparam = []
        # for (i, j) in zip(self._bparam, self._prev_bparam):
        #     vec = i - j
        #     secantvar_bparam.append(vec)
        #     norm += np.linalg.norm(vec)
        #
        # self.secantvar_state = [z/norm for z in secantvar_state]
        # self.secantvar_bparam = [z/norm for z in secantvar_bparam]

    # def _update_prev_states(self):
    #     self._prev_state, self._prev_bparam = self._concat_states[1]

    def prediction_step(self) -> Tuple:
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
        concat = []
        concat.extend(self.secantvar_state)
        concat.extend(self.secantvar_bparam)
        return concat

    # TODO: norm of state
    def get_secant_concat(self):
        concat = []
        concat.extend(self._state)
        concat.extend(self._bparam)
        return concat
