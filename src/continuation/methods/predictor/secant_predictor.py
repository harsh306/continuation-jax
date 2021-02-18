from src.continuation.methods.predictor.base_predictor import Predictor
from jax.tree_util import tree_multimap
from utils.math_trees import *


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
        self.prev_secant_direction = None

    def _assign_states(self):
        self.prev_secant_direction = self._concat_states[2]
        self._state, self._bparam = self._concat_states[1]
        self._prev_state, self._prev_bparam = self._concat_states[0]

    def _choose_direction(self):
        if self.prev_secant_direction:
            inner_prod = pytree_dot(
                pytree_normalized(self.prev_secant_direction),
                pytree_normalized(self.secant_direction),
            )
            state_neg_inner_prod = pytree_dot(
                [
                    self.prev_secant_direction["state"],
                    self.prev_secant_direction["bparam"],
                ],
                [
                    pytree_element_mul(self.secant_direction["state"], -1.0),
                    self.secant_direction["bparam"],
                ],
            )
            bparam_neg_inner_prod = pytree_dot(
                [
                    self.prev_secant_direction["state"],
                    self.prev_secant_direction["bparam"],
                ],
                [
                    self.secant_direction["state"],
                    pytree_element_mul(self.secant_direction["bparam"], -1.0),
                ],
            )
            neg_inner_prod = pytree_dot(
                self.prev_secant_direction,
                pytree_element_mul(self.secant_direction, -1.0),
            )
            prods = [
                inner_prod,
                state_neg_inner_prod,
                bparam_neg_inner_prod,
                neg_inner_prod,
            ]

            # prod_diffs = list(map(lambda x: abs(x - 1.0), prods))
            # print(prod_diffs)
            min_index = prods.index(max(prods))

            if min_index == 1:
                self.secant_direction.update(
                    {"state": pytree_element_mul(self.secant_direction["state"], -1.0)}
                )
            elif min_index == 2:
                self.secant_direction.update(
                    {
                        "bparam": pytree_element_mul(
                            self.secant_direction["bparam"], -1.0
                        )
                    }
                )
            elif min_index == 3:
                self.secant_direction = pytree_element_mul(self.secant_direction, -1.0)
            else:
                pass

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
        self._choose_direction()
        self._state = tree_multimap(
            lambda a, b: a + self.omega * b, self._state, self.secant_direction["state"]
        )
        self._bparam = tree_multimap(
            lambda a, b: a + self.omega * b,
            self._bparam,
            self.secant_direction["bparam"],
        )
        print(self._state, self.bparam)

    def get_secant_vector_concat(self):
        """Concatenated secant vector.

        Returns:
          [state_vector: problem parameters, bparam_vector: continuation parameter] list
        """
        return self.secant_direction

    # TODO: below function needed?
    def get_secant_concat(self):
        """Concatenated secant guess/point.

        Returns:
          [state_guess: problem parameters, bparam_guess: continuation parameter] list
        """
        concat = dict()
        concat.update({"state": self._state})
        concat.update({"bparam": self._bparam})
        return concat
