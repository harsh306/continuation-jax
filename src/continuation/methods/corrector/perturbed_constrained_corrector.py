from src.continuation.methods.corrector.constrained_corrector import (
    ConstrainedCorrector,
)
from typing import Tuple
from jax.tree_util import tree_map, tree_multimap
from jax import random
from utils.math import pytree_sub, pytree_dot
from jax import numpy as np


class PerturbedCorrecter(ConstrainedCorrector):
    def __init__(
        self,
        optimizer,
        objective,
        dual_objective,
        lagrange_multiplier,
        concat_states,
        delta_s,
        ascent_opt,
        key_state,
    ):
        super().__init__(
            optimizer,
            objective,
            dual_objective,
            lagrange_multiplier,
            concat_states,
            delta_s,
            ascent_opt,
        )
        self._parc_vec = None
        self.key = random.PRNGKey(key_state)

    def _perform_perturb(self):
        self._state = tree_map(
            lambda a: a + random.normal(self.key, a.shape), self._state
        )
        self._bparam = tree_map(
            lambda a: a + random.normal(self.key, a.shape), self._bparam
        )
        state_stack = []  # TODO: reove stack list
        state_stack.extend(self._state)
        state_stack.extend(self._bparam)
        self._parc_vec = pytree_sub(state_stack, self._state_secant_c2)

    def _evaluate_perturb(
        self,
    ) -> Tuple:  # TODO: two options - evaluate algebraic way or solve for constraint only
        dot = pytree_dot(self._parc_vec, self._state_secant_vector)
        if not np.isclose(dot, 0.0, rtol=0.15):
            print("Reverting perturb")
            self._state = tree_map(
                lambda a: a - random.normal(self.key, a.shape), self._state
            )
            self._bparam = tree_map(
                lambda a: a - random.normal(self.key, a.shape), self._bparam
            )

    def correction_step(self) -> Tuple:
        self._perform_perturb()
        self._evaluate_perturb()

        for k in range(self.warmup_period):
            grads = self._compute_grads()
            self._state = self.opt.update_params(self._state, grads)

        for k in range(self.ascent_period):
            lagrange_grads = self._compute_max_grads()
            self._lagrange_multiplier = self.opt.update_params(
                self._lagrange_multiplier, lagrange_grads
            )
            for j in range(self.descent_period):
                state_grads, bpram_grads = self._compute_min_grads()
                self._bparam = self.opt.update_params(self._bparam, bpram_grads)
                self._state = self.opt.update_params(self._state, state_grads)

        return self._state, self._bparam
