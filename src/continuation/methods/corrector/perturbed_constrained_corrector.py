from src.continuation.methods.corrector.constrained_corrector import (
    ConstrainedCorrector,
)
from typing import Tuple
from jax.tree_util import tree_map, tree_flatten
from jax import random
from utils.math_trees import *
from jax import numpy as np
from utils.rotation_ndims import get_rotation_pytree
from jax import jit
import math

class PerturbedCorrecter(ConstrainedCorrector):
    """Minimize the objective using gradient based method along with some constraint and noise"""

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
        compute_min_grad_fn,
        compute_max_grad_fn,
        compute_grad_fn
    ):
        super().__init__(
            optimizer,
            objective,
            dual_objective,
            lagrange_multiplier,
            concat_states,
            delta_s,
            ascent_opt,
            compute_min_grad_fn,
            compute_max_grad_fn,
            compute_grad_fn
        )
        self._parc_vec = None
        self.state_stack= dict()
        self.key = random.PRNGKey(key_state)


    def _perform_perturb(self):
        """Add noise to a PyTree"""

        destination, _ = tree_flatten(self._state_secant_vector)
        destination, _ = pytree_to_vec(destination)

        src = np.hstack((np.zeros(len(destination)-1), 1.0))
        src, _ = pytree_to_vec(src)
        assert src.shape == destination.shape
        rotation_matrix = get_rotation_pytree(src, destination) # TODO: Refactor

        sample = tree_map(
            lambda a: a + random.normal(self.key, a.shape),
            self._state,
        )
        z = pytree_zeros_like(self._bparam)
        sample_vec, sample_unravel = pytree_to_vec([sample, z])

        #transform sample to arc-plane
        new_sample = np.dot(rotation_matrix, sample_vec) + 0.2*pytree_normalized(destination)

        # sample_vec to pytree
        new_sample = sample_unravel(new_sample)

        # self._state= new_sample[0]
        # self._bparam = new_sample[1]
        self.state_stack.update({"state": new_sample[0]})
        self.state_stack.update({"bparam": new_sample[1]})
        self._parc_vec = pytree_sub(self.state_stack, self._state_secant_c2)

    def _evaluate_perturb(self):
        """Evaluate weather the perturbed vector is orthogonal to secant vector"""
        dot = pytree_dot(self._parc_vec, self._state_secant_vector)
        if math.isclose(dot, 0.0,  abs_tol=0.25):
            print(f"Perturb was near arc-plane. {dot}")
            self._state = self.state_stack['state']
            self._bparam = self.state_stack['bparam']
        else:
            print(f"Perturb was not on arc-plane.{dot}")

    def correction_step(self) -> Tuple:
        """Given the current state optimize to the correct state.

        Returns:
          (state: problem parameters, bparam: continuation parameter) Tuple
        """
        self._perform_perturb()
        self._evaluate_perturb()
        # super().correction_step()

        for k in range(self.warmup_period):
            grads = self.compute_grad_fn(self._state, self._bparam)
            self._state = self.opt.update_params(self._state, grads[0])

        for k in range(self.ascent_period):
            lagrange_grads = self.compute_max_grad_fn(
                self._state,
                self._bparam,
                self._lagrange_multiplier,
                self._state_secant_c2,
                self._state_secant_vector,
                self.delta_s,
            )
            self._lagrange_multiplier = self.ascent_opt.update_params(
                self._lagrange_multiplier, lagrange_grads[0]
            )
            for j in range(self.descent_period):
                state_grads, bpram_grads = self.compute_min_grad_fn(
                    self._state,
                    self._bparam,
                    self._lagrange_multiplier,
                    self._state_secant_c2,
                    self._state_secant_vector,
                    self.delta_s,
                )
                self._bparam = self.opt.update_params(self._bparam, bpram_grads)
                self._state = self.opt.update_params(self._state, state_grads)

        return self._state, self._bparam
