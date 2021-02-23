from src.continuation.methods.corrector.constrained_corrector import (
    ConstrainedCorrector,
)
from typing import Tuple
from jax.tree_util import tree_map, tree_flatten
from jax import random
from utils.math_trees import *
from jax import numpy as np
from utils.rotation_ndims import *
from jax import jit
import math
from jax.experimental.optimizers import clip_grads
import numpy.random as npr


class PerturbedFixedCorrecter(ConstrainedCorrector):
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
        compute_grad_fn,
        hparams,
        pred_state,
        pred_prev_state,
        counter,
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
            compute_grad_fn,
            hparams,
        )
        self._parc_vec = None
        self.state_stack = dict()
        _, self.key = random.split(random.PRNGKey(key_state))
        self.pred_state = pred_state
        self.pred_prev_state = pred_prev_state
        self.sphere_radius = hparams["sphere_radius"]
        self.counter = counter

    @staticmethod
    @jit
    def _perform_perturb_by_projection(
        _state_secant_vector,
        _state_secant_c2,
        key,
        pred_prev_state,
        _state,
        _bparam,
        counter,
        sphere_radius,
    ):
        ### Secant normal
        n, sample_unravel = pytree_to_vec(
            [_state_secant_vector["state"], _state_secant_vector["bparam"]]
        )
        u = tree_map(
            lambda a: a + random.uniform(key, a.shape),
            pytree_zeros_like(n),
        )
        ### sample a random poin in Rn
        # sample = tree_map(
        #     lambda a: a + random.uniform(key, a.shape),
        #     pytree_zeros_like(_state),
        # )
        # z = tree_map(
        #     lambda a: a + random.normal(key, a.shape),
        #     pytree_zeros_like(_bparam),
        # )
        #u, sample_unravel = pytree_to_vec([sample, z])
        # select a point on the secant normal
        u_0, _ = pytree_to_vec(pred_prev_state)
        # compute projection
        proj_of_u_on_n = projection_affine(len(n), u, n, u_0)
        tmp, _ = pytree_to_vec([_state, _bparam])
        point_on_plane = u + pytree_sub(tmp, proj_of_u_on_n)  ## state= pred_state + n
        inv_vec = np.array([-1.0, 1.0])
        parc = pytree_element_mul(
            pytree_normalized(pytree_sub(point_on_plane, tmp)),
            inv_vec[(counter % 2)],
        )
        point_on_plane_2 = tmp + sphere_radius * parc
        new_sample = sample_unravel(point_on_plane_2)
        state_stack = {}
        state_stack.update({"state": new_sample[0]})
        state_stack.update({"bparam": new_sample[1]})
        _parc_vec = pytree_sub(state_stack, _state_secant_c2)
        return _parc_vec, state_stack

    def _perform_perturb(self):
        """Add noise to a PyTree"""
        destination, _ = pytree_to_vec(
            [self._state_secant_vector["state"], self._state_secant_vector["bparam"]]
        )

        src = np.hstack((np.zeros(len(destination) - 1), 1.0))
        src, _ = pytree_to_vec(src)
        assert src.shape == destination.shape
        rotation_matrix = get_rotation_pytree(src, destination)  # TODO: Refactor

        sample = tree_map(
            lambda a: a + random.normal(self.key, a.shape),
            self._state,
        )
        z = pytree_zeros_like(self._bparam)
        sample_vec, sample_unravel = pytree_to_vec([sample, z])

        # transform sample to arc-plane
        new_sample = np.dot(rotation_matrix, sample_vec) + destination

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
        if math.isclose(dot, 0.0, abs_tol=0.25):
            print(f"Perturb was near arc-plane. {dot}")
            self._state = self.state_stack["state"]
            self._bparam = self.state_stack["bparam"]
        else:
            print(f"Perturb was not on arc-plane.{dot}")

    def correction_step(self) -> Tuple:
        """Given the current state optimize to the correct state.

        Returns:
          (state: problem parameters, bparam: continuation parameter) Tuple
        """
        self._parc_vec, self.state_stack = self._perform_perturb_by_projection(
            self._state_secant_vector,
            self._state_secant_c2,
            self.key,
            self.pred_prev_state,
            self._state,
            self._bparam,
            self.counter,
            self.sphere_radius,
        )
        # self._evaluate_perturb() # does every time

        for j in range(self.descent_period):
            state_grads, bparam_grads = self.compute_min_grad_fn(
                self._state,
                self._bparam,
                self._lagrange_multiplier,
                self._state_secant_c2,
                self._state_secant_vector,
                self.delta_s,
            )

            state_grads = clip_grads(state_grads, self.max_norm_state)
            bparam_grads = clip_grads(bparam_grads, 0.5 * self.max_norm_state)

            self._bparam = self.opt.update_params(self._bparam, bparam_grads)
            self._state = self.opt.update_params(self._state, state_grads)

        return self._state, self._bparam
