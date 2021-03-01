from cjax.continuation.methods.corrector.constrained_corrector import (
    ConstrainedCorrector,
)
from typing import Tuple
from jax.tree_util import tree_map
from jax import random
from cjax.utils.rotation_ndims import *
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
        self.key_state = key_state
        self.pred_state = pred_state
        self.pred_prev_state = pred_prev_state
        self.sphere_radius = hparams["sphere_radius"]
        self.counter = counter

    @staticmethod
    @jit
    def exp_decay(epoch, initial_lrate):
        k = 0.1
        lrate = initial_lrate * np.exp(-k * epoch)
        return lrate

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
        n = pytree_normalized(n)
        ### sample a random poin in Rn
        u = tree_map(
            lambda a: a + random.uniform(key, a.shape),
            pytree_zeros_like(n),
        )

        tmp, _ = pytree_to_vec([_state_secant_c2['state'], _state_secant_c2['bparam']])

        # select a point on the secant normal
        u_0, _ = pytree_to_vec(pred_prev_state)
        # compute projection
        proj_of_u_on_n = projection_affine(len(n), u, n, u_0)

        point_on_plane = u + pytree_sub(tmp, proj_of_u_on_n)  ## state= pred_state + n
        inv_vec = np.array([-1.0, 1.0])
        parc = pytree_element_mul(
            pytree_normalized(pytree_sub(point_on_plane, tmp)),
            inv_vec[(counter % 2)],
        )
        point_on_plane_2 = tmp + sphere_radius* parc
        new_sample = sample_unravel(point_on_plane_2)
        state_stack = {}
        state_stack.update({"state": new_sample[0]})
        state_stack.update({"bparam": new_sample[1]})
        _parc_vec = pytree_sub(state_stack, _state_secant_c2)
        return _parc_vec, state_stack

    def _evaluate_perturb(self):
        """Evaluate weather the perturbed vector is orthogonal to secant vector"""

        dot = pytree_dot(pytree_normalized(self._parc_vec), pytree_normalized(self._state_secant_vector))
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
        _, key = random.split(random.PRNGKey(self.key_state+npr.randint(1,100)))
        del _
        self._parc_vec, self.state_stack = self._perform_perturb_by_projection(
            self._state_secant_vector,
            self._state_secant_c2,
            key,
            self.pred_prev_state,
            self._state,
            self._bparam,
            self.counter,
            self.sphere_radius,
        )
        self._evaluate_perturb() # does every time
        print('corrector_perturb', self.state_stack["bparam"])
        quality = 1.0
        for j in range(self.descent_period):
            state_grads, bparam_grads = self.compute_min_grad_fn(
                self._state,
                self._bparam,
                self._lagrange_multiplier,
                self._state_secant_c2,
                self._state_secant_vector,
                self.delta_s,
            )
            #print('gradsss',bparam_grads)

            if self.hparams['adaptive']:
                self.opt.lr = self.exp_decay(j, self.hparams['natural_lr'])
                quality = l2_norm(state_grads)+l2_norm(bparam_grads)
                if quality>self.hparams['quality_thresh']:
                    pass
                    #self.hparams['natural_lr'] = int(self.hparams['natural_lr'])/8
                    #print(f"quality {quality}, {self.opt.lr}")
                    #print('grads', bparam_grads, state_grads)
                state_grads = clip_grads(state_grads, self.hparams['quality_thresh'])
                bparam_grads = clip_grads(bparam_grads, self.hparams['quality_thresh'])

            self._bparam = self.opt.update_params(self._bparam, bparam_grads, j)
            self._state = self.opt.update_params(self._state, state_grads, j)
        print('coreector 1', self._bparam)

        return self._state, self._bparam, quality
