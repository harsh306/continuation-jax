from cjax.continuation.methods.corrector.constrained_corrector import (
    ConstrainedCorrector, Corrector
)
from typing import Tuple
from jax.tree_util import tree_map
from jax import random
from cjax.utils.rotation_ndims import *
from jax import jit
import math
from cjax.optimizer.optimizer import OptimizerCreator
from jax.experimental.optimizers import clip_grads
import numpy.random as npr
from examples.torch_data import get_data
from cjax.utils.datasets import get_mnist_data, meta_mnist #faster version

class PerturbedFixedCorrecter(Corrector):
    """Minimize the objective using gradient based method along with some constraint and noise"""

    def __init__(
        self,
        objective,
        dual_objective,
        value_fn,
        concat_states,
        key_state,
        compute_min_grad_fn,
        compute_grad_fn,
        hparams,
        pred_state,
        pred_prev_state,
        counter,
    ):
        self.concat_states = concat_states
        self._state = None
        self._bparam = None
        self.opt = OptimizerCreator(
            opt_string=hparams["meta"]["optimizer"], learning_rate=hparams["descent_lr"]
        ).get_optimizer()
        self.objective = objective
        self.dual_objective = dual_objective
        self._lagrange_multiplier = hparams["lagrange_init"]
        self._state_secant_vector = None
        self._state_secant_c2 = None
        self.delta_s = hparams["delta_s"]
        self.descent_period = hparams["descent_period"]
        self.max_norm_state = hparams["max_bounds"]
        self.hparams = hparams
        self.compute_min_grad_fn = compute_min_grad_fn
        self.compute_grad_fn = compute_grad_fn
        self._assign_states()
        self._parc_vec = None
        self.state_stack = dict()
        self.key_state = key_state
        self.pred_state = pred_state
        self.pred_prev_state = pred_prev_state
        self.sphere_radius = hparams["sphere_radius"]
        self.counter = counter
        self.value_fn = value_fn
        # self.data_loader = iter(get_data(dataset=hparams["meta"]['dataset'],
        #                             batch_size=hparams['batch_size'],
        #                             num_workers=hparams['data_workers'],
        #                             train_only=True, test_only=False))
        if hparams["meta"]['dataset'] =='mnist':
            self.data_loader = iter(get_mnist_data(batch_size=hparams['batch_size'],
                                                   resize=hparams['resize_to_small']))
            self.num_batches = meta_mnist(hparams['batch_size'])['num_batches']
        else:
            self.num_batches = 1


    def _assign_states(self):
        self._state = self.concat_states[0]
        self._bparam = self.concat_states[1]
        self._state_secant_vector = self.concat_states[2]
        self._state_secant_c2 = self.concat_states[3]

    @staticmethod
    @jit
    def exp_decay(epoch, initial_lrate):
        k = 0.02
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
        batch_data
    ):
        ### Secant normal
        n, sample_unravel = pytree_to_vec(
            [_state_secant_vector["state"], _state_secant_vector["bparam"]]
        )
        n = pytree_normalized(n)
        ### sample a random poin in Rn
        # u = tree_map(
        #     lambda a: a + random.uniform(key, a.shape),
        #     pytree_zeros_like(n),
        # )
        #print(key)
        u = tree_map(
            lambda a: a + random.normal(key, a.shape),
            pytree_ones_like(n),
        )
        tmp, _ = pytree_to_vec([_state_secant_c2['state'], _state_secant_c2['bparam']])

        # select a point on the secant normal
        u_0, _ = pytree_to_vec(pred_prev_state)
        # compute projection
        proj_of_u_on_n = projection_affine(len(n), u, n, u_0)

        point_on_plane = u + pytree_sub(tmp, proj_of_u_on_n)  ## state= pred_state + n
        noise = random.uniform(key, [1], minval=-0.03, maxval=0.03)
        inv_vec = np.array([-1.0 + noise, 1.0 + noise])
        parc = pytree_element_mul(
            pytree_normalized(pytree_sub(point_on_plane, tmp)),
            inv_vec[(counter % 2)],
        )
        point_on_plane_2 = tmp + sphere_radius* parc
        #print('point on plane ',point_on_plane_2, inv_vec)
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

        quality = 1.0
        if self.hparams["meta"]['dataset'] == 'mnist':  # TODO: make it generic
            batch_data = next(self.data_loader)
        else:
            batch_data = None
        N_opt = 10
        stop = False
        corrector_omega = 1.0
        N_fits = 3
        # bparam_grads = pytree_zeros_like(self._bparam)
        #print('the radius', self.sphere_radius)
        N_quality = [5.0 for _ in range(N_fits)]
        N_states = [self._state for _ in range(N_fits)]
        N_bparams = [self._bparam for _ in range(N_fits)]
        for i_n in range(N_fits):
            _, key = random.split(random.PRNGKey(self.key_state+i_n+ npr.randint(1, (i_n+1)*10)))
            del _
            self._parc_vec, self.state_stack = self._perform_perturb_by_projection(
                self._state_secant_vector,
                self._state_secant_c2,
                key,
                self.pred_prev_state,
                self._state,
                self._bparam,
                i_n,
                self.sphere_radius,
                batch_data
            )
            # if self.hparams['_evaluate_perturb']:
            #     self._evaluate_perturb() # does every time
            #
            N_states[i_n] = self.state_stack['state']
            N_bparams[i_n] = self.state_stack['bparam']

            for j in range(self.descent_period):
                for b_j in range(self.num_batches):

                    # grads = self.compute_grad_fn(self._state, self._bparam, batch_data)
                    # self._state = self.opt.update_params(self._state, grads[0])
                    state_grads, bparam_grads = self.compute_min_grad_fn(
                        N_states[i_n],
                        N_bparams[i_n],
                        self._lagrange_multiplier,
                        self._state_secant_c2,
                        self._state_secant_vector,
                        batch_data,
                        self.delta_s,
                    )

                    if self.hparams['adaptive']:
                        self.opt.lr = self.exp_decay(j, self.hparams['natural_lr'])
                        quality = l2_norm(state_grads) #+l2_norm(bparam_grads)
                        if quality>self.hparams['quality_thresh']:
                            pass
                            #print(f"quality {quality}, {self.opt.lr}, {bparam_grads} ,{j}")
                        else:
                            if N_opt>(j+1):  # To get around folds slowly
                                corrector_omega = min(N_opt/(j+1), 2.0)
                            else:
                                corrector_omega = max(N_opt/(j+1), 0.5)
                            stop = True
                            print(f"quality {quality} stopping at , {j}th step")
                        state_grads = clip_grads(state_grads, self.hparams['max_clip_grad'])
                        bparam_grads = clip_grads(bparam_grads, self.hparams['max_clip_grad'])

                    N_bparams[i_n] = self.opt.update_params(N_bparams[i_n], bparam_grads, j)
                    N_states[i_n] = self.opt.update_params(N_states[i_n], state_grads, j)
                    N_quality[i_n] = quality
                    if stop:
                        break
                    if self.hparams["meta"]['dataset'] == 'mnist': # TODO: make it generic
                        batch_data = next(self.data_loader)
                if stop:
                    break

        m = min(N_quality)
        indicies = [i for i, j in enumerate(N_quality) if j == m]
        index = npr.randint(0,len(indicies))
        cheapest = indicies[index]
        #N_quality.index((min(N_quality)))
        self._state = N_states[cheapest]
        self._bparam = N_bparams[cheapest]
        value = self.value_fn(self._state, self._bparam, batch_data) # Todo: why only final batch data
        return self._state, self._bparam, quality, value, corrector_omega