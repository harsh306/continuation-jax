from cjax.continuation.methods.corrector.constrained_corrector import (
    ConstrainedCorrector,
    Corrector,
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
from cjax.utils.evolve_utils import *
import numpy as onp
from examples.torch_data import get_data
from cjax.utils.datasets import get_mnist_data, meta_mnist, mnist


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
        delta_s,
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
        self.delta_s = delta_s
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
        if hparams["meta"]["dataset"] == "mnist":
            self.data_loader = iter(
                get_mnist_data(
                    batch_size=hparams["batch_size"],
                    resize=hparams["resize_to_small"],
                    filter=hparams["filter"]
                )
            )
            self.num_batches = meta_mnist(hparams["batch_size"], hparams["filter"])["num_batches"]
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
        batch_data,
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
        u = tree_map(
            lambda a: a + random.normal(key, a.shape),
            pytree_ones_like(n),
        )
        tmp, _ = pytree_to_vec([_state_secant_c2["state"], _state_secant_c2["bparam"]])

        # select a point on the secant normal
        u_0, _ = pytree_to_vec(pred_prev_state)
        # compute projection
        proj_of_u_on_n = projection_affine(len(n), u, n, u_0)

        point_on_plane = u + pytree_sub(tmp, proj_of_u_on_n)  ## state= pred_state + n
        #noise = random.uniform(key, [1], minval=-0.003, maxval=0.03)
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

    def _evaluate_perturb(self):
        """Evaluate weather the perturbed vector is orthogonal to secant vector"""

        dot = pytree_dot(
            pytree_normalized(self._parc_vec),
            pytree_normalized(self._state_secant_vector),
        )
        if math.isclose(dot, 0.0, abs_tol=0.15):
            print(f"Perturb was near arc-plane. {dot}")
        else:
            print(f"Perturb was not on arc-plane.{dot}")

    def correction_step(self) -> Tuple:
        """Given the current state optimize to the correct state.

        Returns:
          (state: problem parameters, bparam: continuation parameter) Tuple
        """

        quality = 1.0
        if self.hparams["meta"]["dataset"] == "mnist":  # TODO: make it generic
            batch_data = next(self.data_loader)
        else:
            batch_data = None

        ants_norm_grads = [5.0 for _ in range(self.hparams["n_wall_ants"])]
        ants_loss_values = [5.0 for _ in range(self.hparams["n_wall_ants"])]
        ants_state = [self._state for _ in range(self.hparams["n_wall_ants"])]
        ants_bparam = [self._bparam for _ in range(self.hparams["n_wall_ants"])]
        for i_n in range(self.hparams["n_wall_ants"]):
            corrector_omega = 1.0
            stop = False
            _, key = random.split(
                random.PRNGKey(self.key_state + i_n + npr.randint(1, (i_n + 1) * 10))
            )
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
                batch_data,
            )
            if self.hparams["_evaluate_perturb"]:
                self._evaluate_perturb()  # does every time

            ants_state[i_n] = self.state_stack["state"]
            ants_bparam[i_n] = self.state_stack["bparam"]
            D_values = []
            print(f"num_batches", self.num_batches)
            for j_epoch in range(self.descent_period):
                for b_j in range(self.num_batches):

                    #alternate
                    # grads = self.compute_grad_fn(self._state, self._bparam, batch_data)
                    # self._state = self.opt.update_params(self._state, grads[0])
                    state_grads, bparam_grads = self.compute_min_grad_fn(
                        ants_state[i_n],
                        ants_bparam[i_n],
                        self._lagrange_multiplier,
                        self._state_secant_c2,
                        self._state_secant_vector,
                        batch_data,
                        self.delta_s,
                    )

                    if self.hparams["adaptive"]:
                        self.opt.lr = self.exp_decay(
                            j_epoch, self.hparams["natural_lr"]
                        )
                        quality = l2_norm(state_grads) #l2_norm(bparam_grads)
                        if self.hparams["local_test_measure"] == "norm_gradients":
                            if quality > self.hparams["quality_thresh"]:
                                pass
                                print(
                                    f"quality {quality}, {self.opt.lr}, {bparam_grads} ,{j_epoch}"
                                )
                            else:
                                stop = True
                                print(
                                    f"quality {quality} stopping at , {j_epoch}th step"
                                )
                        else:
                            print(
                                f"quality {quality}, {bparam_grads} ,{j_epoch}"
                            )
                            if len(D_values) >= 20:
                                tmp_means = running_mean(D_values, 10)
                                if (math.isclose(
                                tmp_means[-1],
                                tmp_means[-2],
                                abs_tol=self.hparams["loss_tol"]
                                )):
                                    print(
                                        f"stopping at , {j_epoch}th step, {ants_bparam[i_n]} bparam"
                                    )
                                    stop = True

                        state_grads = clip_grads(
                            state_grads, self.hparams["max_clip_grad"]
                        )
                        bparam_grads = clip_grads(
                            bparam_grads, self.hparams["max_clip_grad"]
                        )

                    if self.hparams["guess_ant_steps"] >= (
                        j_epoch + 1
                    ):  # To get around folds slowly
                        corrector_omega = min(
                            self.hparams["guess_ant_steps"] / (j_epoch + 1), 1.5
                        )
                    else:
                        corrector_omega = max(
                            self.hparams["guess_ant_steps"] / (j_epoch + 1), 0.05
                        )

                    ants_state[i_n] = self.opt.update_params(
                        ants_state[i_n], state_grads, j_epoch
                    )
                    ants_bparam[i_n] = self.opt.update_params(
                        ants_bparam[i_n], bparam_grads, j_epoch
                    )
                    ants_loss_values[i_n] = self.value_fn(
                        ants_state[i_n], ants_bparam[i_n], batch_data
                    )
                    D_values.append(ants_loss_values[i_n])
                    ants_norm_grads[i_n] = quality
                    # if stop:
                    #     break
                    if (
                        self.hparams["meta"]["dataset"] == "mnist"
                    ):  # TODO: make it generic
                        batch_data = next(self.data_loader)
                if stop:
                    break

        # ants_group = dict(enumerate(grouper(ants_state, tolerence), 1))
        # print(f"Number of groups: {len(ants_group)}")
        cheapest_index = get_cheapest_ant(ants_norm_grads, ants_loss_values,
                                          local_test=self.hparams["local_test_measure"])
        self._state = ants_state[cheapest_index]
        self._bparam = ants_bparam[cheapest_index]
        value = self.value_fn(
            self._state, self._bparam, batch_data
        )  # Todo: why only final batch data

        _, _, test_images, test_labels = mnist(permute_train=False, resize=True, filter=self.hparams["filter"])
        del _
        val_loss = self.value_fn(self._state, self._bparam, (test_images,test_labels))
        print(f"val loss: {val_loss}")

        return self._state, self._bparam, quality, value, val_loss, corrector_omega
