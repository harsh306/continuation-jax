from cjax.continuation.arc_len_continuation import PseudoArcLenContinuation, Continuation
from cjax.continuation.states.state_variables import StateWriter
from cjax.continuation.methods.predictor.arc_secant_predictor import SecantPredictor
from cjax.continuation.methods.corrector.perturb_parc_evolve import (
    PerturbedFixedCorrecter,
)
import jax.numpy as np
from jax.tree_util import *
import copy
from cjax.utils.profiler import profile
import gc
from jax.experimental.optimizers import l2_norm
from cjax.continuation.states.state_variables import StateVariable, StateWriter
from jax import jit, grad
import numpy.random as npr

# TODO: make **kwargs availible


class PerturbedPseudoArcLenFixedContinuation(Continuation):
    """Noisy Pseudo Arc-length Continuation strategy.

    Composed of secant predictor and noisy constrained corrector"""

    def __init__(
        self,
        state,
        bparam,
        state_0,
        bparam_0,
        counter,
        objective,
        dual_objective,
        hparams,
        key_state,
    ):

        # states
        self._state_wrap = StateVariable(state, counter)
        self._bparam_wrap = StateVariable(
            bparam, counter
        )  # Todo : save tree def, always unlfatten before compute_grads
        self._prev_state = state_0
        self._prev_bparam = bparam_0

        # objectives
        self.objective = objective
        self.dual_objective = dual_objective
        self.value_func = jit(self.objective)

        self.hparams = hparams

        self._value_wrap = StateVariable(0.2, counter)  # TODO: fix with a static batch (test/train)
        self._quality_wrap = StateVariable(l2_norm(self._state_wrap.state)/10, counter)

        # every step hparams
        self.continuation_steps = hparams["continuation_steps"]

        self._delta_s = hparams["delta_s"]
        self._prev_delta_s = hparams["delta_s"]
        self._omega = hparams["omega"]

        # grad functions # should be pure functional
        self.compute_min_grad_fn = jit(grad(self.dual_objective, [0, 1]))
        self.compute_grad_fn = jit(grad(self.objective, [0]))

        # extras
        self.state_tree_def = None
        self.bparam_tree_def = None
        self.output_file = hparams["meta"]["output_dir"]
        self.prev_secant_direction = None
        self.sw = StateWriter(f"{self.output_file}/version_{key_state}.json")
        self.key_state = key_state + npr.randint(100,200)
        self.clip_lambda_max = lambda g: np.where(
            (g > self.hparams["lambda_max"]), self.hparams["lambda_max"], g
        )
        self.clip_lambda_min = lambda g: np.where(
            (g < self.hparams["lambda_min"]), self.hparams["lambda_min"], g
        )

    @profile(sort_by="cumulative", lines_to_print=10, strip_dirs=True)
    def run(self):
        """Runs the continuation strategy.

        A continuation strategy that defines how predictor and corrector components of the algorithm
        interact with the states of the mathematical system.
        """
        for i in range(self.continuation_steps):

            print(self._value_wrap.get_record(), self._bparam_wrap.get_record(), self._delta_s)
            self._state_wrap.counter = i
            self._bparam_wrap.counter = i
            self._value_wrap.counter = i
            self._quality_wrap.counter = i
            self.sw.write(
                [
                    self._state_wrap.get_record(),
                    self._bparam_wrap.get_record(),
                    self._value_wrap.get_record(),
                    self._quality_wrap.get_record()
                ]
            )

            concat_states = [
                (self._prev_state, self._prev_bparam),
                (self._state_wrap.state, self._bparam_wrap.state),
                self.prev_secant_direction,
            ]

            predictor = SecantPredictor(
                concat_states=concat_states,
                delta_s=self._delta_s,
                prev_delta_s=self._prev_delta_s,
                omega=self._omega,
                net_spacing_param=self.hparams["net_spacing_param"],
                net_spacing_bparam=self.hparams["net_spacing_bparam"],
                hparams=self.hparams
            )
            predictor.prediction_step()

            self.prev_secant_direction = predictor.secant_direction

            self.hparams['sphere_radius'] = self.hparams['sphere_radius_m']*self._delta_s #l2_norm(predictor.secant_direction)
            concat_states = [
                predictor.state,
                predictor.bparam,
                predictor.secant_direction,
                {"state": predictor.state, "bparam": predictor.bparam},
            ]
            del predictor
            gc.collect()
            corrector = PerturbedFixedCorrecter(
                objective=self.objective,
                dual_objective=self.dual_objective,
                value_fn=self.value_func,
                concat_states=concat_states,
                key_state=self.key_state,
                compute_min_grad_fn=self.compute_min_grad_fn,
                compute_grad_fn=self.compute_grad_fn,
                hparams=self.hparams,
                pred_state=[self._state_wrap.state, self._bparam_wrap.state],
                pred_prev_state=[self._state_wrap.state, self._bparam_wrap.state],
                counter=self.continuation_steps,
            )
            self._prev_state = copy.deepcopy(self._state_wrap.state)
            self._prev_bparam = copy.deepcopy(self._bparam_wrap.state)

            (
                state,
                bparam,
                quality,
                value,
                corrector_omega
            ) = (
                corrector.correction_step()
            )  # TODO: make predictor corrector similar api's
            # TODO: Enable MLFlow
            bparam = tree_map(self.clip_lambda_max, bparam)
            bparam = tree_map(self.clip_lambda_min, bparam)

            self._state_wrap.state = state
            self._bparam_wrap.state = bparam
            self._value_wrap.state = value
            self._quality_wrap.state = quality
            #self._omega = corrector_omega
            self._prev_delta_s = self._delta_s
            self._delta_s = corrector_omega * self._delta_s
            self._delta_s = min(self._delta_s, 0.0002)
            self._delta_s = max(self._delta_s, 0.000002)
            del corrector
            del concat_states
            gc.collect()
            if (bparam[0]>=self.hparams['lambda_max']) or (bparam[0]<=self.hparams['lambda_min']):
                self.sw.write(
                    [
                        self._state_wrap.get_record(),
                        self._bparam_wrap.get_record(),
                        self._value_wrap.get_record(),
                        self._quality_wrap.get_record()
                    ]
                )
                break

