from cjax.continuation.base_continuation import Continuation
from cjax.continuation.states.state_variables import StateVariable, StateWriter
from cjax.optimizer.optimizer import OptimizerCreator
from cjax.continuation.methods.predictor.secant_predictor import SecantPredictor
from cjax.continuation.methods.corrector.unconstrained_corrector import (
    UnconstrainedCorrector,
)
from jax.tree_util import *
import gc
from cjax.utils.profiler import profile
from jax import jit, grad
import jax.numpy as np


class SecantContinuation(Continuation):
    """Secant Continuation strategy.

    Composed of natural predictor and unconstrained corrector"""

    def __init__(self, state, bparam, state_0, bparam_0, counter, objective, hparams):
        self._state_wrap = StateVariable(state, counter)
        self._bparam_wrap = StateVariable(bparam, counter)
        self._prev_state = state_0
        self._prev_bparam = bparam_0
        self.objective = objective
        self.value_func = jit(self.objective)
        self._value_wrap = StateVariable(self.objective(state, bparam), counter)
        self.sw = None
        self.hparams = hparams
        self.opt = OptimizerCreator(
            opt_string=hparams["meta"]["optimizer"], learning_rate=hparams["natural_lr"]
        ).get_optimizer()
        self.continuation_steps = hparams["continuation_steps"]

        self.output_file = hparams["meta"]["output_dir"]
        self._delta_s = hparams["delta_s"]
        self._omega = hparams["omega"]
        self.grad_fn = jit(grad(self.objective, argnums=[0]))
        self.prev_secant_direction = None

    @profile(sort_by="cumulative", lines_to_print=10, strip_dirs=True)
    def run(self):
        """Runs the continuation strategy.

        A continuation strategy that defines how predictor and corrector components of the algorithm
        interact with the states of the mathematical system.
        """
        self.sw = StateWriter(f"{self.output_file}/version.json")

        for i in range(self.continuation_steps):
            print(self._value_wrap.get_record(), self._bparam_wrap.get_record())
            self._state_wrap.counter = i
            self._bparam_wrap.counter = i
            self._value_wrap.counter = i
            self.sw.write(
                [
                    self._state_wrap.get_record(),
                    self._bparam_wrap.get_record(),
                    self._value_wrap.get_record(),
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
                omega=self._omega,
                net_spacing_param=self.hparams["net_spacing_param"],
                net_spacing_bparam=self.hparams["net_spacing_bparam"],
                hparams=self.hparams
            )
            predictor.prediction_step()

            self.prev_secant_direction = predictor.secant_direction

            concat_states = [predictor.state, predictor.bparam]
            del predictor
            gc.collect()
            corrector = UnconstrainedCorrector(
                optimizer=self.opt,
                objective=self.objective,
                concat_states=concat_states,
                grad_fn=self.grad_fn,
                warmup_period=self.hparams["warmup_period"],
            )
            state, bparam = corrector.correction_step()

            clip_lambda = lambda g: np.where(
                (g > self.hparams["lambda_max"]), self.hparams["lambda_max"], g
            )
            bparam = tree_map(clip_lambda, bparam)
            clip_lambda = lambda g: np.where(
                (g < self.hparams["lambda_min"]), self.hparams["lambda_min"], g
            )
            bparam = tree_map(clip_lambda, bparam)

            value = self.value_func(state, bparam)
            self._state_wrap.state = state
            self._bparam_wrap.state = bparam
            self._value_wrap.state = value
            del corrector
            gc.collect()
            if self._bparam_wrap.state[0] >= self.hparams["lambda_max"]:
                self.sw.write(
                    [
                        self._state_wrap.get_record(),
                        self._bparam_wrap.get_record(),
                        self._value_wrap.get_record(),
                    ]
                )
                break
