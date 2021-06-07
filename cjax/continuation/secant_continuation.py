from cjax.continuation.base_continuation import Continuation
from cjax.continuation.states.state_variables import StateVariable, StateWriter
from cjax.optimizer.optimizer import OptimizerCreator
from cjax.continuation.methods.predictor.arc_secant_predictor import SecantPredictor
from cjax.utils.data_img_gamma import mnist_gamma
from cjax.utils.datasets import mnist, get_mnist_data, meta_mnist
from cjax.continuation.methods.corrector.unconstrained_corrector import (
    UnconstrainedCorrector,
)
from jax.tree_util import *
import gc
from cjax.utils.profiler import profile
from jax import jit, grad
from cjax.utils.math_trees import *
import jax.numpy as np
import mlflow


class SecantContinuation(Continuation):
    """Secant Continuation strategy.

    Composed of natural predictor and unconstrained corrector"""

    def __init__(self, state, bparam, state_0, bparam_0, counter, objective, accuracy_fn, hparams):
        self._state_wrap = StateVariable(state, counter)
        self._bparam_wrap = StateVariable(bparam, counter)
        self._prev_state = state_0
        self._prev_bparam = bparam_0
        self.objective = objective
        self.accuracy_fn = accuracy_fn
        self.value_func = jit(self.objective)
        self._value_wrap = StateVariable(0.005, counter)
        self._quality_wrap = StateVariable(0.005, counter)
        self.sw = None
        self.hparams = hparams
        self.opt = OptimizerCreator(
            opt_string=hparams["meta"]["optimizer"], learning_rate=hparams["natural_lr"]
        ).get_optimizer()
        if hparams["meta"]["dataset"] == "mnist":
            if hparams["continuation_config"] == 'data':
                self.dataset_tuple = mnist_gamma(
                    resize=hparams["resize_to_small"],
                    filter=hparams["filter"])
            else:
                self.dataset_tuple = mnist(
                    resize=hparams["resize_to_small"],
                    filter=hparams["filter"])
        self.continuation_steps = hparams["continuation_steps"]

        self.output_file = hparams["meta"]["output_dir"]
        self._delta_s = hparams["delta_s"]
        self._prev_delta_s = hparams["delta_s"]
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
            if i == 0 and self.hparams["natural_start"]:
                print(f" unconstrained solver for 1st step")
                concat_states = [
                    self._prev_state,
                    pytree_element_add(self._prev_bparam, 0.05),
                ]

                corrector = UnconstrainedCorrector(
                    objective=self.objective,
                    concat_states=concat_states,
                    grad_fn=self.grad_fn,
                    value_fn=self.value_func,
                    accuracy_fn=self.accuracy_fn,
                    hparams=self.hparams,
                    dataset_tuple=self.dataset_tuple
                )
                state, bparam, quality, value, val_loss = corrector.correction_step()
                self._state_wrap.state = state
                self._bparam_wrap.state = bparam
                del corrector, state, bparam, quality, value, concat_states

            print(self._value_wrap.get_record(), self._bparam_wrap.get_record())
            self._state_wrap.counter = i
            self._bparam_wrap.counter = i
            self._value_wrap.counter = i
            self._quality_wrap.counter = i
            self.sw.write(
                [
                    self._state_wrap.get_record(),
                    self._bparam_wrap.get_record(),
                    self._value_wrap.get_record(),
                    self._quality_wrap.get_record(),
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
                hparams=self.hparams,
            )
            predictor.prediction_step()

            self.prev_secant_direction = predictor.secant_direction

            concat_states = [predictor.state, predictor.bparam]
            del predictor
            gc.collect()

            corrector = UnconstrainedCorrector(
                objective=self.objective,
                concat_states=concat_states,
                grad_fn=self.grad_fn,
                value_fn=self.value_func,
                accuracy_fn=self.accuracy_fn,
                hparams=self.hparams,
                dataset_tuple=self.dataset_tuple
            )
            state, bparam, quality, value, val_loss, val_acc = corrector.correction_step()

            corrector_omega = 0.005 # why fixed check TODO
            self._prev_delta_s = self._delta_s
            self._delta_s = corrector_omega * self._delta_s
            self._delta_s = min(self._delta_s, self.hparams["max_arc_len"])
            self._delta_s = max(self._delta_s, self.hparams["min_arc_len"])

            self._state_wrap.state = state
            self._bparam_wrap.state = bparam
            self._value_wrap.state = value
            self._quality_wrap.state = quality
            del corrector
            del concat_states
            gc.collect()
            if self._bparam_wrap.state[0] >= self.hparams["lambda_max"]:
                self.sw.write(
                    [
                        self._state_wrap.get_record(),
                        self._bparam_wrap.get_record(),
                        self._value_wrap.get_record(),
                        self._quality_wrap.get_record(),
                    ]
                )
                break
            mlflow.log_metrics({
                "train_loss": float(self._value_wrap.state),
                "delta_s": float(self._delta_s),
                "norm grads": float(self._quality_wrap.state),
                "val_loss": float(val_loss)
            }, i)

