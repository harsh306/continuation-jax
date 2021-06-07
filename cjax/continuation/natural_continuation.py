from cjax.continuation.base_continuation import Continuation
from cjax.continuation.states.state_variables import StateVariable, StateWriter
from cjax.optimizer.optimizer import OptimizerCreator
from cjax.continuation.methods.predictor.natural_predictor import NaturalPredictor
from cjax.utils.data_img_gamma import mnist_gamma
from cjax.utils.datasets import mnist, get_mnist_data, meta_mnist
from cjax.continuation.methods.corrector.unconstrained_corrector import (
    UnconstrainedCorrector,
)
from jax.tree_util import *
import gc
from cjax.utils.profiler import profile
from jax import jit, grad
import jax.numpy as np
import mlflow


class NaturalContinuation(Continuation):
    """Natural Continuation strategy.

    Composed of natural predictor and unconstrained corrector"""

    def __init__(self, state, bparam, counter, objective, accuracy_fn, hparams):
        self._state_wrap = StateVariable(state, counter)
        self._bparam_wrap = StateVariable(bparam, counter)
        self.objective = objective
        self.value_func = jit(self.objective)
        self.accuracy_fn = jit(accuracy_fn)
        self._value_wrap = StateVariable(2.0, counter)
        self._quality_wrap = StateVariable(0.25, counter)
        self.sw = None
        self.hparams = hparams
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
        self._delta_s = hparams["delta_bparams"]
        self.grad_fn = jit(
            grad(self.objective, argnums=[0])
        )  # TODO: vmap is not fully supported with stax

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
            self._quality_wrap.counter = i
            self.sw.write(
                [
                    self._state_wrap.get_record(),
                    self._bparam_wrap.get_record(),
                    self._value_wrap.get_record(),
                    self._quality_wrap.get_record(),
                ]
            )

            concat_states = [self._state_wrap.state, self._bparam_wrap.state]
            predictor = NaturalPredictor(
                concat_states=concat_states, delta_s=self._delta_s
            )
            predictor.prediction_step()

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
                dataset_tuple=self.dataset_tuple,
            )
            state, bparam, quality, value, val_loss, val_acc = corrector.correction_step()

            clip_lambda = lambda g: np.where(
                (g > self.hparams["lambda_max"]), self.hparams["lambda_max"], g
            )
            bparam = tree_map(clip_lambda, bparam)
            clip_lambda = lambda g: np.where(
                (g < self.hparams["lambda_min"]), self.hparams["lambda_min"], g
            )
            bparam = tree_map(clip_lambda, bparam)

            self._state_wrap.state = state
            self._bparam_wrap.state = bparam
            self._value_wrap.state = value
            self._quality_wrap.state = quality
            del corrector
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
                "val_loss": float(val_loss),
                "val_acc": float(val_acc)
            }, i)

