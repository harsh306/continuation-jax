from cjax.continuation._arc_len_continuation import Continuation
from cjax.continuation.states.state_variables import StateWriter
from cjax.utils.data_img_gamma import mnist_gamma
from cjax.utils.datasets import mnist
from cjax.continuation.methods.predictor.arc_secant_predictor import SecantPredictor
from cjax.continuation.methods.corrector.perturb_parc_evolve import (
    PerturbedFixedCorrecter,
)
from cjax.continuation.methods.corrector.unconstrained_corrector import (
    UnconstrainedCorrector,
)
import jax.numpy as np
from jax.tree_util import *
import copy
from cjax.utils.profiler import profile
from jax.experimental.optimizers import l2_norm
from cjax.continuation.states.state_variables import StateVariable, StateWriter
from jax import jit, grad
import numpy.random as npr
from cjax.utils.math_trees import pytree_element_add
import mlflow
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
        accuracy_fn,
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
        self.accuracy_fn1= jit(accuracy_fn)
        self.value_func = jit(self.objective)

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

        self._value_wrap = StateVariable(
            0.06, counter
        )  # TODO: fix with a static batch (test/train)
        self._quality_wrap = StateVariable(
            l2_norm(self._state_wrap.state) / 10, counter
        )

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
        self.perturb_index = key_state
        self.sw = StateWriter(f"{self.output_file}/version_{self.perturb_index}.json")
        self.key_state = key_state + npr.randint(100, 200)
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
            self._state_wrap.counter = i
            self._bparam_wrap.counter = i
            self._value_wrap.counter = i
            self._quality_wrap.counter = i

            if i == 0 and self.hparams["natural_start"]:
                print(f" unconstrained solver for 1st step")
                concat_states = [
                    self._prev_state,
                    pytree_element_add(self._prev_bparam, 0.03),
                ]

                corrector = UnconstrainedCorrector(
                    objective=self.objective,
                    concat_states=concat_states,
                    grad_fn=self.compute_grad_fn,
                    value_fn=self.value_func,
                    accuracy_fn=self.accuracy_fn1,
                    hparams=self.hparams,
                    dataset_tuple=self.dataset_tuple,
                )
                state, bparam, quality, value, val_loss, val_acc = corrector.correction_step()
                if self.hparams["double_natural_start"]:  # TODO: refactor natural and double natural start
                    self._prev_state = state
                    self._prev_bparam = bparam
                    print(f" unconstrained solver for 2nd step")
                    concat_states = [
                        self._prev_state,
                        pytree_element_add(self._prev_bparam, 0.07),
                    ]

                    corrector = UnconstrainedCorrector(
                        objective=self.objective,
                        concat_states=concat_states,
                        grad_fn=self.compute_grad_fn,
                        value_fn=self.value_func,
                        accuracy_fn=self.accuracy_fn1,
                        hparams=self.hparams,
                        dataset_tuple=self.dataset_tuple,
                    )
                    state, bparam, quality, value, val_loss, val_acc = corrector.correction_step()

                self._state_wrap.state = state
                self._bparam_wrap.state = bparam

            print("delta_s",
                self._value_wrap.get_record(),
                self._bparam_wrap.get_record(),
                self._delta_s,
            )
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

            self.hparams["sphere_radius"] = (
                self.hparams["sphere_radius_m"] * self._delta_s
            )  # l2_norm(predictor.secant_direction)
            mlflow.log_metric(f"sphere_radius{self.perturb_index}", self.hparams["sphere_radius"], i)
            mlflow.log_metric(f"delta_s{self.perturb_index}", self._delta_s, i)
            concat_states = [
                predictor.state,
                predictor.bparam,
                predictor.secant_direction,
                {"state": predictor.state, "bparam": predictor.bparam},
            ]
            corrector = PerturbedFixedCorrecter(
                objective=self.objective,
                dual_objective=self.dual_objective,
                accuracy_fn1=self.accuracy_fn1,
                value_fn=self.value_func,
                concat_states=concat_states,
                key_state=self.key_state,
                compute_min_grad_fn=self.compute_min_grad_fn,
                compute_grad_fn=self.compute_grad_fn,
                hparams=self.hparams,
                delta_s=self._delta_s,
                pred_state=[self._state_wrap.state, self._bparam_wrap.state],
                pred_prev_state=[self._state_wrap.state, self._bparam_wrap.state],
                counter=self.continuation_steps,
                dataset_tuple=self.dataset_tuple,
            )
            self._prev_state = copy.deepcopy(self._state_wrap.state)
            self._prev_bparam = copy.deepcopy(self._bparam_wrap.state)

            (
                state,
                bparam,
                quality,
                value,
                val_loss,
                val_acc,
                corrector_omega,
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
            # self._omega = corrector_omega
            self._prev_delta_s = self._delta_s
            self._delta_s = corrector_omega * self._delta_s
            self._delta_s = min(self._delta_s, self.hparams["max_arc_len"])
            self._delta_s = max(self._delta_s, self.hparams["min_arc_len"])

            if (bparam[0] >= self.hparams["lambda_max"]) or (
                bparam[0] <= self.hparams["lambda_min"]
            ):
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
                f"train_loss{self.perturb_index}": float(self._value_wrap.state),
                f"delta_s{self.perturb_index}": float(self._delta_s),
                f"norm grads{self.perturb_index}": float(self._quality_wrap.state),
                f"val_loss{self.perturb_index}": float(val_loss),
                f"val_acc{self.perturb_index}": float(val_acc),
                f"corrector_omega{self.perturb_index}": float(corrector_omega)
            }, i)
