from src.continuation.base_continuation import Continuation
from src.continuation.arc_len_continuation import PseudoArcLenContinuation
from src.continuation.states.state_variables import StateVariable, StateWriter
from src.optimizer.optimizer import GDOptimizer
from src.continuation.methods.predictor.secant_predictor import SecantPredictor
from src.continuation.methods.corrector.perturbed_constrained_corrector import (
    PerturbedCorrecter,
)
from jax.tree_util import *
import copy
from utils.profiler import profile
import gc
from utils.math_trees import pytree_relative_error

# TODO: make **kwargs availible


class PerturbedPseudoArcLenContinuation(PseudoArcLenContinuation):
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
        super().__init__(
            state,
            bparam,
            state_0,
            bparam_0,
            counter,
            objective,
            dual_objective,
            hparams,
        )
        self.key_state = key_state

    @profile(sort_by="cumulative", lines_to_print=10, strip_dirs=True)
    def run(self):
        """Runs the continuation strategy.

        A continuation strategy that defines how predictor and corrector components of the algorithm
        interact with the states of the mathematical system.
        """
        self.sw = StateWriter(f"{self.output_file}/version_{self.key_state}.json")

        for i in range(self.continuation_steps):
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
                (self._state_wrap.state, self._bparam_wrap.state),
                (self._prev_state, self._prev_bparam),
                self.prev_secant_direction,
            ]

            predictor = SecantPredictor(
                concat_states=concat_states,
                delta_s=self._delta_s,
                omega=self._omega,
                net_spacing=self.hparams["net_spacing"],
            )
            predictor.prediction_step()
            self.prev_secant_direction = predictor.secant_direction

            concat_states = [
                predictor.state,
                predictor.bparam,
                predictor.secant_direction,
                predictor.get_secant_concat(),
            ]
            del predictor
            gc.collect()
            corrector = PerturbedCorrecter(
                optimizer=self.opt,
                objective=self.objective,
                dual_objective=self.dual_objective,
                lagrange_multiplier=self._lagrange_multiplier,
                concat_states=concat_states,
                delta_s=self._delta_s,
                ascent_opt=self.ascent_opt,
                key_state=self.key_state,
                compute_min_grad_fn=self.compute_min_grad_fn,
                compute_max_grad_fn=self.compute_max_grad_fn,
                compute_grad_fn=self.compute_grad_fn,
                hparams=self.hparams,
                pred_state=[self._state_wrap.state, self._bparam_wrap.state],
                pred_prev_state=[self._state_wrap.state, self._bparam_wrap.state],
            )
            self._prev_state = copy.deepcopy(self._state_wrap.state)
            self._prev_bparam = copy.deepcopy(self._bparam_wrap.state)

            state, bparam = corrector.correction_step()
            value = self.value_func(state, bparam)
            print(
                "How far ....", pytree_relative_error(self._bparam_wrap.state, bparam)
            )
            self._state_wrap.state = state
            self._bparam_wrap.state = bparam
            self._value_wrap.state = value
            del corrector
            gc.collect()
