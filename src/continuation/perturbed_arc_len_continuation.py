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


class PerturbedPseudoArcLenContinuation(PseudoArcLenContinuation):
    def __init__(
        self,
        state,
        bparam,
        state_0,
        bparam_0,
        counter,
        objective,
        dual_objective,
        lagrange_multiplier,
        output_file,
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
            lagrange_multiplier,
            output_file,
            hparams,
        )
        self.key_state = key_state

    def run(self):
        self.sw = StateWriter(f"{self.output_file}/version_{self.key_state}.json")

        for i in range(self.continuation_steps):
            self._state_wrap.counter = i
            self._bparam_wrap.counter = i
            self.sw.write(
                [self._state_wrap.get_record(), self._bparam_wrap.get_record()]
            )

            concat_states = [
                (self._state_wrap.state, self._bparam_wrap.state),
                (self._prev_state, self._prev_bparam),
            ]

            predictor = SecantPredictor(
                concat_states=concat_states, delta_s=self._delta_s, omega=self._omega
            )

            state_guess, bparam_guess = predictor.prediction_step()
            secant_vector = predictor.get_secant_vector_concat()
            secant_concat = predictor.get_secant_concat()

            self._prev_state = copy.deepcopy(self._state_wrap.state)
            self._prev_bparam = copy.deepcopy(self._bparam_wrap.state)

            concat_states = [
                state_guess,
                bparam_guess,
                secant_vector,
                secant_concat,
            ]
            corrector = PerturbedCorrecter(
                optimizer=self.opt,
                objective=self.objective,
                dual_objective=self.dual_objective,
                lagrange_multiplier=self._lagrange_multiplier,
                concat_states=concat_states,
                delta_s=self._delta_s,
                ascent_opt=self.ascent_opt,
                key_state=self.key_state,
            )
            state, bparam = corrector.correction_step()

            self._state_wrap.state = state
            self._bparam_wrap.state = bparam
