from src.continuation.base_continuation import Continuation
from src.continuation.states.state_variables import StateVariable, StateWriter
from src.optimizer.optimizer import OptimizerCreator
from src.continuation.methods.predictor.natural_predictor import NaturalPredictor
from src.continuation.methods.corrector.unconstrained_corrector import (
    UnconstrainedCorrector,
)
from jax.tree_util import *


class NaturalContinuation(Continuation):
    """Natural Continuation strategy.

    Composed of natural predictor and unconstrained corrector"""

    def __init__(self, state, bparam, counter, objective, output_file, hparams):
        self._state_wrap = StateVariable(state, counter)
        self._bparam_wrap = StateVariable(bparam, counter)
        self.objective = objective
        self.inputs = 0.0
        self.outputs = 0.0
        self.opt = OptimizerCreator(
            opt_string=hparams["meta"]["optimizer"], learning_rate=hparams["natural_lr"]
        ).get_optimizer()
        self.continuation_steps = hparams["continuation_steps"]
        self.sw = None
        self.output_file = output_file
        self._delta_s = hparams["delta_s"]

    def run(self):
        """Runs the continuation strategy.

        A continuation strategy that defines how predictor and corrector components of the algorithm
        interact with the states of the mathematical system.
        """
        self.sw = StateWriter(f"{self.output_file}/version.json")
        for i in range(self.continuation_steps):

            self._state_wrap.counter = i
            self._bparam_wrap.counter = i
            self.sw.write(
                [self._state_wrap.get_record(), self._bparam_wrap.get_record()]
            )

            concat_states = [self._state_wrap.state, self._bparam_wrap.state]
            predictor = NaturalPredictor(
                concat_states=concat_states, delta_s=self._delta_s
            )
            state, bparam = predictor.prediction_step()

            concat_states = [state, bparam]
            corrector = UnconstrainedCorrector(
                optimizer=self.opt,
                objective=self.objective,
                concat_states=concat_states,
            )
            state, bparam = corrector.correction_step()

            self._state_wrap.state = state
            self._bparam_wrap.state = bparam
