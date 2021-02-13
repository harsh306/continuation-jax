from src.continuation.base_continuation import Continuation
from src.continuation.states.state_variables import StateVariable, StateWriter
from src.optimizer.optimizer import GDOptimizer, GAOptimizer
from src.continuation.methods.predictor.secant_predictor import SecantPredictor
from src.continuation.methods.corrector.constrained_corrector import (
    ConstrainedCorrector,
)
from jax.tree_util import *
import copy
from jax import jit, grad
from utils.profiler import profile


class PseudoArcLenContinuation(Continuation):
    # May be refactor to only one continuation TODO
    """Pseudo Arc-length Continuation strategy.

    Composed of secant predictor and constrained corrector"""

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
    ):
        self._state_wrap = StateVariable(state, counter)
        self._bparam_wrap = StateVariable(
            bparam, counter
        )  # Todo : save tree def, always unlfatten before compute_grads
        self.objective = objective
        self.dual_objective = dual_objective
        # self.inputs = inputs
        # self.outputs = outputs
        self.opt = GDOptimizer(learning_rate=hparams["natural_lr"])
        self.ascent_opt = GAOptimizer(learning_rate=hparams["ascent_lr"])
        self.continuation_steps = hparams["continuation_steps"]
        self._prev_state = state_0
        self._prev_bparam = bparam_0
        self._lagrange_multiplier = lagrange_multiplier
        self.sw = None
        self.state_tree_def = None
        self.bparam_tree_def = None
        self.hparams = hparams
        self._delta_s = hparams["delta_s"]
        self._omega = hparams["omega"]
        self.output_file = output_file
        self.compute_min_grad_fn = jit(grad(self.dual_objective, [0, 1]))
        self.compute_max_grad_fn = jit(grad(self.dual_objective, [2]))
        self.compute_grad_fn = jit(grad(self.objective, [0]))

    @profile(sort_by="cumulative", lines_to_print=10, strip_dirs=True)
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

            concat_states = [
                (self._state_wrap.state, self._bparam_wrap.state),
                (self._prev_state, self._prev_bparam),
            ]

            predictor = SecantPredictor(
                concat_states=concat_states, delta_s=self._delta_s, omega=self._omega
            )
            predictor.prediction_step()

            self._prev_state = self._state_wrap.state
            self._prev_bparam = self._bparam_wrap.state

            concat_states = [
                predictor.state,
                predictor.bparam,
                predictor.secant_direction,
                predictor.get_secant_concat(),
            ]
            del predictor
            corrector = ConstrainedCorrector(
                optimizer=self.opt,
                objective=self.objective,
                dual_objective=self.dual_objective,
                lagrange_multiplier=self._lagrange_multiplier,
                concat_states=concat_states,
                delta_s=self._delta_s,
                ascent_opt=self.ascent_opt,
                compute_min_grad_fn=self.compute_min_grad_fn,
                compute_max_grad_fn=self.compute_max_grad_fn,
                compute_grad_fn=self.compute_grad_fn,
            )
            state, bparam = corrector.correction_step()

            self._state_wrap.state = state
            self._bparam_wrap.state = bparam
