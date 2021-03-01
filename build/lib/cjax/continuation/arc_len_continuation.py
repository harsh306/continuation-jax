from cjax.continuation.base_continuation import Continuation
from cjax.continuation.states.state_variables import StateVariable, StateWriter
from cjax.continuation.methods.predictor.secant_predictor import SecantPredictor
from cjax.continuation.methods.corrector.constrained_corrector import (
    ConstrainedCorrector,
)
from jax import jit, grad
import gc
from cjax.utils import profile
from cjax.optimizer.optimizer import OptimizerCreator


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
        hparams,
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

        self._value_wrap = StateVariable(self.objective(state, bparam), counter)

        # optimizer
        self.opt = OptimizerCreator(
            opt_string=hparams["meta"]["optimizer"], learning_rate=hparams["natural_lr"]
        ).get_optimizer()
        self.ascent_opt = OptimizerCreator(
            opt_string=hparams["meta"]["ascent_optimizer"], learning_rate=hparams["ascent_lr"]
        ).get_optimizer()

        # every step hparams
        self.continuation_steps = hparams["continuation_steps"]
        self._lagrange_multiplier = hparams["lagrange_init"]

        self._delta_s = hparams["delta_s"]
        self._omega = hparams["omega"]

        # grad functions # should be pure functional
        self.compute_min_grad_fn = jit(grad(self.dual_objective, [0, 1]))
        self.compute_max_grad_fn = jit(grad(self.dual_objective, [2]))
        self.compute_grad_fn = jit(grad(self.objective, [0]))

        # extras
        self.sw = None
        self.state_tree_def = None
        self.bparam_tree_def = None
        self.output_file = hparams["meta"]["output_dir"]
        self.prev_secant_direction = None

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
                net_spacing_param=self.hparams["net_spacing_param"],
                net_spacing_bparam=self.hparams["net_spacing_bparam"],
                hparams=self.hparams
            )
            predictor.prediction_step()
            self.prev_secant_direction = predictor.secant_direction
            self._prev_state = self._state_wrap.state
            self._prev_bparam = self._bparam_wrap.state

            concat_states = [
                predictor.state,
                predictor.bparam,
                predictor.secant_direction,
                predictor.get_secant_concat(),
            ]
            del predictor
            gc.collect()
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
                hparams=self.hparams,
            )
            state, bparam = corrector.correction_step()
            value = self.value_func(state, bparam)
            self._state_wrap.state = state
            self._bparam_wrap.state = bparam
            self._value_wrap.state = value
            del corrector
            gc.collect()
