from typing import Tuple

from jax import grad, jit

from cjax.continuation.methods.corrector.base_corrector import Corrector
from cjax.optimizer.optimizer import Optimizer


class UnconstrainedCorrector(Corrector):
    """Minimize the objective using gradient based method."""

    def __init__(
        self, optimizer: Optimizer, objective, concat_states, grad_fn, warmup_period
    ):
        self.concat_states = concat_states
        self._state = None
        self._bparam = None
        self.opt = optimizer
        self.objective = objective
        self.warmup_period = warmup_period
        self.grad_fn = grad_fn

    def _assign_states(self):
        self._state, self._bparam = self.concat_states

    def correction_step(self) -> Tuple:
        """Given the current state optimize to the correct state.

        Returns:
          (state: problem parameters, bparam: continuation parameter) Tuple
        """
        self._assign_states()
        for k in range(self.warmup_period):
            grads = self.grad_fn(self._state, self._bparam)
            self._state = self.opt.update_params(self._state, grads[0])
        return self._state, self._bparam
