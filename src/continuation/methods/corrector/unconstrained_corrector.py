from typing import Tuple

from jax import grad, jit

from src.continuation.methods.corrector.base_corrector import Corrector
from src.optimizer.optimizer import Optimizer


class UnconstrainedCorrector(Corrector):
    def __init__(
        self,
        optimizer: Optimizer,
        objective,
        concat_states,
    ):
        self.concat_states = concat_states
        self._state = None
        self._bparam = None
        self.opt = optimizer
        self.objective = objective

    def _compute_grads(self):
        grad_fn = jit(grad(self.objective, argnums=[0]))
        grads = grad_fn(self._state, self._bparam)
        return grads[0]

    def correction_step(self) -> Tuple:
        self.assign_states()
        grads = self._compute_grads()
        self._state = self.opt.update_params(self._state, grads)
        return self._state, self._bparam

    def assign_states(self):
        self._state, self._bparam = self.concat_states
