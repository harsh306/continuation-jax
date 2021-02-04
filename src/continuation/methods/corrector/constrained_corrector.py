from typing import Tuple
from jax import grad, jit
from src.continuation.methods.corrector.base_corrector import Corrector


class ConstrainedCorrector(Corrector):
    def __init__(
        self,
        optimizer,
        objective,
        dual_objective,
        lagrange_multiplier,
        concat_states,
        delta_s,
        ascent_opt
    ):
        self.concat_states = concat_states
        self._state = None
        self._bparam = None
        self.opt = optimizer
        self.ascent_opt = ascent_opt
        self.objective = objective
        self.dual_objective = dual_objective
        self._lagrange_multiplier = lagrange_multiplier
        self._state_secant_vector = None
        self._state_secant_c2 = None
        self.delta_s = delta_s
        self.warmup_period = 10
        self.ascent_period = 5
        self.descent_period = 10
        self.assign_states()

    def assign_states(self):
        self._state = self.concat_states[0]
        self._bparam = self.concat_states[1]
        self._state_secant_vector = self.concat_states[2]
        self._state_secant_c2 = self.concat_states[3]

    def _compute_grads(self) -> list:
        grad_fn = jit(grad(self.objective, argnums=[0]))
        grads = grad_fn(self._state, self._bparam)
        return grads[0]

    def _compute_min_grads(self) -> Tuple:
        grad_fn = jit(grad(self.dual_objective, [0, 1]))
        state_grads, bparam_grads = grad_fn(
            self._state,
            self._bparam,
            self._lagrange_multiplier,
            self._state_secant_c2,
            self._state_secant_vector,
            self.delta_s
        )
        return state_grads, bparam_grads

    def _compute_max_grads(self) -> list:
        grad_fn = jit(grad(self.dual_objective, argnums=[2]))
        grads = grad_fn(
            self._state,
            self._bparam,
            self._lagrange_multiplier,
            self._state_secant_c2,
            self._state_secant_vector,
            self.delta_s,
        )
        return grads[0]

    def correction_step(self) -> Tuple:
        # TODO: Multiple optimizers can be made available

        for k in range(self.warmup_period):
            grads = self._compute_grads()
            self._state = self.opt.update_params(self._state, grads)

        for k in range(self.ascent_period):
            lagrange_grads = self._compute_max_grads()
            self._lagrange_multiplier = self.ascent_opt.update_params(
                self._lagrange_multiplier, lagrange_grads
            )
            for j in range(self.descent_period):
                state_grads, bpram_grads = self._compute_min_grads()
                self._bparam = self.opt.update_params(self._bparam, bpram_grads)
                self._state = self.opt.update_params(self._state, state_grads)

        return self._state, self._bparam
