from typing import Tuple
from jax import grad, jit
from cjax.continuation.methods.corrector.base_corrector import Corrector


class ConstrainedCorrector(Corrector):
    """Minimize the objective using gradient based method along with some constraint"""

    def __init__(
        self,
        optimizer,
        objective,
        dual_objective,
        lagrange_multiplier,
        concat_states,
        delta_s,
        ascent_opt,
        compute_min_grad_fn,
        compute_max_grad_fn,
        compute_grad_fn,
        hparams,
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
        self.warmup_period = hparams["warmup_period"]
        self.ascent_period = hparams["ascent_period"]
        self.descent_period = hparams["descent_period"]
        self.max_norm_state = hparams["max_bounds"]
        self.hparams = hparams
        self.compute_min_grad_fn = compute_min_grad_fn
        self.compute_max_grad_fn = compute_max_grad_fn
        self.compute_grad_fn = compute_grad_fn
        self._assign_states()

    def _assign_states(self):
        self._state = self.concat_states[0]
        self._bparam = self.concat_states[1]
        self._state_secant_vector = self.concat_states[2]
        self._state_secant_c2 = self.concat_states[3]

    def correction_step(self) -> Tuple:
        """Given the current state optimize to the correct state.

        Returns:
          (state: problem parameters, bparam: continuation parameter) Tuple
        """
        for k in range(self.warmup_period):
            grads = self.compute_grad_fn(self._state, self._bparam)
            self._state = self.opt.update_params(self._state, grads[0])

        for k in range(self.ascent_period):
            lagrange_grads = self.compute_max_grad_fn(
                self._state,
                self._bparam,
                self._lagrange_multiplier,
                self._state_secant_c2,
                self._state_secant_vector,
                self.delta_s,
            )
            self._lagrange_multiplier = self.ascent_opt.update_params(
                self._lagrange_multiplier, lagrange_grads[0]
            )
            for j in range(self.descent_period):
                state_grads, bpram_grads = self.compute_min_grad_fn(
                    self._state,
                    self._bparam,
                    self._lagrange_multiplier,
                    self._state_secant_c2,
                    self._state_secant_vector,
                    self.delta_s,
                )
                self._bparam = self.opt.update_params(self._bparam, bpram_grads)
                self._state = self.opt.update_params(self._state, state_grads)

        return self._state, self._bparam
