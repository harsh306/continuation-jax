from cjax.continuation.methods.predictor.base_predictor import Predictor
from jax.tree_util import tree_multimap
from cjax.utils.math_trees import *
from jax import jit


class SecantPredictor(Predictor):
    def __init__(
        self,
        concat_states,
        delta_s,
        prev_delta_s,
        omega,
        net_spacing_param,
        net_spacing_bparam,
        hparams,
    ):
        super().__init__(concat_states)
        self._prev_state = None
        self._prev_bparam = None
        self.prev_delta_s = prev_delta_s
        self.delta_s = delta_s
        self.omega = omega
        self.net_spacing_param = net_spacing_param
        self.net_spacing_bparam = net_spacing_bparam
        self.secantvar_state = None
        self.secantvar_bparam = None
        self.secant_direction = dict()
        self.prev_secant_direction = None # only for choose direction logic.
        self.hparams = hparams

    def _assign_states(self):
        self._prev_state, self._prev_bparam = self._concat_states[0]
        self._state, self._bparam = self._concat_states[1]
        self.prev_secant_direction = self._concat_states[2]

    @staticmethod
    @jit
    def _compute_secant(
        _state,
        _bparam,
        _prev_state,
        _prev_bparam,
        delta_s,
        prev_delta_s,
    ):
        secant_direction = {}
        state_sub = pytree_sub(_state, _prev_state)
        bparam_sub = pytree_sub(_bparam, _prev_bparam)
        secant_direction.update(
            {"state": pytree_element_mul(state_sub, delta_s / prev_delta_s)}
        )

        secant_direction.update(
            {"bparam": pytree_element_mul(bparam_sub, delta_s / prev_delta_s)}
        )

        return secant_direction

    def prediction_step(self):
        """Given current state predict next state.
        Updates (state_guess: problem parameters, bparam_guess: continuation parameter)
        """
        self._assign_states()

        self.secant_direction = self._compute_secant(
            self._state,
            self._bparam,
            self._prev_state,
            self._prev_bparam,
            self.delta_s,
            self.prev_delta_s,
        )
        # self._choose_direction()
        self._state = tree_multimap(
            lambda a, b: a + b,
            self._state,
            self.secant_direction["state"],
        )
        self._bparam = tree_multimap(
            lambda a, b: a + b,
            self._bparam,
            self.secant_direction["bparam"],
        )

    def get_secant_vector_concat(self):
        """Concatenated secant vector.

        Returns:
          [state_vector: problem parameters, bparam_vector: continuation parameter] list
        """
        return self.secant_direction

    # TODO: below function needed?
    def get_secant_concat(self):
        """Concatenated secant guess/point.

        Returns:
          [state_guess: problem parameters, bparam_guess: continuation parameter] list
        """
        concat = dict()
        concat.update({"state": self._state})
        concat.update({"bparam": self._bparam})
        return concat
