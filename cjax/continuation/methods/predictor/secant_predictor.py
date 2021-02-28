from cjax.continuation.methods.predictor.base_predictor import Predictor
from jax.tree_util import tree_multimap
from cjax.utils.math_trees import *
from jax import jit


class SecantPredictor(Predictor):
    def __init__(self, concat_states, delta_s, omega, net_spacing_param, net_spacing_bparam, hparams):
        super().__init__(concat_states)
        self._prev_state = None
        self._prev_bparam = None
        self.delta_s = delta_s
        self.omega = omega
        self.net_spacing_param = net_spacing_param
        self.net_spacing_bparam = net_spacing_bparam
        self.secantvar_state = None
        self.secantvar_bparam = None
        self.secant_direction = dict()
        self.prev_secant_direction = None
        self.hparams = hparams

    def _assign_states(self):
        self._prev_state, self._prev_bparam = self._concat_states[0]
        self._state, self._bparam = self._concat_states[1]
        self.prev_secant_direction = self._concat_states[2]

    def _choose_direction(self):
        if self.prev_secant_direction:
            inner_prod = pytree_dot(
                pytree_normalized(self.prev_secant_direction),
                pytree_normalized(self.secant_direction),
            )
            state_neg_inner_prod = pytree_dot(
                [
                    self.prev_secant_direction["state"],
                    self.prev_secant_direction["bparam"],
                ],
                [
                    pytree_element_mul(self.secant_direction["state"], -1.0),
                    self.secant_direction["bparam"],
                ],
            )
            bparam_neg_inner_prod = pytree_dot(
                [
                    self.prev_secant_direction["state"],
                    self.prev_secant_direction["bparam"],
                ],
                [
                    self.secant_direction["state"],
                    pytree_element_mul(self.secant_direction["bparam"], -1.0),
                ],
            )
            neg_inner_prod = pytree_dot(
                self.prev_secant_direction,
                pytree_element_mul(self.secant_direction, -1.0),
            )
            prods = [
                inner_prod,
                state_neg_inner_prod,
                bparam_neg_inner_prod,
                neg_inner_prod,
            ]

            # prod_diffs = list(map(lambda x: abs(x - 1.0), prods))
            # print(prod_diffs)
            min_index = prods.index(max(prods))

            if min_index == 1:
                self.secant_direction.update(
                    {"state": pytree_element_mul(self.secant_direction["state"], -1.0)}
                )
            elif min_index == 2:
                self.secant_direction.update(
                    {
                        "bparam": pytree_element_mul(
                            self.secant_direction["bparam"], -1.0
                        )
                    }
                )
            elif min_index == 3:
                self.secant_direction = pytree_element_mul(self.secant_direction, -1.0)
            else:
                pass

    @staticmethod
    @jit
    def _compute_secant(_state, _bparam, _prev_state, _prev_bparam, net_spacing_param, net_spacing_bparam, omega):
        secant_direction = {}
        state_sub = pytree_sub(_state, _prev_state)
        secant_direction.update(
            {
                "state": pytree_element_mul(
                    state_sub, net_spacing_param/(l2_norm(state_sub) + 1e-2)
                )
            }
        )
        bparam_sub = pytree_sub(_bparam, _prev_bparam)
        secant_direction.update(
            {"bparam": pytree_element_mul(bparam_sub, net_spacing_bparam/np.square(l2_norm(bparam_sub)))
             }
        )
        return secant_direction


    def prediction_step(self):
        """Given current state predict next state.
        Updates (state_guess: problem parameters, bparam_guess: continuation parameter)
        """
        self._assign_states()

        self.secant_direction = self._compute_secant(self._state, self._bparam, self._prev_state,
                                                     self._prev_bparam, self.net_spacing_param,
                                                     self.net_spacing_bparam, self.omega)
        # self._choose_direction()
        try_state = None
        try_bparam = None
        for u in range(self.hparams['retry']):
            try_state = tree_multimap(
                lambda a, b: a + self.omega * b, self._state, self.secant_direction["state"]
            )
            try_bparam = tree_multimap(
                lambda a, b: a + self.omega * b,
                self._bparam,
                self.secant_direction["bparam"],
            )
            relative_error = l2_norm(pytree_relative_error(self._bparam, try_bparam))
            if self.hparams['adaptive']:
                if (
                        relative_error > self.hparams['re_tresh']
                ):
                    self.omega = self.omega * self.hparams['omega_d']
                    print(
                        f"retry as relative_error: {relative_error}"
                    )
                else:
                    break
            else:
                break

        self._state = try_state
        self._bparam = try_bparam
        print('predictor', self._bparam)
        del try_bparam, try_state

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
