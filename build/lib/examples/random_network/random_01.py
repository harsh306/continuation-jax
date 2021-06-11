from cjax.utils.random_network_data_pair import generate_data_01
from cjax.utils.abstract_problem import AbstractProblem
from jax.experimental.optimizers import l2_norm
from jax.tree_util import tree_map
from jax.nn import sigmoid
import jax.numpy as np

inputs, outputs, param, bparam, init_fun, predict_fun = generate_data_01()


class RandomExp(AbstractProblem):
    def __init__(self):
        pass

    @staticmethod
    def objective(params, bparam) -> float:
        logits = predict_fun(params, inputs, bparam=bparam[0], activation_func=sigmoid)
        loss = np.mean((np.subtract(logits, outputs)))
        loss += l2_norm(params) + l2_norm(bparam)
        return loss

    def initial_value(self):
        return param, bparam

    def initial_values(self):
        state, bparam = self.initial_value()
        state_1 = tree_map(lambda a: a + 0.005, state)
        states = [state, state_1]
        bparam_1 = tree_map(lambda a: a + 0.10, bparam)
        bparams = [bparam, bparam_1]
        return states, bparams
