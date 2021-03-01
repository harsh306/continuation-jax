from examples.abstract_problem import AbstractProblem
from typing import Tuple
import jax.numpy as np
from jax import grad, jit, vmap
from jax import random
from jax.scipy.special import logsumexp
from jax.tree_util import *

inputs = random.normal(random.PRNGKey(1), (1, 2 * 2))
outputs = np.ones(shape=(1, 1))


class SimpleNeuralNetwork(AbstractProblem):
    def __init__(self):
        pass

    @staticmethod
    def objective(params, bparam) -> float:
        def neural_net_predict(params, bparam, inputs):
            # per-example predictions
            activations = inputs
            for w, b in params[:-1]:
                activations = np.dot(w, activations) + b + bparam
                # activations = relu(_outputs)

            final_w, final_b = params[-1]
            logits = np.dot(final_w, activations) + final_b
            return logits - logsumexp(logits)

        def regularizer_l2(params):
            reg_add = 0.0
            for (w, b) in params:
                reg_add += np.sum([np.sum(np.square(w)), np.sum(np.square(b))])
            return reg_add

        # vectorization of mini-batch of data.
        # #3rd argument's 0th-axis is vmaped. --> inputs(10)
        batched_predict = vmap(neural_net_predict, in_axes=(None, None, 0))

        preds = batched_predict(params, bparam, inputs)
        mse = -np.mean(preds * outputs)
        reg = regularizer_l2(params)
        return mse + reg

    def init_network_params(self, sizes, key):
        keys = random.split(key, len(sizes))
        return [
            self.random_layer_params(m, n, k)
            for m, n, k in zip(sizes[:-1], sizes[1:], keys)
        ]

    @staticmethod
    def random_layer_params(m, n, key, scale=1e-1):
        w_key, b_key = random.split(key)
        return scale * random.normal(w_key, (n, m)), scale * random.normal(b_key, (n,))

    def initial_value(self) -> Tuple:
        bparam = [np.array([-0.50], dtype=np.float64)]
        layer_sizes = [4, 1]
        state = self.init_network_params(layer_sizes, random.PRNGKey(0))
        return state, bparam

    def initial_values(self) -> Tuple:
        state, bparam = self.initial_value()
        state_1 = tree_map(lambda a: a + 0.05, state)
        states = [state, state_1]
        bparam_1 = tree_map(lambda a: a + 0.05, bparam)
        bparams = [bparam, bparam_1]
        return states, bparams
