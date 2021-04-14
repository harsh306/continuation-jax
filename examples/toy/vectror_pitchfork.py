import jax.numpy as np
from jax.config import config
from cjax.utils.abstract_problem import AbstractProblem
from jax.tree_util import *
from jax import hessian, grad, jit
from jax.flatten_util import ravel_pytree
config.update("jax_debug_nans", True)


class QuadraticProblem(AbstractProblem):
    def __init__(self):
        self.inputs = None
        self.outputs = None
        self.HPARAMS_PATH = "hparams.json"

    @staticmethod
    def objective(params: list, bparam: list, batch_input) -> float:
        result = 0.0
        for w1 in params:
            result += np.mean(np.divide(np.power(w1, 2), 2.0) * np.square(bparam[0]))
        return result

    def initial_values(self) -> Tuple:
        """
        PyTreeDef(list, [PyTreeDef(tuple, [*,*])])
        :return:
        """
        states = [
            [np.array([0.05])],
            [np.array([0.03])],
        ]
        # states = [
        #     [np.array([-1.734])],
        #     [np.array([-1.632])],
        # ]
        bparams = [[np.array([3.1])], [np.array([2.8])]]

        return states, bparams

    def initial_value(self) -> Tuple:
        """
        PyTreeDef(list, [PyTreeDef(tuple, [*,*])])
        :return:
        """
        state = [np.array([0.04])]
        bparam = [np.array([3.0])]
        return state, bparam


class SigmoidFold(AbstractProblem):
    def __init__(self):
        self.HPARAMS_PATH = "hparams.json"

    @staticmethod
    def objective(params, bparam, batch_input) -> float:
        targets = np.multiply(0.5, params[0])
        logits = np.divide(1, 1 + np.exp(-(np.multiply(5.0, params[0]) + bparam[0])))
        loss = np.mean(np.square(np.subtract(logits, targets)))
        return loss

    def initial_value(self):
        state = [np.array([2.1])]
        bparam = [np.array([1.0])]
        return state, bparam

    def initial_values(self):
        states = [
            [np.array([0.61])],
            [np.array([0.62])],
        ]
        bparams = [[np.array([-2.71])], [np.array([-2.68])]]

        return states, bparams


class PitchForkProblem(AbstractProblem):
    def __init__(self):
        self.inputs = None
        self.outputs = None
        self.HPARAMS_PATH = "hparams.json"

    @staticmethod
    def objective(params: list, bparam: list, batch_input) -> float:
        result = 25.0
        for w1 in params:
            result += np.mean(
                np.divide(np.power(w1, 4), 4.0)
                + bparam[0] * np.divide(np.power(w1, 2), 2.0)
            )
        return result

    def initial_values(self) -> Tuple:
        """
        PyTreeDef(list, [PyTreeDef(tuple, [*,*])])
        :return:
        """
        states = [
            [np.array([-1.734])],
            [np.array([-1.73])],
        ]
        # states = [
        #     [np.array([-1.734])],
        #     [np.array([-1.632])],
        # ]
        bparams = [[np.array([-3.1])], [np.array([-3.0])]]

        return states, bparams

    def initial_value(self) -> Tuple:
        """
        PyTreeDef(list, [PyTreeDef(tuple, [*,*])])
        :return:
        """
        state = [np.array([-1.734])]
        bparam = [np.array([-3.0])]
        return state, bparam


class VectorPitchFork(AbstractProblem):
    # TODO: check self with jax grad
    # TODO: flatten vs unflatten grad at various levels

    def __init__(self):
        """
        I can't pass param and bparam here because of grad tractibility issue
        """
        self.inputs = 0.0
        self.outputs = 0.0
        self.HPARAMS_PATH = "examples/toy/hparams.json"

    @staticmethod
    def objective(state, bparam, batch_input):
        """
        Computes scalar objective.
        :param params: pytree PyTreeDef(list, [PyTreeDef(tuple, [*,*])])
        :param bparam: pytree *
        :param inputs: pytree *
        :param outputs: pytree *
        :return: pytree (scalar) *
        """
        result = 0.0

        for (w, b) in state:
            result += np.mean(
                np.sum(
                    np.divide(np.power(w, 4), 4.0)
                    + bparam[0] * np.divide(np.power(w, 2), 2.0)
                )
                + np.sum(
                    np.divide(np.power(b, 4), 4.0)
                    + bparam[0] * np.divide(np.power(b, 2), 2.0)
                )
            )
        return result

    def initial_value(self) -> Tuple:
        """
        PyTreeDef(list, [PyTreeDef(tuple, [*,*])])
        :return:
        """
        state = [(np.array([-0.234]), np.array([-1.73205080757]))]
        bparam = [np.array([-3.0])]
        return state, bparam

    def initial_values(self):
        """
        PyTreeDef(list, [PyTreeDef(tuple, [*,*])])
        :return:
        """
        # states = [
        #     [(np.array([-1.734]), np.array([-1.734]))],
        #     [(np.array([-1.732]), np.array([-1.732]))],
        # ]

        # bparams = [[np.array([-3.0])], [np.array([-2.9])]]
        # states = [
        #     [np.array([-1.734])],
        #     [np.array([-1.632])],
        # ]

        states = [
            [(np.array([0.05]), np.array([0.05]))],
            [(np.array([0.02]), np.array([0.02]))],
        ]

        bparams = [[np.array([3.2])], [np.array([3.0])]]

        return states, bparams


if __name__ == "__main__":
    s = VectorPitchFork()
    param, bparam = s.initial_value()
    print(param, bparam)
    g = jit(grad(s.objective, argnums=[0]))(param, bparam, 0.0)
    print(g)
    dg2 = hessian(s.objective, argnums=[0])(param, bparam, 0.0)
    print(dg2)
    mtree, _ = ravel_pytree(dg2)
    eigen = np.linalg.eigvals(mtree.reshape(len(param),len(param)))
    print(eigen)