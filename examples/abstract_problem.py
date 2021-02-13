from abc import ABC, abstractmethod
from jax.tree_util import *
import jax.numpy as np
from jax import grad
from jax.experimental.optimizers import l2_norm
from jax import lax
from utils.math import pytree_dot, pytree_sub


class AbstractProblem(ABC):
    @staticmethod
    @abstractmethod
    def objective(params, bparam) -> float:
        pass

    @abstractmethod
    def initial_value(self):
        pass

    @abstractmethod
    def initial_values(self):
        pass


class ProblemWraper:
    def __init__(self, problem_object: AbstractProblem):
        self.problem_object = problem_object
        self.objective = self.problem_object.objective
        self.initial_value_func = self.problem_object.initial_value
        self.initial_values_func = self.problem_object.initial_values

    def dual_objective(
        self,
        params: list,
        bparam: list,
        lagrange_multiplier: float,
        c2: list,
        secant: list,
        delta_s=0.02,
    ) -> float:
        return np.mean(
            self.objective(params, bparam)
            + (
                lagrange_multiplier
                * self.normal_vector(params, bparam, c2, secant, delta_s)
            )
        )

    def initial_value(self):
        return self.initial_value_func()

    def initial_values(self):
        return self.initial_values_func()

    @staticmethod
    def normal_vector(
        params: list, bparams: list, secant_guess: list, secant_vec: list, delta_s
    ) -> float:
        """"""
        result = 0.0
        # params, _ = tree_flatten(params) # TODO: remove flatten
        # bparams, _ = tree_flatten(bparams)
        # state_stack = []  # TODO: reove stack list
        # state_stack.extend(params)
        # state_stack.extend(bparams)
        state_stack = dict()
        state_stack.update({"state": params})
        state_stack.update({"bparam": bparams})
        parc_vec = pytree_sub(state_stack, secant_guess)
        # parc_vec = [i - j for (i, j) in zip(state_stack, secant_guess)] # tree_multimap
        result += pytree_dot(parc_vec, secant_vec)
        # print(parc_vec, secant_vec)
        # result += np.dot(np.asarray(parc_vec).reshape(-1), np.asarray(secant_vec).reshape(-1))
        return result - delta_s

    def objective_grad(self, params, bparam):  # TODO: JIT?
        grad_J = grad(self.objective, [0, 1])
        params_grad, bparam_grad = grad_J(params, bparam)
        result = l2_norm(params_grad) + l2_norm(bparam_grad)
        return result

    def dual_objective_grad(
        self,
        params: list,
        bparam: list,
        lagrange_multiplier: float,
        c2: list,
        secant: list,
        delta_s=0.02,
    ) -> float:
        return np.mean(
            self.objective_grad(params, bparam)
            + (
                lagrange_multiplier
                * self.normal_vector(params, bparam, c2, secant, delta_s)
            )
        )

    @staticmethod
    def reparm_bijection(params):
        b = 2.0  # greater equal to  0
        a = -0.6
        params = np.power(np.abs(params + b), a) * params
        return params
