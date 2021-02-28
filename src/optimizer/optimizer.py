from abc import ABC, abstractmethod

from jax.experimental.optimizers import adam, sgd
from jax.experimental.optimizers import optimizer, make_schedule


@optimizer
def gradient_ascent(step_size):
    """Construct optimizer triple for stochastic gradient descent.

    Args:
      step_size: positive scalar, or a callable representing a step size schedule
        that maps the iteration index to positive scalar.

    Returns:
      An (init_fun, update_fun, get_params) triple.
    """
    step_size = make_schedule(step_size)

    def init(x0):
        return x0

    def update(i, g, x):
        return x + step_size(i) * g

    def get_params(x):
        return x

    return init, update, get_params


class Optimizer:
    """Abstract Optimizer to be inherited by developer for any new optimizer."""
    def __init__(self, lr):
        self._lr =lr

    @property
    def lr(self):
        return self._lr

    @lr.setter
    def lr(self, lr):
        self._lr = lr

    def update_params(
        self, params: list, grad_params: list, step_index: int = 0
    ) -> list:
        pass


class GDOptimizer(Optimizer):
    def __init__(self, learning_rate):
        super().__init__(learning_rate)
        self.opt_init, self.opt_update, self.get_params = sgd(step_size=self.lr)

    def update_params(
        self, params: list, grad_params: list, step_index: int = 0
    ) -> list:
        opt_state = self.opt_init(params)
        params = self.get_params(self.opt_update(step_index, grad_params, opt_state))
        return params


class _GDOptimizer(Optimizer):
    def __init__(self, learning_rate):
        super().__init__(learning_rate)

    def update_params(
        self, params: list, grad_params: list, step_index: int = 0
    ) -> list:
        for (w, dw) in zip(params, grad_params):
            print(w, dw)
        return [w - self.lr * dw for (w, dw) in zip(params, grad_params)]


class GAOptimizer(Optimizer):
    def __init__(self, learning_rate):
        super().__init__(learning_rate)
        self.opt_init, self.opt_update, self.get_params = gradient_ascent(
            step_size=self.lr
        )

    def update_params(
        self, params: list, grad_params: list, step_index: int = 0
    ) -> list:
        opt_state = self.opt_init(params)
        params = self.get_params(self.opt_update(step_index, grad_params, opt_state))
        return params


class AdamOptimizer(Optimizer):
    def __init__(self, learning_rate):
        super().__init__(learning_rate)
        self.opt_init, self.opt_update, self.get_params = adam(step_size=self.lr)

    def update_params(
        self, params: list, grad_params: list, step_index: int = 0
    ) -> list:
        opt_state = self.opt_init(params)
        params = self.get_params(self.opt_update(step_index, grad_params, opt_state))
        return params


class OptimizerCreator:
    def __init__(self, opt_string, learning_rate):
        self.learning_rate = learning_rate
        self._opt_string = opt_string

    def get_optimizer(self) -> Optimizer:
        if self._opt_string == "gradient-descent":
            return GDOptimizer(self.learning_rate)
        elif self._opt_string == "adam":
            return AdamOptimizer(self.learning_rate)
        elif self._opt_string == "gradient-ascent":
            return GAOptimizer(self.learning_rate)
        else:
            print(f"Optimizer not implemented: {self._opt_string}")
            raise NotImplementedError

if __name__ == '__main__':
    opt = GDOptimizer(0.5)
    print(opt.lr)