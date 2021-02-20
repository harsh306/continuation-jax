import jax.numpy as np
import numpy as onp
from jax.experimental import stax
from jax.nn.initializers import zeros, ones
from jax.nn import sigmoid
from jax.experimental.optimizers import l2_norm
from jax.experimental.stax import Dense, elementwise, Identity
import numpy.random as npr
from jax import random
from examples.abstract_problem import AbstractProblem
from jax.tree_util import tree_map

batch_size = 8
input_shape = (8, 8)
step_size = 0.1
num_steps = 10
code_dim = 1


def homotopy_activation(x, alpha=1.0, activation_func=None):
    return alpha * x + (1 - alpha) * activation_func(x)


def constant_init(m, dtype=np.float32):
    """
    Constant weight initializer
    Args:
        m: constant matrix
        dtype: matrix dtype

    Returns:
        init function
    """
    def init(key, shape, dtype=dtype):
        return m[:shape[0], :shape[1]]
    return init

def constant(matrix, dtype=np.float32):
    def init(key, shape, dtype=dtype):
        if matrix.shape != shape:
            raise ValueError(f"input array of shape {matrix.shape} is not equal to the weight array of shape {shape}")
        return matrix
    return init


def synth_batches():
    while True:
        images = npr.rand(*input_shape).astype("float32")
        yield images


batches = synth_batches()
inputs = next(batches)
u, s, v_t = onp.linalg.svd(inputs, full_matrices=False)
I = np.eye(v_t.shape[-1])
I_add =npr.normal(0.0, 0.008, size=I.shape)
noisy_I = I + I_add

def HomotopyDense(out_dim, W_init=zeros, b_init=zeros):
    """Layer constructor function for a dense (fully-connected) layer."""

    def init_fun(rng, input_shape):
        output_shape = input_shape[:-1] + (out_dim,)
        k1, k2 = random.split(rng)
        W, b = W_init(k1, (input_shape[-1], out_dim)), b_init(k2, (out_dim,))
        return output_shape, (W, b)

    def apply_fun(params, inputs, **kwargs):
        W, b = params
        dense = np.dot(inputs, W) + b
        return homotopy_activation(
            dense, alpha=kwargs.get("bparam"), activation_func=kwargs.get("activation_func")
        )
    return init_fun, apply_fun


init_fun, predict_fun = stax.serial(
    HomotopyDense(out_dim=4, W_init=constant_init(v_t.T), b_init=zeros),
    HomotopyDense(out_dim=2, W_init=constant_init(noisy_I), b_init=zeros),
    HomotopyDense(out_dim=4, W_init=constant_init(noisy_I), b_init=zeros),
    Dense(out_dim=input_shape[-1], W_init=constant_init(v_t), b_init=zeros),
)


class PCATopologyAE(AbstractProblem):
    def __init__(self):
        pass

    @staticmethod
    def objective(params, bparam) -> float:
        logits = predict_fun(params, inputs, bparam=bparam[0], activation_func=sigmoid)
        loss = np.mean((np.subtract(logits, inputs)))
        #loss += l2_norm(params) + l2_norm(bparam)
        return loss

    def initial_value(self):
        ae_shape, ae_params = init_fun(random.PRNGKey(0), input_shape)
        assert ae_shape == input_shape
        bparam = [np.array([0.00], dtype=np.float64)]
        return ae_params, bparam

    def initial_values(self):
        state, bparam = self.initial_value()
        state_1 = tree_map(lambda a: a + 0.005, state)
        states = [state, state_1]
        bparam_1 = tree_map(lambda a: a + 0.05, bparam)
        bparams = [bparam, bparam_1]
        return states, bparams


if __name__ == "__main__":
    problem = PCATopologyAE()
    ae_params, bparams = problem.initial_value()
    loss = problem.objective(ae_params, bparams)
    print(loss)

    init_c = constant(matrix=I)
    print(init_c(key=0, shape=(8,8)))


