from jax.nn.initializers import zeros
from jax.nn import sigmoid
from jax.experimental.optimizers import l2_norm
from jax.experimental.stax import Dense, elementwise, Identity
from jax import random
import jax.numpy as np


def constant_2d(m, dtype=np.float32):
    """
    Constant weight initializer
    Args:
        m: constant matrix
        dtype: matrix dtype

    Returns:
        init function
    """

    def init(key, shape, dtype=dtype):
        return m[: shape[0], : shape[1]]

    return init


def HomotopyDropout(rate, mode="train"):
    """Layer construction function for a dropout layer with given rate."""

    def init_fun(rng, input_shape):
        return input_shape, ()

    def apply_fun(params, inputs, **kwargs):
        rng = kwargs.get("rng", None)
        if rng is None:
            msg = (
                "Dropout layer requires apply_fun to be called with a PRNG key "
                "argument. That is, instead of `apply_fun(params, inputs)`, call "
                "it like `apply_fun(params, inputs, rng)` where `rng` is a "
                "jax.random.PRNGKey value."
            )
            raise ValueError(msg)
        if mode == "train":
            hrate = kwargs.get("bparam") + rate
            _, key = random.split(random.PRNGKey(0))
            keep = random.bernoulli(key, hrate, inputs.shape)
            return np.where(keep, inputs, 0)
        else:
            return inputs

    return init_fun, apply_fun


def v_2d(m, dtype=np.float32):
    """
    Constant weight initializer
    Args:
        m: constant matrix
        dtype: matrix dtype

    Returns:
        init function
    """

    def init(key, shape, dtype=dtype):
        return m[: shape[0], : shape[1]]

    return init


def constant(matrix, dtype=np.float32):
    def init(key, shape, dtype=dtype):
        if matrix.shape != shape:
            raise ValueError(
                f"input array of shape {matrix.shape} is not equal to weight array of shape {shape}"
            )
        return matrix

    return init


def homotopy_activation(x, alpha=1.0, activation_func=None):
    """

    Args:
        x:
        alpha:
        activation_func:

    Returns:

    """
    if activation_func:
        return alpha * x + (1 - alpha) * activation_func(x)
    else:
        return x


def HomotopyDense(out_dim, W_init=zeros, b_init=zeros):
    """
    Layer constructor function for a dense (fully-connected) layer.
    Args:
        out_dim:
        W_init:
        b_init:

    Returns:

    """

    def init_fun(rng, input_shape):
        output_shape = input_shape[:-1] + (out_dim,)
        k1, k2 = random.split(rng)
        W, b = W_init(k1, (input_shape[-1], out_dim)), b_init(k2, (out_dim,))
        return output_shape, (W, b)

    def apply_fun(params, inputs, **kwargs):
        bparam = kwargs.get("bparam")
        W, b = params
        dense = np.dot(inputs, W) + b
        return homotopy_activation(
            dense, alpha=bparam, activation_func=kwargs.get("activation_func")
        )

    return init_fun, apply_fun


def LambdaIdentity():
    """Layer construction function for an identity layer."""
    init_fun = lambda rng, input_shape: (input_shape, ())
    apply_fun = lambda params, inputs, **kwargs: inputs * kwargs.get("bparam")
    return init_fun, apply_fun


LambdaIdentity = LambdaIdentity()
