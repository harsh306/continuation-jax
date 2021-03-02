from jax.nn.initializers import glorot_uniform, normal
from jax.nn import sigmoid
from jax.experimental.optimizers import l2_norm
from jax.experimental.stax import Sigmoid
from jax import random
import jax.numpy as np
from jax.experimental import stax
import numpy.random as npr
from cjax.utils.custom_nn import HomotopyDense


def generate_data_01():
    batch_size = 8
    input_shape = (batch_size, 4)

    def synth_batches():
        while True:
            images = npr.rand(*input_shape).astype("float32")
            yield images

    batches = synth_batches()
    inputs = next(batches)

    init_func, predict_func = stax.serial(
        HomotopyDense(out_dim=4, W_init=glorot_uniform(), b_init=normal()),
        HomotopyDense(out_dim=1, W_init=glorot_uniform(), b_init=normal()),
        Sigmoid,
    )

    ae_shape, ae_params = init_func(random.PRNGKey(0), input_shape)
    # assert ae_shape == input_shape
    bparam = [np.array([0.0], dtype=np.float64)]
    logits = predict_func(ae_params, inputs, bparam=bparam[0], activation_func=sigmoid)
    loss = np.mean((np.subtract(logits, logits))) + l2_norm(ae_params) + l2_norm(bparam)

    return inputs, logits, ae_params, bparam, init_func, predict_func


if __name__ == "__main__":
    x, y, param, bparam, _, _ = generate_data_01()
    print(x.shape, y.shape)
    print(y)
