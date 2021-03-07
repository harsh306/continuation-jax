import jax.numpy as np
import numpy as onp
from jax.experimental import stax
from jax.nn.initializers import zeros
from jax.nn import sigmoid
from jax.experimental.stax import Dense, Sigmoid
import numpy.random as npr
from jax import random
from jax import jit, vmap
from cjax.utils.abstract_problem import AbstractProblem
from jax.tree_util import tree_map
from cjax.utils.custom_nn import constant_2d, HomotopyDense, v_2d
from cjax.utils.datasets import mnist, get_mnist_data
from examples.torch_data import get_data
batch_size = 40000
input_shape = (batch_size, 36)
step_size = 0.1
num_steps = 10
code_dim = 1
npr.seed(7)


train_images, labels, _, _ = mnist(permute_train=True)
del _
inputs = train_images[:batch_size]
del train_images

u, s, v_t = onp.linalg.svd(inputs, full_matrices=False)
I = np.eye(v_t.shape[-1])
I_add = npr.normal(0.0, 0.002, size=I.shape)
noisy_I = I + I_add

init_fun, predict_fun = stax.serial(
    HomotopyDense(out_dim=4, W_init=v_2d(v_t.T), b_init=zeros),
    HomotopyDense(out_dim=2, W_init=constant_2d(noisy_I), b_init=zeros),
    HomotopyDense(out_dim=4, W_init=constant_2d(noisy_I), b_init=zeros),
    Dense(out_dim=input_shape[-1], W_init=v_2d(v_t), b_init=zeros),
)

#
# init_fun, predict_fun = stax.serial(
#     Dense(out_dim=4), Sigmoid,
#     Dense(out_dim=2), Sigmoid,
#     Dense(out_dim=4), Sigmoid,
#     Dense(out_dim=input_shape[-1]),
# )


class PCATopologyAE(AbstractProblem):
    def __init__(self):
        self.HPARAMS_PATH = "hparams.json"

    @staticmethod
    def objective(params, bparam, batch) -> float:
        x, _ = batch
        x = np.reshape(x, (x.shape[0], -1))
        logits = predict_fun(
            params, x, bparam=bparam[0], activation_func=sigmoid
        )
        loss = np.mean(np.square((np.subtract(logits, x))))
        #loss += 0.1 * (l2_norm(params) + l2_norm(bparam))
        return loss

    def initial_value(self):
        ae_shape, ae_params = init_fun(random.PRNGKey(0), input_shape)
        # print(ae_shape)
        assert ae_shape == input_shape
        bparam = [np.array([0.01], dtype=np.float64)]
        return ae_params, bparam

    def initial_values(self):
        state_0, bparam_0 = self.initial_value()
        state_1 = tree_map(lambda a: a - 0.08, state_0)
        states = [state_0, state_1]
        bparam_1 = tree_map(lambda a: a + 0.08, bparam_0)
        bparams = [bparam_0, bparam_1]
        return states, bparams


if __name__ == "__main__":
    problem = PCATopologyAE()
    ae_params, bparams = problem.initial_value()
    td = get_mnist_data(batch_size=5, resize=True)
    loss = jit(problem.objective)
    for i in range(10):
        lv = loss(ae_params, bparams, next(iter(td)))
        print(lv)

