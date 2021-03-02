import jax.numpy as np
from jax.experimental import stax
from jax.nn.initializers import zeros
from jax.experimental.stax import Dense, Sigmoid
import numpy.random as npr
from jax import random
from cjax.utils.abstract_problem import AbstractProblem
from jax.tree_util import tree_map
from cjax.utils.custom_nn import HomotopyDropout
from cjax.utils.datasets import mnist

batch_size = 10000
input_shape = (batch_size, 36)
npr.seed(7)

def synth_batches():
    while True:
        images = npr.rand(*input_shape).astype("float32")
        yield images


# batches = synth_batches()
# inputs = next(batches)

train_images, labels, _, _ = mnist(permute_train=True)
del _
inputs = train_images[:batch_size]

del train_images

# u, s, v_t = onp.linalg.svd(inputs, full_matrices=False)
# I = np.eye(v_t.shape[-1])
# I_add = npr.normal(0.0, 0.002, size=I.shape)
# noisy_I = I + I_add


init_fun, predict_fun = stax.serial(
    HomotopyDropout(rate=0.0),
    #Dense(4, b_init=zeros),
    Dense(4, b_init=zeros), Sigmoid,
    #Dense(4, b_init=zeros),
    Dense(out_dim=input_shape[-1], b_init=zeros),
)
_, key = random.split(random.PRNGKey(0))

class DataTopologyAE(AbstractProblem):
    def __init__(self):
        self.HPARAMS_PATH = "examples/data_cont_ae/hparams.json"

    @staticmethod
    def objective(params, bparam) -> float:
        logits = predict_fun(
            params, inputs, bparam=bparam[0], rng=key
        )
        keep = random.bernoulli(key, bparam[0], inputs.shape)
        inputs_d = np.where(keep, inputs, 0)

        loss = np.mean(np.square((np.subtract(logits, inputs_d))))
        # loss += 0.1*(l2_norm(params) + l2_norm(bparam))
        return loss

    def initial_value(self):
        ae_shape, ae_params = init_fun(random.PRNGKey(0), input_shape)
        assert ae_shape == input_shape
        bparam = [np.array([0.01], dtype=np.float64)]
        return ae_params, bparam

    def initial_values(self):
        state, bparam = self.initial_value()
        state_1 = tree_map(lambda a: a + 0.084, state)
        states = [state, state_1]
        bparam_1 = tree_map(lambda a: a + 0.08, bparam)
        bparams = [bparam, bparam_1]
        return states, bparams


if __name__ == "__main__":
    problem = DataTopologyAE()
    ae_params, bparams = problem.initial_value()


    loss = problem.objective(ae_params, bparams)

    print(loss)

    # init_c = constant_2d(I)
    # print(init_c(key=0, shape=(8,8)))
