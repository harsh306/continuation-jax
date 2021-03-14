import jax.numpy as np
from jax.experimental import stax
from jax.nn.initializers import zeros
from jax.experimental.stax import (
    Dense,
    Sigmoid,
    Conv,
    BatchNorm,
    GeneralConv,
    Flatten,
    LogSoftmax,
    Relu,
)
import numpy.random as npr
from jax import random
from cjax.utils.abstract_problem import AbstractProblem
from jax.tree_util import tree_map
from cjax.utils.custom_nn import HomotopyDropout
from cjax.optimizer.optimizer import GDOptimizer, AdamOptimizer
from jax.experimental.optimizers import l2_norm
from cjax.utils.datasets import mnist
from jax import jit, grad
import json
import math
from cjax.utils.datasets import get_mnist_data, meta_mnist
from cjax.utils.math_trees import pytree_element_add
from cjax.utils.evolve_utils import running_mean

data_size = 40000
input_shape = (data_size, 36)
npr.seed(7)
num_classes = 10


def synth_batches():
    while True:
        images = npr.rand(*input_shape).astype("float32")
        yield images


# batches = synth_batches()
# inputs = next(batches)

# train_images, labels, _, _ = mnist(permute_train=True, resize=True)
# del _
# inputs = train_images[:data_size]
#
# del train_images

# u, s, v_t = onp.linalg.svd(inputs, full_matrices=False)
# I = np.eye(v_t.shape[-1])
# I_add = npr.normal(0.0, 0.002, size=I.shape)
# noisy_I = I + I_add


init_fun, conv_net = stax.serial(
    Conv(32, (5, 5), (2, 2), padding="SAME"),
    BatchNorm(),
    Relu,
    Conv(10, (3, 3), (2, 2), padding="SAME"),
    Relu,
    Flatten,
    Dense(num_classes),
    LogSoftmax,
)
_, key = random.split(random.PRNGKey(0))


class DataTopologyAE(AbstractProblem):
    def __init__(self):
        self.HPARAMS_PATH = "hparams.json"

    @staticmethod
    def objective(params, bparam, batch) -> float:
        x, _ = batch
        x = np.reshape(x, (x.shape[0], -1))
        logits = predict_fun(params, x, bparam=bparam[0], rng=key)
        keep = random.bernoulli(key, bparam[0], x.shape)
        inputs_d = np.where(keep, x, 0)

        loss = np.mean(np.square((np.subtract(logits, inputs_d))))
        loss += 1e-8 * (l2_norm(params) + 1 / l2_norm(bparam))
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
        bparam_1 = tree_map(lambda a: a + 0.03, bparam)
        bparams = [bparam, bparam_1]
        return states, bparams


def exp_decay(epoch, initial_lrate):
    k = 0.02
    lrate = initial_lrate * np.exp(-k * epoch)
    return lrate


if __name__ == "__main__":
    problem = DataTopologyAE()
    ae_params, bparam = problem.initial_value()
    bparam = pytree_element_add(bparam, 0.99)
    data_loader = iter(get_mnist_data(batch_size=25000, resize=True))
    num_batches = meta_mnist(batch_size=25000)["num_batches"]
    print(f"num of bathces: {num_batches}")
    compute_grad_fn = jit(grad(problem.objective, [0]))

    with open(problem.HPARAMS_PATH, "r") as hfile:
        hparams = json.load(hfile)
    opt = AdamOptimizer(learning_rate=hparams["descent_lr"])
    ma_loss = []
    for epoch in range(500):
        for b_j in range(num_batches):
            batch = next(data_loader)
            grads = compute_grad_fn(ae_params, bparam, batch)
            ae_params = opt.update_params(ae_params, grads[0], step_index=epoch)
            loss = problem.objective(ae_params, bparam, batch)
            ma_loss.append(loss)
            print(f"loss:{loss}  norm:{l2_norm(grads)}")
        opt.lr = exp_decay(epoch, hparams["descent_lr"])
        if len(ma_loss) > 40:
            loss_check = running_mean(ma_loss, 30)
            if math.isclose(
                loss_check[-1], loss_check[-2], abs_tol=hparams["loss_tol"]
            ):
                print(f"stopping at {epoch}")
                break

    train_images, train_labels, test_images, test_labels = mnist(
        permute_train=False, resize=True
    )
    val_loss = problem.objective(ae_params, bparam, (test_images, test_labels))
    print(f"val loss: {val_loss}")

    # init_c = constant_2d(I)
    # print(init_c(key=0, shape=(8,8)))
