from cjax.utils.abstract_problem import AbstractProblem
import jax.numpy as np
from jax import random
from jax.experimental.optimizers import l2_norm
from jax.tree_util import *
from flax import linen as nn  # The Linen API
import jax
import numpy.random as npr
from cjax.utils import datasets

#
# num_classes = 10
# inputs = random.normal(random.PRNGKey(1), (1, 10, 10))
# outputs = np.ones(shape=(num_classes, 10))

batch_size = 20
train_images, train_labels, test_images, test_labels = datasets.mnist()
num_train = train_images.shape[0]
num_complete_batches, leftover = divmod(num_train, batch_size)
num_batches = num_complete_batches + bool(leftover)


def data_stream():
    rng = npr.RandomState(0)
    while True:
        perm = rng.permutation(num_train)
        for i in range(num_batches):
            batch_idx = perm[i * batch_size : (i + 1) * batch_size]
            yield train_images[batch_idx], train_labels[batch_idx]


batches = data_stream()
inputs, outputs = next(batches)


class CNN(nn.Module):
    """Flax CNN Module"""

    @nn.compact
    def __call__(self, x):
        x = nn.Conv(features=32, kernel_size=(3, 3))(x)
        x = nn.relu(x)
        x = nn.avg_pool(x, window_shape=(2, 2), strides=(2, 2))
        x = nn.Conv(features=64, kernel_size=(3, 3))(x)
        x = nn.relu(x)
        x = nn.avg_pool(x, window_shape=(2, 2), strides=(2, 2))
        x = x.reshape((x.shape[0], -1))  # Flatten
        x = nn.Dense(features=256)(x)
        x = nn.relu(x)
        x = nn.Dense(features=10)(x)  # There are 10 classes in MNIST
        x = nn.log_softmax(x)
        return x


class ConvNeuralNetwork(AbstractProblem):
    def __init__(self):
        pass

    @staticmethod
    def objective(params, bparam) -> float:
        def cross_entropy_loss(logits, labels):
            one_hot_labels = jax.nn.one_hot(labels, num_classes=10)
            return -np.mean(np.sum(one_hot_labels * logits, axis=-1))

        logits = CNN().apply({"params": params[0]}, inputs)
        loss = cross_entropy_loss(logits, outputs)
        loss += l2_norm(params) + l2_norm(bparam)
        # vectorization of mini-batch of data.
        # #3rd argument's 0th-axis is vmaped. --> inputs(10)
        # batched_predict = vmap(neural_net_predict, in_axes=(None, None, 0))
        return loss

    @staticmethod
    def init_network_params(key):
        init_shape = np.ones((1, 28, 28, 1), np.float32)
        initial_params = CNN().init(key, init_shape)["params"]
        return initial_params

    def initial_value(self) -> Tuple:
        bparam = [np.array([-0.50], dtype=np.float64)]
        state = [self.init_network_params(random.PRNGKey(0))]
        return state, bparam

    def initial_values(self) -> Tuple:
        state, bparam = self.initial_value()
        state_1 = tree_map(lambda a: a + 0.05, state)
        states = [state, state_1]
        bparam_1 = tree_map(lambda a: a + 0.05, bparam)
        bparams = [bparam, bparam_1]
        return states, bparams
