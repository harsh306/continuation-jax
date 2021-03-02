import numpy.random as npr

from jax.experimental import stax
from jax.experimental.stax import (
    AvgPool,
    BatchNorm,
    Conv,
    Dense,
    FanInSum,
    FanOut,
    Flatten,
    GeneralConv,
    Identity,
    MaxPool,
    Relu,
    LogSoftmax,
)


from cjax.utils.abstract_problem import AbstractProblem
import jax.numpy as np
from jax import random
from jax.experimental.optimizers import l2_norm
from jax.tree_util import *

batch_size = 8
num_classes = 1001
input_shape = (224, 224, 3, batch_size)
step_size = 0.1
num_steps = 10

# ResNet blocks compose other layers


def LambdaIdentity():
    """Layer construction function for an identity layer."""
    init_fun = lambda rng, input_shape: (input_shape, ())
    apply_fun = lambda params, inputs, **kwargs: inputs * kwargs.get("bparams")
    return init_fun, apply_fun


LambdaIdentity = LambdaIdentity()


def ConvBlock(kernel_size, filters, strides=(2, 2)):
    ks = kernel_size
    filters1, filters2, filters3 = filters
    Main = stax.serial(
        Conv(filters1, (1, 1), strides),
        BatchNorm(),
        Relu,
        Conv(filters2, (ks, ks), padding="SAME"),
        BatchNorm(),
        Relu,
        Conv(filters3, (1, 1)),
        BatchNorm(),
    )
    Shortcut = stax.serial(Conv(filters3, (1, 1), strides), BatchNorm())
    return stax.serial(FanOut(2), stax.parallel(Main, Shortcut), FanInSum, Relu)


def IdentityBlock(kernel_size, filters):
    ks = kernel_size
    filters1, filters2 = filters

    def make_main(input_shape):
        # the number of output channels depends on the number of input channels
        return stax.serial(
            Conv(filters1, (1, 1)),
            BatchNorm(),
            Relu,
            Conv(filters2, (ks, ks), padding="SAME"),
            BatchNorm(),
            Relu,
            Conv(input_shape[3], (1, 1)),
            BatchNorm(),
        )

    Main = stax.shape_dependent(make_main)
    return stax.serial(FanOut(2), stax.parallel(Main, Identity), FanInSum, Relu)


# ResNet architectures compose layers and ResNet blocks


def ResNet50(num_classes):
    return stax.serial(
        GeneralConv(("HWCN", "OIHW", "NHWC"), 64, (7, 7), (2, 2), "SAME"),
        BatchNorm(),
        Relu,
        MaxPool((3, 3), strides=(2, 2)),
        ConvBlock(3, [64, 64, 256], strides=(1, 1)),
        IdentityBlock(3, [64, 64]),
        IdentityBlock(3, [64, 64]),
        ConvBlock(3, [128, 128, 512]),
        IdentityBlock(3, [128, 128]),
        IdentityBlock(3, [128, 128]),
        IdentityBlock(3, [128, 128]),
        ConvBlock(3, [256, 256, 1024]),
        IdentityBlock(3, [256, 256]),
        IdentityBlock(3, [256, 256]),
        IdentityBlock(3, [256, 256]),
        IdentityBlock(3, [256, 256]),
        IdentityBlock(3, [256, 256]),
        ConvBlock(3, [512, 512, 2048]),
        IdentityBlock(3, [512, 512]),
        IdentityBlock(3, [512, 512]),
        AvgPool((7, 7)),
        Flatten,
        Dense(num_classes),
        LogSoftmax,
    )


def ResNet(num_classes):
    return stax.serial(
        GeneralConv(("HWCN", "OIHW", "NHWC"), 64, (7, 7), (2, 2), "SAME"),
        BatchNorm(),
        Relu,
        MaxPool((3, 3), strides=(2, 2)),
        ConvBlock(3, [4, 4, 4], strides=(1, 1)),
        IdentityBlock(3, [4, 4]),
        AvgPool((3, 3)),
        Flatten,
        Dense(num_classes),
        LogSoftmax,
    )


def synth_batches():
    rng = npr.RandomState(0)
    while True:
        images = rng.rand(*input_shape).astype("float32")
        labels = rng.randint(num_classes, size=(batch_size, 1))
        onehot_labels = labels == np.arange(num_classes)
        yield images, onehot_labels


batches = synth_batches()
inputs, outputs = next(batches)
init_fun, predict_fun = ResNet(num_classes)


class ResNet50Network(AbstractProblem):
    def __init__(self):
        pass

    @staticmethod
    def objective(params, bparam) -> float:
        logits = predict_fun(params, inputs)
        loss = -jnp.sum(logits * outputs)
        loss += l2_norm(params) + l2_norm(bparam)
        return loss

    def initial_value(self):
        _, init_params = init_fun(random.PRNGKey(0), input_shape)
        bparam = [np.array([-0.50], dtype=np.float64)]
        state = init_params
        return state, bparam

    def initial_values(self):
        state, bparam = self.initial_value()
        state_1 = tree_map(lambda a: a + 0.05, state)
        states = [state, state_1]
        bparam_1 = tree_map(lambda a: a + 0.05, bparam)
        bparams = [bparam, bparam_1]
        return states, bparams
