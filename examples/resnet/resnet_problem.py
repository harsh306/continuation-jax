from examples.resnet.flax_v_homotopy import ResNet18, ResNet1
from jax.experimental.optimizers import l2_norm
from cjax.utils.abstract_problem import AbstractProblem
import jax.numpy as np
from jax import random, tree_map
from collections import OrderedDict
from frozendict import frozendict
from flax.core.frozen_dict import unfreeze

model = ResNet1(num_classes=1)
input_shape = (32, 6, 6, 1)


class ResNetProblem(AbstractProblem):
    def __init__(self):
        self.HPARAMS_PATH = "hparams.json"

    @staticmethod
    def objective(params, bparam, batch) -> float:
        train_batch, targets = batch
        train_batch = np.moveaxis(train_batch, 1, -1)
        # train_batch = np.moveaxis(train_batch, -1, 0)
        # train_batch = np.moveaxis(train_batch, -1, 0)
        # train_batch = np.moveaxis(train_batch, -2, -1)

        logits = model.apply(params, train_batch, bparam=bparam[0], mutable=True)
        #logits = predict_fun(params, train_batch)
        loss = -np.sum(logits[0] * targets)
        #loss += l2_norm(params) + l2_norm(bparam)
        return loss

    @staticmethod
    def accuracy(params, bparam, batch):
        train_batch, targets = batch
        #x = np.reshape(x, (x.shape[0], -1))
        train_batch = np.moveaxis(train_batch, 1, -1)
        # train_batch = np.moveaxis(train_batch, -1, 0)
        # train_batch = np.moveaxis(train_batch, -2, -1)
        target_class = np.argmax(targets, axis=-1)
        predicted_class = np.argmax(model.apply(params, train_batch, bparam=bparam[0], mutable=True)[0], axis=-1)
        return np.mean(predicted_class == target_class)

    def initial_value(self):
        #_, init_params = init_fun(random.PRNGKey(0), (28, 28, 1, 64))
        # state = init_params
        state = model.init(random.PRNGKey(0), np.ones(input_shape, np.float32))
        state = unfreeze(state)
        bparam = [np.array([1.02], dtype=np.float32)]
        return state, bparam

    def initial_values(self):
        state, bparam = self.initial_value()
        state_1 = tree_map(lambda a: a + 0.05, state)
        states = [state, state_1]
        bparam_1 = tree_map(lambda a: a + 0.05, bparam)
        bparams = [bparam, bparam_1]
        return states, bparams
