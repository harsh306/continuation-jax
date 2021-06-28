

from cjax.utils.abstract_problem import AbstractProblem
import jax.numpy as np
from jax import random
from jax.experimental.optimizers import l2_norm
from jax.tree_util import *
import resnet_flax_1
import mlflow
from cjax.utils.datasets import get_mnist_data, meta_mnist, mnist
from jax import jit, grad
from cjax.continuation.states.state_variables import StateWriter
from cjax.optimizer.optimizer import OptimizerCreator
import json
import pickle
import math
from cjax.utils.evolve_utils import running_mean, exp_decay


def objective(params, batch):
    train_batch, targets = batch
    print(type(targets))
    print(train_batch.shape)
    train_batch = np.moveaxis(train_batch, 1, -1)
    # train_batch = np.moveaxis(train_batch, -1, 0)
    # train_batch = np.moveaxis(train_batch, -1, 0)
    # train_batch = np.moveaxis(train_batch, -2, -1)
    print(35 * "#")
    print(params)
    print(train_batch.shape)
    logits = resnet_model_def.apply({'params': params}, train_batch, mutable=True)
    print(logits)
    # logits = predict_fun(params, train_batch)
    loss = -np.sum(logits * targets)
    #loss += l2_norm(params) + l2_norm(bparam)
    return loss

def accuracy(params, batch):
    train_batch, targets = batch
    #x = np.reshape(x, (x.shape[0], -1))
    train_batch = np.moveaxis(train_batch, 1, -1)
    # train_batch = np.moveaxis(train_batch, -1, 0)
    # train_batch = np.moveaxis(train_batch, -2, -1)

    target_class = np.argmax(targets, axis=-1)
    predicted_class = resnet_model_def.apply({'params': params["params"]}, train_batch)
    #predicted_class = np.argmax(predict_fun(params, train_batch, bparam=bparam[0], activation_func=relu), axis=-1)
    return np.mean(predicted_class == target_class)


if __name__ == '__main__':
    img_size = 6
    channels = 1

    # resnet_model_def.apply()
    input_shape = (1, img_size, img_size, channels)

    resnet_model_def = resnet_flax_1.ResNet18(num_classes=10)
    resnet_params = resnet_model_def.init(random.PRNGKey(0), np.ones(input_shape, np.float32))
    data_loader = iter(get_mnist_data(batch_size=32, resize=True, filter=False))
    one_batch = next(data_loader)
    print(one_batch[0].shape)
    num_batches = meta_mnist(batch_size=32, filter=False)["num_batches"]
    print(f"num of bathces: {num_batches}")
    print(resnet_params.keys())

    compute_grad_fn = jit(grad(objective, [0]))
    opt = OptimizerCreator("gradient-descent", learning_rate=0.1).get_optimizer()
    ma_loss = []
    for epoch in range(100):
        for b_j in range(num_batches):
            batch = next(data_loader)
            ae_grads = compute_grad_fn(resnet_params, batch)
            print(ae_grads)
            resnet_params.update({"params": opt.update_params(resnet_params["params"], ae_grads[0], step_index=epoch)})
            # bparam = opt.update_params(bparam, b_grads, step_index=epoch)
            loss = objective(resnet_params, batch)
            ma_loss.append(loss)
            print(f"loss:{loss}  norm:{l2_norm(ae_grads)}")
        opt.lr = exp_decay(epoch, 0.1)