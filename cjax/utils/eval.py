from cjax.utils.math_trees import *
from jax.experimental.optimizers import l2_norm
import pickle
from jax.flatten_util import ravel_pytree
from jax.tree_util import tree_flatten
import matplotlib.pyplot as plt
import numpy as onp
from jax.experimental import stax
from jax.nn.initializers import zeros, orthogonal, delta_orthogonal, uniform
from jax.nn import sigmoid, hard_tanh, soft_sign, relu
from jax.experimental.stax import Dense, Sigmoid, Softplus, Tanh, LogSoftmax,Relu
from cjax.optimizer.optimizer import GDOptimizer, AdamOptimizer, OptimizerCreator
import numpy.random as npr
from jax import random
from jax import jit, vmap, grad
from cjax.utils.abstract_problem import AbstractProblem
from jax.tree_util import tree_map
from cjax.utils.math_trees import *
from cjax.utils.evolve_utils import *
from cjax.continuation.states.state_variables import StateWriter
import math
import mlflow
import json
from cjax.utils.custom_nn import constant_2d, HomotopyDense, v_2d
from cjax.utils.datasets import mnist, get_mnist_data, meta_mnist, mnist_preprocess_cont
import pickle
import jax.numpy as np
from examples.torch_data import get_data

def accuracy(params, bparam, batch):
    x, targets = batch
    x = np.reshape(x, (x.shape[0], -1))
    target_class = np.argmax(targets, axis=-1)
    predicted_class = np.argmax(predict_fun(params, x, bparam=bparam[0], activation_func=relu), axis=-1)
    return np.mean(predicted_class == target_class)

def objective(params, bparam, batch) -> float:
    x, targets = batch
    x = np.reshape(x, (x.shape[0], -1))
    logits = predict_fun(params, x, bparam=bparam[0], activation_func=relu)
    loss = -np.mean(np.sum(logits * targets, axis=1))
    loss += 5e-7 * (l2_norm(params)) #+ l2_norm(bparam))
    return loss

path = '/opt/ml/mlruns/5/5a518d1762cb4b1eafc73e5cfdb11760/artifacts/output/parc_6_i3.pkl'
path_std = "/opt/ml/mlruns/5/ddb66f98e43647c2a0c65489569ba9f7/artifacts/output/params.pkl"
with open(path_std, 'rb') as fp:
    my_params = pickle.load(fp)

# init_fun, predict_fun = stax.serial(
#         HomotopyDense(out_dim=18, W_init=orthogonal(), b_init=zeros),
#         Dense(out_dim=10, W_init=orthogonal(), b_init=zeros), LogSoftmax
#     )

init_fun, predict_fun = stax.serial(
        Dense(out_dim=18), Relu,
        Dense(out_dim=10), LogSoftmax
    )


bparam = [0.019664965569972992]
bparam = [1.1531]
bparam = [1.0]
train_images, train_labels, test_images, test_labels = mnist(
            permute_train=False, resize=True, filter=False
        )

print(len(test_labels))
val_loss = objective(my_params, bparam, (test_images, test_labels))
val_acc = accuracy(my_params, bparam, (test_images, test_labels))
print(val_loss, val_acc)

# _, _, test_images, test_labels = mnist(
#             permute_train=False, resize=True, filter=True
#         )
ae_shape, ae_params = init_fun(random.PRNGKey(0), (10000, 36))
print(tree_util.tree_structure(ae_params))

print(tree_util.tree_structure(my_params[0]))
new_params = []
print(len(my_params[0][0]))
print(len(my_params[1]))
for i in my_params[0][0]:
    new_params.append(np.asarray(i))
W = np.asarray(new_params).reshape(36,18)
b = np.asarray(my_params[0][1]).reshape(18)
one  = (W,b)
print(tree_util.tree_structure(one))
new_params = []
p1 = my_params[1]
for i in my_params[2][0]:
    new_params.append(np.asarray(i))
W = np.asarray(new_params).reshape(18,10)
b = np.asarray(my_params[2][1]).reshape(10)
two  = (W,b)
print(tree_util.tree_structure(two))
three = my_params[3]

final_params = [one, p1,two, three]
print(tree_util.tree_structure(final_params))


# _, def_tree = ravel_pytree(ae_params)
# leaves, _ = ravel_pytree(params)
#
# params = def_tree(leaves)
# print(tree_util.tree_structure(params))
#
val_loss = objective(final_params, bparam, (test_images, test_labels))
val_acc = accuracy(final_params, bparam, (test_images, test_labels))
print(val_loss, val_acc)
