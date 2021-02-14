import matplotlib.pyplot as plt
import jax.numpy as np
import json
import jsonlines
from jax.experimental.optimizers import l2_norm
from jax.tree_util import *
from jax import flatten_util
from utils.math_trees import *


def pick_array(data: list, start:int = 0, end:int = 1):
    param = []
    bparam = []
    for k in range(len(data)):
        param.append(data[k][0][start:end])
        bparam.append(data[k][1][0])
    return param, bparam


def coine_data_transform(data):
    params = []
    bparam = []
    for k in range(len(data)-1):
        params.append(pytree_dot(data[k][0], data[k+1][0]))
        bparam.append(data[k][1][0])
    return params, bparam


def norm_data_transform(data):
    params = []
    bparam = []
    for k in range(len(data)):
        params.append(l2_norm(data[k][0]))
        bparam.append(data[k][1][0])
    return params, bparam


# TODO
def export():
    pass

# TODO
def perturbplot():
    pass

if __name__ == '__main__':

    path = '/opt/ml/output/pca_ae/version.json'
    data = []
    with jsonlines.open(path) as reader:
        for obj in reader.iter(type=list, skip_invalid=True, skip_empty=True):
            data.append([flatten_util.ravel_pytree(obj[0])[0],
                         flatten_util.ravel_pytree(obj[1])[0]])

    print(len(data))
    y, x = coine_data_transform(data)
    y =y[:10]
    x= x[:10]
    plt.plot(y, x)
    plt.show()

    y, x = norm_data_transform(data)
    y =y[:10]
    x= x[:10]
    plt.plot(y, x)
    plt.show()

