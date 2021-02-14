import matplotlib.pyplot as plt
import jax.numpy as np
import json
import jsonlines
from jax.experimental.optimizers import l2_norm
from jax.tree_util import *
from jax import flatten_util
from utils.math_trees import *
from typing import Tuple


def pick_array(data: list, start:int = 0, end:int = 1) -> Tuple:
    """
    Pick one or more element from params to be used in plot
    Args:
        data: state values at every continuation step
        start: start index
        end: end index

    Returns:
        Tuple of lists y, x
    """
    param = []
    bparam = []
    for k in range(len(data)):
        param.append(data[k][0][start:end])
        bparam.append(data[k][1][0])
    return param, bparam


def coine_data_transform(data) -> Tuple:
    """
    Delta dot product between consecutive param vectors to be used in plot
    Args:
        data: state values at every continuation step

    Returns:
        Tuple of lists y, x
    """
    params = []
    bparam = []
    for k in range(len(data)-1):
        params.append(pytree_dot(data[k][0], data[k+1][0]))
        bparam.append(data[k][1][0])
    return params, bparam


def norm_data_transform(data) -> Tuple:
    """
    L2 norm of param vectors to be used in plot.
    Args:
        data: state values at every continuation step

    Returns:
        Tuple of lists y, x
    """
    params = []
    bparam = []
    for k in range(len(data)):
        params.append(l2_norm(data[k][0]))
        bparam.append(data[k][1][0])
    return params, bparam

def read_data(path):
    data = []
    with jsonlines.open(path) as reader:
        for obj in reader.iter(type=list, skip_invalid=True, skip_empty=True):
            data.append([flatten_util.ravel_pytree(obj[0])[0],
                           flatten_util.ravel_pytree(obj[1])[0]])
    return data


# TODO
def export():
    pass

# TODO
def perturbplot():
    pass

if __name__ == '__main__':

    for i in range(3):
        path = f'/opt/ml/output/random_01/version_{i}.json'
        data = read_data(path)
        y, x = coine_data_transform(data)
        plt.plot(y, x)
    plt.show()

    for i in range(3):
        path = f'/opt/ml/output/random_01/version_{i}.json'
        data = read_data(path)
        y, x = norm_data_transform(data)
        plt.plot(y, x)
    plt.show()


