import matplotlib.pyplot as plt
import jax.numpy as np
import json
import jsonlines
from jax.experimental.optimizers import l2_norm
from jax.tree_util import *
from jax import flatten_util
import numpy as onp
from utils.math_trees import *
import matplotlib as mpl
from typing import Tuple

"""
cmaps['Sequential'] = [
            'Greys', 'Purples', 'Blues', 'Greens', 'Oranges', 'Reds',
            'YlOrBr', 'YlOrRd', 'OrRd', 'PuRd', 'RdPu', 'BuPu',
            'GnBu', 'PuBu', 'YlGnBu', 'PuBuGn', 'BuGn', 'YlGn']
            

cmaps['Diverging'] = [
            'PiYG', 'PRGn', 'BrBG', 'PuOr', 'RdGy', 'RdBu',
            'RdYlBu', 'RdYlGn', 'Spectral', 'coolwarm', 'bwr', 'seismic']
            
plt.scatter(y, x, c=z, cmap='gray_r')
"""


def pick_array(data: list, start: int = 0, end: int = 1) -> Tuple:
    """
    Pick one or more element from params to be used in plot
    Args:
        data: state values at every continuation step
        start: start index
        end: end index

    Returns:
        Tuple of lists y, x
    """
    # print(data[0], data[1])
    param = []
    bparam = []
    value = []
    for k in range(len(data)):
        param.append(data[k][0][start:end])
        bparam.append(data[k][1][0])
        value.append(data[k][2][0])
    return param, bparam, value


def cosine_data_transform(data) -> Tuple:
    """
    Delta dot product between consecutive delta-param vectors to be used in plot
    Args:
        data: state values at every continuation step

    Returns:
        Tuple of lists y, x
    """
    params = []
    bparam = []
    value = []
    k = 0
    while k + 2 < len(data):
        params.append(
            pytree_dot(
                pytree_sub(data[k][0], data[k + 1][0]),
                pytree_sub(data[k + 1][0], data[k + 2][0]),
            )
        )
        bparam.append(data[k][1][0])
        value.append(data[k][2][0])
        k = k + 1
    return params, bparam, value


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
    value = []
    for k in range(len(data)):
        params.append(l2_norm(data[k][0]))
        bparam.append(data[k][1][0])
        value.append(data[k][2][0])
    return params, bparam, value


def read_data(path):
    data = []
    with jsonlines.open(path) as reader:
        for obj in reader.iter(type=list, skip_invalid=True, skip_empty=True):
            for k in range(len(obj)):
                data.append(
                    [
                        flatten_util.ravel_pytree(obj[0])[0],
                        flatten_util.ravel_pytree(obj[1])[0],
                        flatten_util.ravel_pytree(obj[2])[0],
                    ]
                )
    return data


# TODO
def export():
    pass


# TODO
def perturbplot():
    pass


def get_loss(thetas):
    z = onp.random.normal(1.0, 1.0, len(thetas))
    return z


def bif_plot(dpath, func, n=3):
    cmaps = [
        "Greys",
        "Reds",
        "Purples",
        "Blues",
        "Greens",
        "Oranges",
        "YlOrBr",
        "YlOrRd",
        "OrRd",
        "PuRd",
        "RdPu",
        "BuPu",
        "GnBu",
        "PuBu",
        "YlGnBu",
        "PuBuGn",
        "BuGn",
        "YlGn",
    ]
    # cmaps = ['coolwarm', 'PuOr']

    for i in range(n):
        _path = f"{dpath}/version_{i}.json"
        data = read_data(_path)
        y, x, z = func(data)
        # plt.plot(x, y)
        plt.scatter(x, y, c=z, cmap=cmaps[i] + "_r", alpha=1.0)
        plt.plot(x, y, alpha=1.0)

    plt.ylabel(f"{func.__name__} Network Parameters")
    plt.xlabel(f"Continuation Parameter")
    sm1 = plt.cm.ScalarMappable(
        cmap=cmaps[0] + "_r", norm=plt.Normalize(vmin=min(z), vmax=max(z))
    )
    sm = plt.cm.ScalarMappable(
        cmap=cmaps[1] + "_r", norm=plt.Normalize(vmin=min(z), vmax=max(z))
    )
    plt.colorbar(sm1)
    plt.colorbar(sm)
    plt.show()
    plt.clf()


def bif_plotv(path, func):
    cmaps = [
        "Greys",
        "Purples",
        "Blues",
        "Greens",
        "Oranges",
        "Reds",
        "YlOrBr",
        "YlOrRd",
        "OrRd",
        "PuRd",
        "RdPu",
        "BuPu",
        "GnBu",
        "PuBu",
        "YlGnBu",
        "PuBuGn",
        "BuGn",
        "YlGn",
    ]
    path = f"{path}/version.json"
    data = read_data(path)
    y, x, z = func(data)
    # plt.plot(x, y)
    plt.scatter(x, y, c=z, cmap="coolwarm_r", alpha=0.6)
    plt.ylabel(f"{func.__name__} Network Parameters")
    plt.xlabel(f"Continuation Parameter")
    sm = plt.cm.ScalarMappable(
        cmap="coolwarm_r", norm=plt.Normalize(vmin=min(z), vmax=max(z))
    )
    plt.colorbar(sm)
    plt.show()
    plt.clf()


if __name__ == "__main__":

    path = f"/opt/ml/output/"
    # bif_plotv(path, pick_array)
    # bif_plotv(path, norm_data_transform)
    bif_plot(path, pick_array, 5)
    # bif_plotv(path, norm_data_transform)
    # bif_plot(path, cosine_data_transform, 2)
    # bif_plot(path, norm_data_transform, 2)
