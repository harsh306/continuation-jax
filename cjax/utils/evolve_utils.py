import numpy as onp
from jax.experimental.optimizers import l2_norm
from numpy import random as npr
from cjax.utils.math_trees import pytree_sub

def exp_decay(epoch, initial_lrate):
    k = 0.02
    lrate = initial_lrate * onp.exp(-k * epoch)
    return lrate

def center_data(X):
    mean_x = onp.mean(X, axis=0, keepdims=True)
    reduced_mean = onp.subtract(X, mean_x)
    reduced_mean = reduced_mean.astype(onp.float32)
    return reduced_mean

def grouper(iterable, threshold=0.01):
    prev = None
    group = []
    for item in iterable:
        if not prev or l2_norm(pytree_sub(item, prev)) <= threshold:
            group.append(item)
        else:
            yield group
            group = [item]
        prev = item
    if group:
        yield group


def running_mean(x, N):
    cumsum = onp.cumsum(onp.insert(x, 0, 0))
    return (cumsum[N:] - cumsum[:-N]) / float(N)


def get_cheapest_ant(ants_norm_grads, ants_loss_values, local_test="loss"):
    m = min(ants_norm_grads)
    g_indicies = [i for i, j in enumerate(ants_norm_grads) if j == m]
    print(f"Number of minima by norm: {len(g_indicies)}")

    m = min(ants_loss_values)
    indicies = [i for i, j in enumerate(ants_loss_values) if j == m]
    print(f"Number of minima by loss: {len(indicies)}")
    if local_test == "ma_loss":
        index = npr.randint(0, len(indicies))
        cheapest = indicies[index]
    elif local_test == "norm_gradients":
        index = npr.randint(0, len(g_indicies))
        cheapest = g_indicies[index]
    else:
        index = npr.randint(0, len(indicies))
        cheapest = indicies[index]
    return cheapest
