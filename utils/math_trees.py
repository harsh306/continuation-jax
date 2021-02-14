"""
Inspired from:
@misc{gehring2019fax,
  author = {Clement Gehring, Pierre-Luc Bacon, Florian Schaefer},
  title = {{FAX: differentiating fixed point problems in JAX}},
  note = {Available at: https://github.com/gehring/fax},
  year = {2019}
}
"""

import operator

from jax import lax
from jax import numpy as np
from jax import tree_util, flatten_util
from jax.experimental.optimizers import l2_norm


def pytree_dot(x, y) -> float:
    partial_dot = tree_util.tree_multimap(lambda arr1, arr2: np.sum(arr1 * arr2), x, y)
    return tree_util.tree_reduce(lax.add, partial_dot)


def pytree_sub(x, y):
    return tree_util.tree_multimap(lax.sub, x, y)


def pytree_array_equal(x, y):
    is_eq = tree_util.tree_multimap(lambda arr1, arr2: np.array_equal(arr1, arr2), x, y)
    return tree_util.tree_reduce(operator.and_, is_eq)


def pytree_shape_array_equal(x, y):
    is_eq = tree_util.tree_multimap(lambda arr1, arr2: (arr1.shape==arr2.shape), x, y)
    return tree_util.tree_reduce(operator.and_, is_eq)

def pytree_zeros_like(x):
    return tree_util.tree_map(lambda arr: 0 * arr, x)

def pytree_ones_like(x):
    return tree_util.tree_map(lambda arr: 0 * arr +1, x)


def pytree_element_add(x, s):
    return tree_util.tree_map(lambda a: a + s, x)

def pytree_to_vec(x):
    return flatten_util.ravel_pytree(x)

def pytree_normalized(x):
    return tree_util.tree_map(lambda a: a / l2_norm(x), x)
