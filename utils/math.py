"""
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
from jax import tree_util


def pytree_dot(x, y) -> float:
    partial_dot = tree_util.tree_multimap(lambda arr1, arr2: np.sum(arr1 * arr2), x, y)
    return tree_util.tree_reduce(lax.add, partial_dot)


def pytree_sub(x, y):
    return tree_util.tree_multimap(lax.sub, x, y)


def pytree_array_equal(x, y):
    is_eq = tree_util.tree_multimap(lambda arr1, arr2: np.array_equal(arr1, arr2), x, y)
    return tree_util.tree_reduce(operator.and_, is_eq)
