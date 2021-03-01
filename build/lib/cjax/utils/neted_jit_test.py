from jax import lax
from jax import numpy as np
from jax import tree_util
from jax import jit, random
from datetime import datetime

@jit
def pytree_dot(x, y) -> float:
    partial_dot = tree_util.tree_multimap(lambda arr1, arr2: np.sum(arr1 * arr2), x, y)
    return tree_util.tree_reduce(lax.add, partial_dot)


def pytree_dot2(x, y) -> float:
    partial_dot = tree_util.tree_multimap(lambda arr1, arr2: np.sum(arr1 * arr2), x, y)
    return tree_util.tree_reduce(lax.add, partial_dot)

@jit
def _nested_jit(x, y):
    s = pytree_dot(x, y)
    s += pytree_dot(x, y)
    s += pytree_dot(x, y)
    s += pytree_dot(x, y)
    s += pytree_dot(x, y)
    return s

@jit
def _jit(x, y):
    s = pytree_dot2(x, y)
    s += pytree_dot2(x, y)
    s += pytree_dot2(x, y)
    s += pytree_dot2(x, y)
    s += pytree_dot2(x, y)
    return s


def no_jit(x, y):
    s = pytree_dot2(x, y)
    s += pytree_dot2(x, y)
    s += pytree_dot2(x, y)
    s += pytree_dot2(x, y)
    s += pytree_dot2(x, y)
    return s


def internal_only_jit(x, y):
    s = pytree_dot(x, y)
    s += pytree_dot(x, y)
    s += pytree_dot(x, y)
    s += pytree_dot(x, y)
    s += pytree_dot(x, y)
    return s

if __name__ == '__main__':
    k1, k2, k3 = random.split(random.PRNGKey(3), 3)
    x = [np.array([0.5, 0.5]), random.uniform(k1, shape=(50000,)), {"beta": 0.5}]
    y = [np.array([0.5, 0.5]), random.uniform(k2, shape=(50000,)), {"beta": 0.5}]
    z = [np.array([0.5, 0.5]), random.uniform(k3, shape=(50000,)), {"beta": 0.5}]
    x1 = x
    s = 0.0

    start_time = datetime.now()
    for i in range(10000):
        if (i%2 ==0):
            x1 = z
        else:
            x1 = x
        s = _jit(x, y)
    end_time = datetime.now()
    print('Std Jit Duration: {}'.format(end_time - start_time))
    print(f'Dot: {s}')

    start_time = datetime.now()
    for i in range(100000):
        if (i % 2 == 0):
            x1 = z
        else:
            x1 = x
        s = _nested_jit(x, y)

    end_time = datetime.now()
    print('Nested Jit Duration: {}'.format(end_time - start_time))
    print(f'Dot: {s}')

    start_time = datetime.now()
    for i in range(100000):
        if (i % 2 == 0):
            x1 = z
        else:
            x1 = x
        s = internal_only_jit(x, y)

    end_time = datetime.now()
    print('internal ops jit Duration: {}'.format(end_time - start_time))
    print(f'Dot: {s}')

    start_time = datetime.now()
    for i in range(100000):
        if (i % 2 == 0):
            x1 = z
        else:
            x1 = x
        s = _nested_jit(x, y)

    end_time = datetime.now()
    print('No Jit Duration: {}'.format(end_time - start_time))





