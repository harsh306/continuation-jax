"""
author: Paul Bruillard, harsh
"""

import jax.numpy as jnp
from cjax.utils.math_trees import *
from typing import Any


def get_rotation_pytree(src: Any, dst: Any) -> Any:
    """
    Takes two n-dimensional vectors/Pytree and returns an
    nxn rotation matrix mapping cjax to dst.
    Raises Value Error when unsuccessful.
    """

    def __assert_rotation(R):
        if R.ndim != 2:
            print("R must be a matrix")
        a, b = R.shape
        if a != b:
            print("R must be square")
        if (
            not jnp.isclose(jnp.abs(jnp.eye(a) - jnp.dot(R, R.T)).max(), 0.0, rtol=0.5)
        ) or (
            not jnp.isclose(jnp.abs(jnp.eye(a) - jnp.dot(R.T, R)).max(), 0.0, rtol=0.5)
        ):
            print("R is not diagonal")

    if not pytree_shape_array_equal(src, dst):
        print("cjax and dst must be 1-dimensional arrays with the same shape.")

    x = pytree_normalized(src)
    y = pytree_normalized(dst)
    n = len(dst)

    # compute angle between x and y in their spanning space
    theta = jnp.arccos(jnp.dot(x, y))  # they are normalized so there is no denominator
    if jnp.isclose(theta, 0):
        print("x and y are co-linear")
    # construct the 2d rotation matrix connecting x to y in their spanning space
    R = jnp.array([[jnp.cos(theta), -jnp.sin(theta)], [jnp.sin(theta), jnp.cos(theta)]])
    __assert_rotation(R)
    # get projections onto Span<x,y> and its orthogonal complement
    u = x
    v = pytree_normalized(pytree_sub(y, (jnp.dot(u, y) * u)))
    P = jnp.outer(u, u.T) + jnp.outer(
        v, v.T
    )  # projection onto 2d space spanned by x and y
    Q = jnp.eye(n) - P  # projection onto the orthogonal complement of Span<x,y>
    # lift the rotation matrix into the n-dimensional space
    uv = jnp.hstack((u[:, None], v[:, None]))

    R = Q + jnp.dot(uv, jnp.dot(R, uv.T))
    __assert_rotation(R)
    if jnp.any(jnp.logical_not(jnp.isclose(jnp.dot(R, x), y, rtol=0.25))):
        print("Rotation matrix did not work")
    return R


def get_rotation_array(src: Any, dst: Any) -> Any:
    """
    Takes two n-dimensional vectors and returns an
    nxn rotation matrix mapping cjax to dst.
    Raises Value Error when unsuccessful.
    """

    def __assert_rotation(R):
        if R.ndim != 2:
            raise ValueError("R must be a matrix")
        a, b = R.shape
        if a != b:
            raise ValueError("R must be square")
        if (not np.isclose(np.abs(np.eye(a) - np.dot(R, R.T)).max(), 0)) or (
            not np.isclose(np.abs(np.eye(a) - np.dot(R.T, R)).max(), 0)
        ):
            raise ValueError("R is not diagonal")

    def __normalize(x):
        return x / np.sqrt(np.sum(x ** 2))

    if src.shape != dst.shape or src.ndim != 1:
        raise ValueError(
            "cjax and dst must be 1-dimensional arrays with the same shape."
        )
    x = __normalize(src.copy())
    y = __normalize(dst.copy())

    # compute angle between x and y in their spanning space
    theta = np.arccos(np.dot(x, y))  # they are normalized so there is no denominator
    if np.isclose(theta, 0):
        raise ValueError("x and y are co-linear")
    # construct the 2d rotation matrix connecting x to y in their spanning space
    R = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    __assert_rotation(R)
    # get projections onto Span<x,y> and its orthogonal complement
    u = x
    v = __normalize((y - (np.dot(u, y) * u)))
    P = np.outer(u, u.T) + np.outer(
        v, v.T
    )  # projection onto 2d space spanned by x and y
    Q = np.eye(n) - P  # projection onto the orthogonal complement of Span<x,y>
    # lift the rotation matrix into the n-dimensional space
    uv = np.hstack((u[:, None], v[:, None]))
    R = Q + np.dot(uv, np.dot(R, uv.T))
    __assert_rotation(R)
    if np.any(np.logical_not(np.isclose(np.dot(R, x), y))):
        raise ValueError("Rotation matrix did not work")
    return R


def projection_affine(n_dim, u, n, u_0):
    """

    Args:
        n_dim: affine transformation space
        u: random point to be projected on n as L
        n: secant normal vector
        u_0: secant starting point

    Returns:

    """
    n_norm = l2_norm(n)
    I = jnp.eye(n_dim)

    p2 = [0 * k for k in range(n_dim)]
    for k in range(n_dim):
        p2[k] = (jnp.dot(n, I[k]) / n_norm ** 2) * n

    p2 = jnp.asarray([p2[i] for i in range(n_dim)])
    u_0 = u_0.reshape(n_dim, 1)
    I = jnp.eye(n_dim)
    t1 = jnp.block([[I, u_0], [jnp.zeros(shape=(1, n_dim)), 1.0]])
    t2 = jnp.block(
        [[p2, jnp.zeros(shape=(n_dim, 1))], [jnp.zeros(shape=(1, n_dim)), 1.0]]
    )
    t3 = jnp.block([[I, -1 * u_0], [jnp.zeros(shape=(1, n_dim)), 1.0]])
    P = jnp.matmul(jnp.matmul(t1, t2), t3)
    pr = jnp.matmul(P, jnp.hstack([u, 1.0]))
    pr = lax.slice(pr, [0], [n_dim])
    return pr


if __name__ == "__main__":
    n = 5
    # key = random.PRNGKey(10)
    # k1, k2 = random.split(key, 2)
    # cjax = random.normal(k1, [n])
    # dst = random.normal(k2, [n])
    # R = get_rotation_pytree(cjax, dst)
    # transformed_vector = np.dot(R, cjax)
    # print(jnp.dot(transformed_vector, dst))

    # n = 3
    # cjax= np.array([0.0,0.0,1.0])
    # dst = np.array([3.0,3.0,3.0])
    # sample = np.array([1.0,2.0,0.0])
    # R = get_rotation_array(cjax, dst)
    # transformed_vector = np.dot(R, sample) + dst
    # print(transformed_vector)
