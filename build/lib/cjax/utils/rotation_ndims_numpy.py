"""
author: Paul Bruillard, harsh
"""

# import jax.numpy as jnp
# from jax import random


# from utils.math_trees import *
import numpy as np
from typing import Any

# def get_rotation_pytree(cjax: Any, dst: Any) -> Any:
#     """
#     Takes two n-dimensional vectors/Pytree and returns an
#     nxn rotation matrix mapping cjax to dst.
#     Raises Value Error when unsuccessful.
#     """
#     def __assert_rotation(R):
#         if R.ndim != 2:
#             raise ValueError("R must be a matrix")
#         a, b = R.shape
#         if a != b:
#             raise ValueError("R must be square")
#         if (
#                 not jnp.isclose(jnp.abs(jnp.eye(a) - jnp.dot(R, R.T)).min(), 0.0, rtol=0.85)
#         ) or (not jnp.isclose(jnp.abs(jnp.eye(a) - jnp.dot(R.T, R)).min(), 0.0, rtol=0.85)):
#             raise ValueError("R is not diagonal")
#
#     if not pytree_shape_array_equal(cjax, dst):
#         raise ValueError(
#             "cjax and dst must be 1-dimensional arrays with the same shape."
#         )
#
#     x = pytree_normalized(cjax)
#     y = pytree_normalized(dst)
#     n = len(dst)
#
#     # compute angle between x and y in their spanning space
#     theta = jnp.arccos(
#         jnp.dot(x, y)
#     )  # they are normalized so there is no denominator
#     if jnp.isclose(theta, 0):
#         raise ValueError("x and y are co-linear")
#     # construct the 2d rotation matrix connecting x to y in their spanning space
#     R = jnp.array(
#         [[jnp.cos(theta), -jnp.sin(theta)], [jnp.sin(theta), jnp.cos(theta)]]
#     )
#     __assert_rotation(R)
#     # get projections onto Span<x,y> and its orthogonal complement
#     u = x
#     v = pytree_normalized((y - (jnp.dot(u, y) * u)))
#     P = jnp.outer(u, u.T) + jnp.outer(
#         v, v.T
#     )  # projection onto 2d space spanned by x and y
#     Q = jnp.eye(n) - P  # projection onto the orthogonal complement of Span<x,y>
#     # lift the rotation matrix into the n-dimensional space
#     uv = jnp.hstack((u[:, None], v[:, None]))
#     R = Q + jnp.dot(uv, jnp.dot(R, uv.T))
#     __assert_rotation(R)
#     if jnp.any(jnp.logical_not(jnp.isclose(jnp.dot(R, x), y, rtol=0.25))):
#         raise ValueError("Rotation matrix did not work")
#     return R


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


if __name__ == "__main__":
    # n = 5
    # key = random.PRNGKey(10)
    # k1, k2 = random.split(key, 2)
    # cjax = random.normal(k1, [n])
    # dst = random.normal(k2, [n])
    # R = get_rotation_pytree(cjax, dst)
    # transformed_vector = np.dot(R, cjax)
    # print(jnp.dot(transformed_vector, dst))

    n = 6
    src = np.hstack([np.zeros(n - 1), 1.0])
    dst = 4 * np.ones(n)
    sample = np.hstack([np.ones(n - 1), 0.0])
    R = get_rotation_array(src, dst)
    transformed_vector = np.dot(R, sample) + dst
    print(transformed_vector)
