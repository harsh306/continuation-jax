"""
author: Paul Bruillard
"""

import numpy


def get_rotation(src, dst):
    """
    Takes two n-dimensional vectors and returns an
    nxn rotation matrix mapping src to dst.
    Raises Value Error when unsuccessful.
    """

    def __assert_rotation(R):
        if R.ndim != 2:
            raise ValueError("R must be a matrix")
        a, b = R.shape
        if a != b:
            raise ValueError("R must be square")
        if (
            not numpy.isclose(numpy.abs(numpy.eye(a) - numpy.dot(R, R.T)).max(), 0)
        ) or (not numpy.isclose(numpy.abs(numpy.eye(a) - numpy.dot(R.T, R)).max(), 0)):
            raise ValueError("R is not diagonal")

    def __normalize(x):
        return x / numpy.sqrt(numpy.sum(x ** 2))

    if src.shape != dst.shape or src.ndim != 1:
        raise ValueError(
            "src and dst must be 1-dimensional arrays with the same shape."
        )
    x = __normalize(src.copy())
    y = __normalize(dst.copy())
    # compute angle between x and y in their spanning space
    theta = numpy.arccos(
        numpy.dot(x, y)
    )  # they are normalized so there is no denominator
    if numpy.isclose(theta, 0):
        raise ValueError("x and y are co-linear")
    # construct the 2d rotation matrix connecting x to y in their spanning space
    R = numpy.array(
        [[numpy.cos(theta), -numpy.sin(theta)], [numpy.sin(theta), numpy.cos(theta)]]
    )
    __assert_rotation(R)
    # get projections onto Span<x,y> and its orthogonal complement
    u = x
    v = __normalize((y - (numpy.dot(u, y) * u)))
    P = numpy.outer(u, u.T) + numpy.outer(
        v, v.T
    )  # projection onto 2d space spanned by x and y
    Q = numpy.eye(n) - P  # projection onto the orthogonal complement of Span<x,y>
    # lift the rotation matrix into the n-dimensional space
    uv = numpy.hstack((u[:, None], v[:, None]))
    R = Q + numpy.dot(uv, numpy.dot(R, uv.T))
    __assert_rotation(R)
    if numpy.any(numpy.logical_not(numpy.isclose(numpy.dot(R, x), y))):
        raise ValueError("Rotation matrix did not work")
    return R


if __name__ == "__main__":
    numpy.random.seed(0)
    n = 5
    src = numpy.random.random(n)
    dst = numpy.random.random(n)
    R = get_rotation(src, dst)
