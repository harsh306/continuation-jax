import operator
import jax.numpy as np
from typing import Any, Optional, Tuple, Union
from flax import linen as nn

Array = Any

def homotopy(x: Array, alpha: float = 1.0) -> Array:
  r"""Exponential linear unit activation function.

  Computes the element-wise function:

  .. math::
    \mathrm{elu}(x) = \begin{cases}
      x, & x > 0\\
      \alpha \left(\exp(x) - 1\right), & x \le 0
    \end{cases}

  Args:
    x : input array
    alpha : scalar or array of alpha values (default: 1.0)
  """
  return np.multiply((1-alpha), x) + np.multiply(alpha, nn.relu(x))