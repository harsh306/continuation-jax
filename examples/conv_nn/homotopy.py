import operator
import numpy as np
from typing import Any, Optional, Tuple, Union


Array = Any

def homotopy(x: Array, alpha: Array = 1.0) -> Array:
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
  return (1-alpha) * x + alpha * x