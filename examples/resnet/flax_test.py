from functools import partial
from typing import Any, Callable, Sequence, Tuple
from flax import linen as nn
import jax.numpy as jnp
import homotopy
from jax import random
ModuleDef = Any

class Multiplier(nn.Module):
    @nn.compact
    def __call__(self, x, alpha: float = 0.0):
        return jnp.square(x)*alpha

if __name__ == '__main__':
    m = Multiplier()
    v = m.init(random.PRNGKey(0), jnp.ones((), jnp.float32))
    loss = ()
    out = m.apply(v, 2.0, alpha=100)
    print(out)
    out = m.apply(v, 2.0, alpha=10)
    print(out)