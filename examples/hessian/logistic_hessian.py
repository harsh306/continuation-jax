import jax.numpy as jnp
from jax import grad, hessian, jit, random
from jax.flatten_util import ravel_pytree
from jax.experimental.optimizers import l2_norm
from jax.nn import softmax
from jax.nn import log_softmax, logsumexp
key = random.PRNGKey(0)
def sigmoid(x):
    return 0.5 * (jnp.tanh(x / 2) + 1)

# Outputs probability of a label being true.
def predict(W, b, inputs):
    return jnp.dot(inputs, W) + b

# Build a toy dataset.
inputs = jnp.array([[0.52, 1.12,  0.77],
                   [0.88, -1.08, 0.15],
                   [0.52, 0.06, -1.30],
                   [0.74, -2.49, 1.39]])
targets = jnp.array([1, 1, 2, 3])
def _one_hot(x, k, dtype=jnp.float32):
    """Create a one-hot encoding of x of size k."""
    return jnp.array(x[:, None] == jnp.arange(k), dtype)
targets = _one_hot(targets, 3)


def loss(W, b):
    logits = predict(W, b, inputs)
    preds = logits - logsumexp(logits, axis=1, keepdims=True)
    loss = -jnp.mean(jnp.sum(preds * targets, axis=1))
    loss += 0.001* (l2_norm(W) + l2_norm(b))
    return loss



# Initialize random model coefficients
key, W_key, b_key = random.split(key, 3)
W = random.normal(W_key, (3,3))
b = random.normal(b_key, (3,))

f = lambda W: predict(W, b, inputs)
print("loss:",loss(W, b))
for i in range(1000):
    W_grad, b_grad = grad(loss, (0, 1))(W, b)
    W = W -0.005*W_grad
    b = b - 0.005 * b_grad
print("loss:",loss(W, b))
H = hessian(loss)(W, b)
h, _ = ravel_pytree(H)
eigen_vals = jnp.linalg.eigvals(h.reshape(9,9)).real
eigen_vals = sorted(eigen_vals, reverse=True)
print(eigen_vals) # should be all positive for convex function
#outputs: [0.2406996, 0.16962789, 0.13137847, 0.07919562, 0.037625454, 0.02834966, 0.00042202187, 0.00042201488, 0.00037239227]