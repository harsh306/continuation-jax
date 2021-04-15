"""
Original code from: https://github.com/google/jax/tree/master/examples
"""
import time

import numpy.random as npr

from jax import jit, grad, hessian
from jax.scipy.special import logsumexp
from jax.flatten_util import ravel_pytree
import jax.numpy as jnp
from cjax.utils.datasets import mnist


def init_random_params(scale, layer_sizes, rng=npr.RandomState(0)):
  return [(scale * rng.randn(m, n), scale * rng.randn(n))
          for m, n, in zip(layer_sizes[:-1], layer_sizes[1:])]

def predict(params, inputs):
  activations = inputs
  for w, b in params[:-1]:
    outputs = jnp.dot(activations, w) + b
    activations = jnp.tanh(outputs)

  final_w, final_b = params[-1]
  logits = jnp.dot(activations, final_w) + final_b
  return logits - logsumexp(logits, axis=1, keepdims=True)

def loss(params, batch):
  inputs, targets = batch
  preds = predict(params, inputs)
  return -jnp.mean(jnp.sum(preds * targets, axis=1))

def accuracy(params, batch):
  inputs, targets = batch
  target_class = jnp.argmax(targets, axis=1)
  predicted_class = jnp.argmax(predict(params, inputs), axis=1)
  return jnp.mean(predicted_class == target_class)


if __name__ == "__main__":
  layer_sizes = [36,10]
  param_scale = 0.1
  step_size = 0.001
  num_epochs = 10000
  batch_size = 5000

  train_images, train_labels, test_images, test_labels = mnist(resize=True,permute_train=False)
  num_train = train_images.shape[0]
  num_complete_batches, leftover = divmod(num_train, batch_size)
  num_batches = num_complete_batches + bool(leftover)

  def data_stream():
    rng = npr.RandomState(0)
    while True:
      perm = rng.permutation(num_train)
      for i in range(num_batches):
        batch_idx = perm[i * batch_size:(i + 1) * batch_size]
        yield train_images[batch_idx], train_labels[batch_idx]
  batches = data_stream()

  @jit
  def update(params, batch):
    grads = grad(loss)(params, batch)
    return [(w - step_size * dw, b - step_size * db)
            for (w, b), (dw, db) in zip(params, grads)]

  params = init_random_params(param_scale, layer_sizes)
  for epoch in range(num_epochs):
    start_time = time.time()
    for _ in range(num_batches):
      params = update(params, next(batches))
    epoch_time = time.time() - start_time

    train_acc = accuracy(params, (train_images, train_labels))
    test_acc = accuracy(params, (test_images, test_labels))
    if epoch%100==0:
        print("Epoch {} in {:0.2f} sec".format(epoch, epoch_time))
        print("Training set accuracy {}".format(train_acc))
        print("Test set accuracy {}".format(test_acc))
  H = hessian(loss)(params, (test_images, test_labels))
  h, _ = ravel_pytree(H)
  eigen_vals = jnp.linalg.eigvals(h.reshape(370, 370)).real
  eigen_vals = sorted(eigen_vals, reverse=True)
  print(eigen_vals)
  neg_count = len(list(filter(lambda x: (x < 0), eigen_vals)))

  # we can also do len(list1) - neg_count
  pos_count = len(list(filter(lambda x: (x >= 0), eigen_vals)))

  print("Positive numbers in the eigen: ", pos_count)
  print("Negative numbers in the eigen: ", neg_count)

  # outputs:
  #Positive numbers in the eigen:  194
  #Negative numbers in the eigen:  176