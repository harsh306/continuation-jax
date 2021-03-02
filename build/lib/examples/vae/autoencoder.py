import jax.numpy as np
import numpy as onp
import numpy.random as npr
from cjax.utils.abstract_problem import AbstractProblem
from jax.tree_util import tree_map

import jax.numpy as jnp
from jax import random
from jax.experimental import stax
from jax.experimental.stax import Dense, FanOut, Relu, Softplus


batch_size = 8
input_shape = (batch_size, 8)
step_size = 0.1
num_steps = 10
code_dim = 1
npr.seed(7)


def synth_batches():
    while True:
        images = npr.rand(*input_shape).astype("float32")
        yield images


batches = synth_batches()
inputs = next(batches)
u, s, v_t = onp.linalg.svd(inputs, full_matrices=False)
I = np.eye(v_t.shape[-1])
I_add = npr.normal(0.0, 0.002, size=I.shape)
noisy_I = I + I_add

encoder_init, encode = stax.serial(
    Dense(512),
    Relu,
    Dense(512),
    Relu,
    FanOut(2),
    stax.parallel(Dense(10), stax.serial(Dense(10), Softplus)),
)

decoder_init, decode = stax.serial(
    Dense(512),
    Relu,
    Dense(512),
    Relu,
    Dense(8),
)


def gaussian_kl(mu, sigmasq):
    """KL divergence from a diagonal Gaussian to the standard Gaussian."""
    return -0.5 * jnp.sum(1.0 + jnp.log(sigmasq) - mu ** 2.0 - sigmasq)


def gaussian_sample(rng, mu, sigmasq):
    """Sample a diagonal Gaussian."""
    return mu + jnp.sqrt(sigmasq) * random.normal(rng, mu.shape)


def bernoulli_logpdf(logits, x):
    """Bernoulli log pdf of data x given logits."""
    return -jnp.sum(jnp.logaddexp(0.0, jnp.where(x, -1.0, 1.0) * logits))


def elbo(rng, params, bparam, images):
    """Monte Carlo estimate of the negative evidence lower bound."""
    enc_params, dec_params = params
    mu_z, sigmasq_z = encode(enc_params, images)
    logits_x = decode(dec_params, gaussian_sample(rng, mu_z, sigmasq_z))
    return bernoulli_logpdf(logits_x, images) - bparam[0] * gaussian_kl(mu_z, sigmasq_z)


class TopologyVAE(AbstractProblem):
    def __init__(self):
        self.HPARAMS_PATH = "examples/vae/hparams.json"

    @staticmethod
    def objective(params, bparam) -> float:
        elbo_rng, data_rng = random.split(random.PRNGKey(1))
        loss = -elbo(elbo_rng, params, bparam, inputs)
        # logits = predict_fun(params, inputs, bparam=bparam[0], activation_func=sigmoid)
        # loss = np.mean(np.square((np.subtract(logits, inputs))))
        # loss += 0.1*(l2_norm(params) + l2_norm(bparam))
        return np.mean(loss)

    def initial_value(self):
        enc_init_rng, dec_init_rng = random.split(random.PRNGKey(2))
        _, init_encoder_params = encoder_init(enc_init_rng, (batch_size, 8))  # 28*28
        _, init_decoder_params = decoder_init(dec_init_rng, (batch_size, 10))
        init_params = [init_encoder_params, init_decoder_params]
        bparam = [np.array([0.01], dtype=np.float64)]
        return init_params, bparam

    def initial_values(self):
        state, bparam = self.initial_value()
        state_1 = tree_map(lambda a: a + 0.005, state)
        states = [state, state_1]
        bparam_1 = tree_map(lambda a: a + 0.05, bparam)
        bparams = [bparam, bparam_1]
        return states, bparams


if __name__ == "__main__":
    problem = TopologyVAE()
    ae_params, bparams = problem.initial_value()
    loss = problem.objective(ae_params, bparams)
    print(loss)

    # init_c = constant_2d(I)
    # print(init_c(key=0, shape=(8,8)))
