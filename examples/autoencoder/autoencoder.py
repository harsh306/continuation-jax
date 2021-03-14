import jax.numpy as np
import numpy as onp
from jax.experimental import stax
from jax.nn.initializers import zeros
from jax.nn import sigmoid
from jax.experimental.stax import Dense, Sigmoid
from cjax.optimizer.optimizer import GDOptimizer, AdamOptimizer
import numpy.random as npr
from jax import random
from jax import jit, vmap, grad
from cjax.utils.abstract_problem import AbstractProblem
from jax.tree_util import tree_map
from cjax.utils.math_trees import *
from cjax.utils.evolve_utils import *
import math
import mlflow
import json
from cjax.utils.custom_nn import constant_2d, HomotopyDense, v_2d
from cjax.utils.datasets import mnist, get_mnist_data, meta_mnist
from examples.torch_data import get_data

data_size = 40000
input_shape = (data_size, 36)
step_size = 0.1
num_steps = 10
code_dim = 1
npr.seed(7)


train_images, labels, _, _ = mnist(permute_train=True)
del _
inputs = train_images[:data_size]
del train_images

u, s, v_t = onp.linalg.svd(inputs, full_matrices=False)
I = np.eye(v_t.shape[-1])
I_add = npr.normal(0.0, 0.002, size=I.shape)
noisy_I = I + I_add

init_fun, predict_fun = stax.serial(
    HomotopyDense(out_dim=4, W_init=v_2d(v_t.T), b_init=zeros),
    HomotopyDense(out_dim=2, W_init=constant_2d(noisy_I), b_init=zeros),
    HomotopyDense(out_dim=4, W_init=constant_2d(noisy_I), b_init=zeros),
    Dense(out_dim=input_shape[-1], W_init=v_2d(v_t), b_init=zeros),
)
del inputs
#
# init_fun, predict_fun = stax.serial(
#     Dense(out_dim=4), Sigmoid,
#     Dense(out_dim=2), Sigmoid,
#     Dense(out_dim=4), Sigmoid,
#     Dense(out_dim=input_shape[-1]),
# )


class PCATopologyAE(AbstractProblem):
    def __init__(self):
        self.HPARAMS_PATH = "hparams.json"

    @staticmethod
    def objective(params, bparam, batch) -> float:
        x, _ = batch
        x = np.reshape(x, (x.shape[0], -1))
        logits = predict_fun(params, x, bparam=bparam[0], activation_func=sigmoid)
        loss = np.mean(np.square((np.subtract(logits, x))))
        # loss += 0.1 * (l2_norm(params) + l2_norm(bparam))
        return loss

    def initial_value(self):
        ae_shape, ae_params = init_fun(random.PRNGKey(0), input_shape)
        # print(ae_shape)
        assert ae_shape == input_shape
        bparam = [np.array([0.01], dtype=np.float64)]
        return ae_params, bparam

    def initial_values(self):
        state_0, bparam_0 = self.initial_value()
        state_1 = tree_map(lambda a: a - 0.08, state_0)
        states = [state_0, state_1]
        bparam_1 = tree_map(lambda a: a + 0.02, bparam_0)
        bparams = [bparam_0, bparam_1]
        return states, bparams



if __name__ == "__main__":

    problem = PCATopologyAE()
    with open(problem.HPARAMS_PATH, "r") as hfile:
        hparams = json.load(hfile)
    mlflow.set_tracking_uri(hparams['meta']["mlflow_uri"])
    mlflow.set_experiment(hparams['meta']["name"])
    with mlflow.start_run(run_name=hparams['meta']["method"]) as run:
        ae_params, bparam = problem.initial_value()
        bparam = pytree_element_add(bparam, 0.99)
        mlflow.log_dict(hparams, artifact_file="hparams/hparams.json")
        artifact_uri = mlflow.get_artifact_uri()
        print("Artifact uri: {}".format(artifact_uri))
        data_loader = iter(get_mnist_data(batch_size=hparams["batch_size"], resize=True))
        num_batches = meta_mnist(batch_size=hparams["batch_size"])["num_batches"]
        print(f"num of bathces: {num_batches}")
        compute_grad_fn = jit(grad(problem.objective, [0]))

        opt = AdamOptimizer(learning_rate=hparams["descent_lr"])
        ma_loss = []
        for epoch in range(500):
            for b_j in range(num_batches):
                batch = next(data_loader)
                grads = compute_grad_fn(ae_params, bparam, batch)
                ae_params = opt.update_params(ae_params, grads[0], step_index=epoch)
                loss = problem.objective(ae_params, bparam, batch)
                ma_loss.append(loss)
                print(f"loss:{loss}  norm:{l2_norm(grads)}")
            opt.lr = exp_decay(epoch, hparams["descent_lr"])
            mlflow.log_metrics({
                "train_loss": float(loss),
                "ma_loss": float(ma_loss[-1]),
                "learning_rate": float(opt.lr),
                "norm grads": float(l2_norm(grads))
            }, epoch)

            if len(ma_loss) > 40:
                loss_check = running_mean(ma_loss, 30)
                if math.isclose(
                        loss_check[-1], loss_check[-2], abs_tol=hparams["loss_tol"]
                ):
                    print(f"stopping at {epoch}")
                    break

        train_images, train_labels, test_images, test_labels = mnist(
            permute_train=False, resize=True
        )
        val_loss = problem.objective(ae_params, bparam, (test_images, test_labels))
        print(f"val loss: {val_loss}")
        mlflow.log_metric("val_loss", float(val_loss))

    # init_c = constant_2d(I)
    # print(init_c(key=0, shape=(8,8)))

