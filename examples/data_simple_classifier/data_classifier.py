import jax.numpy as np
import numpy as onp
from jax.experimental import stax
from jax.nn.initializers import zeros, orthogonal, delta_orthogonal, uniform
from jax.nn import sigmoid, hard_tanh, soft_sign, relu
from jax.experimental.stax import Dense, Sigmoid, Softplus, Tanh, LogSoftmax,Relu
from cjax.optimizer.optimizer import GDOptimizer, AdamOptimizer, OptimizerCreator
import numpy.random as npr
from jax import random
from jax import jit, vmap, grad
from cjax.utils.abstract_problem import AbstractProblem
from jax.tree_util import tree_map
from cjax.utils.math_trees import *
from cjax.utils.evolve_utils import *
from cjax.continuation.states.state_variables import StateWriter
import math
import mlflow
import json
from cjax.utils.custom_nn import constant_2d, HomotopyDense, v_2d
from cjax.utils.datasets import mnist, get_mnist_data, meta_mnist
import pickle
from examples.torch_data import get_data

npr.seed(7)
orth_init_cont = False
input_shape = (30000, 36)
def accuracy(params, bparams, batch):
    inputs, targets = batch
    target_class = np.argmax(targets, axis=-1)
    predicted_class = np.argmax(predict_fun(params, inputs), axis=-1)
    return np.mean(predicted_class == target_class)

if orth_init_cont:
    init_fun, predict_fun = stax.serial(
        HomotopyDense(out_dim=18, W_init=orthogonal(), b_init=zeros),
        Dense(out_dim=10, W_init=orthogonal(), b_init=zeros), LogSoftmax
    )
else:
    # baseline network
    init_fun, predict_fun = stax.serial(
        Dense(out_dim=18), Relu,
        Dense(out_dim=10), LogSoftmax
    )

class DataContClassifier(AbstractProblem):
    def __init__(self):
        self.HPARAMS_PATH = "hparams.json"

    @staticmethod
    def objective(params, bparam, batch) -> float:
        x, targets = batch
        x = np.reshape(x, (x.shape[0], -1))
        logits = predict_fun(params, x, bparam=bparam[0], activation_func=relu)
        loss = -np.mean(np.sum(logits * targets, axis=1))
        loss += 5e-7 * (l2_norm(params)) #+ l2_norm(bparam))
        return loss

    @staticmethod
    def accuracy(params, bparam, batch):
        x, targets = batch
        x = np.reshape(x, (x.shape[0], -1))
        target_class = np.argmax(targets, axis=-1)
        predicted_class = np.argmax(predict_fun(params, x, bparam=bparam[0], activation_func=relu), axis=-1)
        return np.mean(predicted_class == target_class)

    def initial_value(self):
        ae_shape, ae_params = init_fun(random.PRNGKey(0), input_shape)
        bparam = [np.array([0.01], dtype=np.float64)]
        return ae_params, bparam

    def initial_values(self):
        state_0, bparam_0 = self.initial_value()
        state_1 = tree_map(lambda a: a - 0.08, state_0)
        states = [state_0, state_1]
        bparam_1 = tree_map(lambda a: a + 0.05, bparam_0)
        bparams = [bparam_0, bparam_1]
        return states, bparams


if __name__ == "__main__":

    problem = DataContClassifier()
    with open(problem.HPARAMS_PATH, "r") as hfile:
        hparams = json.load(hfile)
    mlflow.set_tracking_uri(hparams['meta']["mlflow_uri"])
    mlflow.set_experiment(hparams['meta']["name"])
    with mlflow.start_run(run_name=hparams['meta']["method"]+"-"+hparams["meta"]["optimizer"]) as run:
        ae_params, bparam = problem.initial_value()
        bparam = pytree_element_add(bparam, 0.0)
        mlflow.log_dict(hparams, artifact_file="hparams/hparams.json")
        artifact_uri = mlflow.get_artifact_uri()
        print("Artifact uri: {}".format(artifact_uri))

        mlflow.log_text("", artifact_file="output/_touch.txt")
        artifact_uri2 = mlflow.get_artifact_uri("output/")
        print("Artifact uri: {}".format(artifact_uri2))
        hparams["meta"]["output_dir"] = artifact_uri2
        file_name = f"{artifact_uri2}/version.jsonl"

        sw = StateWriter(file_name=file_name)

        data_loader = iter(get_mnist_data(batch_size=hparams["batch_size"], resize=True, filter=hparams['filter']))
        num_batches = meta_mnist(batch_size=hparams["batch_size"], filter=hparams['filter'])["num_batches"]
        print(f"num of bathces: {num_batches}")
        compute_grad_fn = jit(grad(problem.objective, [0,1]))

        opt = OptimizerCreator(hparams["meta"]["optimizer"], learning_rate=hparams["natural_lr"]).get_optimizer()
        ma_loss = []
        for epoch in range(hparams["warmup_period"]):
            for b_j in range(num_batches):
                batch = next(data_loader)
                ae_grads, b_grads = compute_grad_fn(ae_params, bparam, batch)
                grads = ae_grads
                ae_params = opt.update_params(ae_params, ae_grads, step_index=epoch)
                bparam = opt.update_params(bparam, b_grads, step_index=epoch)
                loss = problem.objective(ae_params, bparam, batch)
                ma_loss.append(loss)
                print(f"loss:{loss}  norm:{l2_norm(grads)}")
            opt.lr = exp_decay(epoch, hparams["natural_lr"])
            mlflow.log_metrics({
                "train_loss": float(loss),
                "ma_loss": float(ma_loss[-1]),
                "learning_rate": float(opt.lr),
                "bparam":float(bparam[0]),
                "norm_grads": float(l2_norm(grads))
            }, epoch)

            if len(ma_loss) > 100:
                loss_check = running_mean(ma_loss, 50)
                if math.isclose(
                        loss_check[-1], loss_check[-2], abs_tol=hparams["loss_tol"]
                ):
                    print(f"stopping at {epoch}")
                    break

        train_images, train_labels, test_images, test_labels = mnist(
            permute_train=False, resize=True, filter=hparams["filter"]
        )

        val_loss = problem.objective(ae_params, bparam, (test_images, test_labels))
        print(f"val loss: {val_loss, type(ae_params)}")
        val_acc = accuracy(ae_params, bparam, (test_images, test_labels))
        print(f"val acc: {val_acc}")
        mlflow.log_metric("val_acc", float(val_acc))
        mlflow.log_metric("val_loss", float(val_loss))



        q = float(l2_norm(grads[0]))
        if sw:
            sw.write([
                {'u':ae_params},
                {'t': bparam},
                {'f':loss},
                {'q':q},
            ])
        else:
            print('sw none')
    with open(artifact_uri2+'params.pkl', 'wb') as file:
        pickle.dump(ae_params, file)
