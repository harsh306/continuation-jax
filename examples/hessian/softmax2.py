import jax.numpy as np
import numpy as onp
from jax.experimental import stax
from jax.nn.initializers import zeros, orthogonal, delta_orthogonal, uniform, normal
from jax.nn import sigmoid, hard_tanh, soft_sign, relu, logsumexp
from jax.experimental.stax import Dense, Sigmoid, Softplus, Tanh, LogSoftmax,Relu
from cjax.optimizer.optimizer import GDOptimizer, AdamOptimizer, OptimizerCreator
import numpy.random as npr
from jax import random
from jax import jit, vmap, grad, hessian
from cjax.utils.abstract_problem import AbstractProblem
from jax.tree_util import tree_map
from jax.flatten_util import ravel_pytree
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
orth_init_cont = True
input_shape = (30000, 36)
def accuracy(params, batch):
    inputs, targets = batch
    target_class = np.argmax(targets, axis=-1)
    predicted_class = np.argmax(predict_fun(params, inputs), axis=-1)
    return np.mean(predicted_class == target_class)


init_fun, predict_fun = Dense(out_dim=10, W_init=normal(), b_init=normal())


class ModelContClassifier(AbstractProblem):
    def __init__(self):
        self.HPARAMS_PATH = "hparams.json"

    @staticmethod
    def objective(params, batch) -> float:
        x, targets = batch
        logits = predict_fun(params, x)
        logits = logits - logsumexp(logits, axis=1, keepdims=True)
        loss = -np.mean(np.sum(logits * targets, axis=1))
        loss += 5e-6 * (l2_norm(params)) #+ l2_norm(bparam))
        return loss

    @staticmethod
    def accuracy(params, batch):
        x, targets = batch
        x = np.reshape(x, (x.shape[0], -1))
        target_class = np.argmax(targets, axis=-1)
        predicted_class = np.argmax(predict_fun(params, x), axis=-1)
        return np.mean(predicted_class == target_class)

    def initial_value(self):
        ae_shape, ae_params = init_fun(random.PRNGKey(0), input_shape)
        return ae_params

    def initial_values(self):
        return 0


if __name__ == "__main__":

    problem = ModelContClassifier()
    with open(problem.HPARAMS_PATH, "r") as hfile:
        hparams = json.load(hfile)
    mlflow.set_tracking_uri(hparams['meta']["mlflow_uri"])
    mlflow.set_experiment(hparams['meta']["name"])
    with mlflow.start_run(run_name=hparams['meta']["method"]+"-"+hparams["meta"]["optimizer"]) as run:
        ae_params = problem.initial_value()

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
        num_batches = meta_mnist(batch_size=hparams["batch_size"],filter=hparams['filter'])["num_batches"]
        print(f"num of bathces: {num_batches}")
        compute_grad_fn = jit(grad(problem.objective, [0]))

        opt = OptimizerCreator(hparams["meta"]["optimizer"], learning_rate=hparams["natural_lr"]).get_optimizer()
        ma_loss = []
        for epoch in range(hparams["warmup_period"]):
            for b_j in range(num_batches):
                batch = next(data_loader)
                ae_grads = compute_grad_fn(ae_params, batch)
                ae_params = opt.update_params(ae_params, ae_grads[0], step_index=epoch)
                loss = problem.objective(ae_params, batch)
                ma_loss.append(loss)
                print(f"loss:{loss}  norm:{l2_norm(ae_grads)}")
            #opt.lr = exp_decay(epoch, hparams["natural_lr"])
            mlflow.log_metrics({
                "train_loss": float(loss),
                "ma_loss": float(ma_loss[-1]),
                "learning_rate": float(opt.lr),
                "norm grads": float(l2_norm(ae_grads))
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

        val_loss = problem.objective(ae_params, (test_images, test_labels))
        print(f"val loss: {val_loss, type(ae_params)}")
        val_acc = accuracy(ae_params, (test_images, test_labels))
        print(f"val acc: {val_acc}")
        mlflow.log_metric("val_acc", float(val_acc))
        mlflow.log_metric("val_loss", float(val_loss))



        q = float(l2_norm(ae_grads[0]))
        if sw:
            sw.write([
                {'u':ae_params},
                {'f':loss},
                {'q':q},
            ])
        else:
            print('sw none')
    with open(artifact_uri2+'params.pkl', 'wb') as file:
        pickle.dump(ae_params, file)

    dg2 = hessian(problem.objective, argnums=[0])(ae_params, batch)
    mtree, _ = ravel_pytree(dg2)
    eigen = np.linalg.eigvals(mtree.reshape(370, 370)).real
    eigen = sorted(eigen, reverse=True)
    neg_count = len(list(filter(lambda x: (x < 0), eigen)))

    # we can also do len(list1) - neg_count
    pos_count = len(list(filter(lambda x: (x >= 0), eigen)))

    print("Positive numbers in the eigen: ", pos_count)
    print("Negative numbers in the eigen: ", neg_count)
