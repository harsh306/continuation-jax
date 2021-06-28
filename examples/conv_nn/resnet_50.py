from jax.experimental import stax
from jax.experimental.stax import (
    AvgPool,
    BatchNorm,
    Conv,
    Dense,
    FanInSum,
    FanOut,
    Flatten,
    GeneralConv,
    Identity,
    MaxPool,
    Relu,
    LogSoftmax,
)


from cjax.utils.abstract_problem import AbstractProblem
import jax.numpy as np
from jax import random
from jax.experimental.optimizers import l2_norm
from jax.tree_util import *
import resnet_flax_1
import mlflow
from cjax.utils.datasets import get_mnist_data, meta_mnist, mnist
from jax import jit, grad
from cjax.continuation.states.state_variables import StateWriter
from cjax.optimizer.optimizer import OptimizerCreator
import json
import pickle
import math
from cjax.utils.evolve_utils import running_mean, exp_decay

# batch_size = 64
# num_classes = 10
# input_shape = (28, 28, 1, batch_size)
# step_size = 0.1
# num_steps = 10
#
# # ResNet blocks compose other layers
#
#
# def LambdaIdentity():
#     """Layer construction function for an identity layer."""
#     init_fun = lambda rng, input_shape: (input_shape, ())
#     apply_fun = lambda params, inputs, **kwargs: inputs * kwargs.get("bparams")
#     return init_fun, apply_fun
#
#
# LambdaIdentity = LambdaIdentity()
#
#
# def ConvBlock(kernel_size, filters, strides=(2, 2)):
#     ks = kernel_size
#     filters1, filters2, filters3 = filters
#     Main = stax.serial(
#         Conv(filters1, (1, 1), strides),
#         BatchNorm(),
#         Relu,
#         Conv(filters2, (ks, ks), padding="SAME"),
#         BatchNorm(),
#         Relu,
#         Conv(filters3, (1, 1)),
#         BatchNorm(),
#     )
#     Shortcut = stax.serial(Conv(filters3, (1, 1), strides), BatchNorm())
#     return stax.serial(FanOut(2), stax.parallel(Main, Shortcut), FanInSum, Relu)
#
#
# def IdentityBlock(kernel_size, filters):
#     ks = kernel_size
#     filters1, filters2 = filters
#
#     def make_main(input_shape):
#         # the number of output channels depends on the number of input channels
#         return stax.serial(
#             Conv(filters1, (1, 1)),
#             BatchNorm(),
#             Relu,
#             Conv(filters2, (ks, ks), padding="SAME"),
#             BatchNorm(),
#             Relu,
#             Conv(input_shape[3], (1, 1)),
#             BatchNorm(),
#         )
#
#     Main = stax.shape_dependent(make_main)
#     return stax.serial(FanOut(2), stax.parallel(Main, Identity), FanInSum, Relu)
#
#
# # ResNet architectures compose layers and ResNet blocks
#
#
# def ResNet50(num_classes):
#     return stax.serial(
#         GeneralConv(("HWCN", "OIHW", "NHWC"), 64, (7, 7), (2, 2), "SAME"),
#         BatchNorm(),
#         Relu,
#         MaxPool((3, 3), strides=(2, 2)),
#         ConvBlock(3, [64, 64, 256], strides=(1, 1)),
#         IdentityBlock(3, [64, 64]),
#         IdentityBlock(3, [64, 64]),
#         ConvBlock(3, [128, 128, 512]),
#         IdentityBlock(3, [128, 128]),
#         IdentityBlock(3, [128, 128]),
#         IdentityBlock(3, [128, 128]),
#         ConvBlock(3, [256, 256, 1024]),
#         IdentityBlock(3, [256, 256]),
#         IdentityBlock(3, [256, 256]),
#         IdentityBlock(3, [256, 256]),
#         IdentityBlock(3, [256, 256]),
#         IdentityBlock(3, [256, 256]),
#         ConvBlock(3, [512, 512, 2048]),
#         IdentityBlock(3, [512, 512]),
#         IdentityBlock(3, [512, 512]),
#         AvgPool((7, 7)),
#         Flatten,
#         Dense(num_classes),
#         LogSoftmax,
#     )
#
#
# def ResNet(num_classes):
#     return stax.serial(
#         GeneralConv(("HWCN", "OIHW", "NHWC"), 32, (7, 7), (2, 2), "SAME"),
#         BatchNorm(),
#         Relu,
#         MaxPool((3, 3), strides=(2, 2)),
#         ConvBlock(3, [4, 4, 4], strides=(1, 1)),
#         IdentityBlock(3, [4, 4]),
#         AvgPool((3, 3)),
#         Flatten,
#         Dense(num_classes),
#         LogSoftmax,
#     )
#

# def synth_batches():
#     rng = npr.RandomState(0)
#     while True:
#         images = rng.rand(*input_shape).astype("float32")
#         labels = rng.randint(num_classes, size=(batch_size, 1))
#         onehot_labels = labels == np.arange(num_classes)
#         yield images, onehot_labels

# batches = synth_batches()
# inputs, outputs = next(batches)

#
# init_fun, predict_fun = ResNet(num_classes)

img_size = 6
channels = 1
resnet_model_def = resnet_flax_1.ResNet18(num_classes=10)
#resnet_model_def.apply()
input_shape = (1, img_size, img_size, channels)

class ResNetProblem(AbstractProblem):
    def __init__(self):
        self.HPARAMS_PATH = "hparams.json"

    @staticmethod
    def objective(params, batch) -> float:
        train_batch, targets = batch
        print(train_batch.shape)
        train_batch = np.moveaxis(train_batch, 1, -1)
        # train_batch = np.moveaxis(train_batch, -1, 0)
        # train_batch = np.moveaxis(train_batch, -1, 0)
        # train_batch = np.moveaxis(train_batch, -2, -1)
        print(35*"#")
        print(params)
        print(train_batch.shape)
        logits = resnet_model_def.apply({'params': params["params"], "batch_stats": params["batch_stats"]}, train_batch)
        #logits = predict_fun(params, train_batch)
        loss = -np.sum(logits * targets)
        loss += l2_norm(params) + l2_norm(bparam)
        return loss

    @staticmethod
    def accuracy(params, batch):
        train_batch, targets = batch
        #x = np.reshape(x, (x.shape[0], -1))
        train_batch = np.moveaxis(train_batch, 1, -1)
        # train_batch = np.moveaxis(train_batch, -1, 0)
        # train_batch = np.moveaxis(train_batch, -2, -1)

        target_class = np.argmax(targets, axis=-1)
        predicted_class = resnet_model_def.apply({'params': params["params"]}, train_batch)
        #predicted_class = np.argmax(predict_fun(params, train_batch, bparam=bparam[0], activation_func=relu), axis=-1)
        return np.mean(predicted_class == target_class)

    def initial_value(self):
        #_, init_params = init_fun(random.PRNGKey(0), (28, 28, 1, 64))
        # state = init_params
        state = resnet_model_def.init(random.PRNGKey(0), np.ones(input_shape, np.float32))
        bparam = [np.array([1.02], dtype=np.float32)]
        return state, bparam

    def initial_values(self):
        state, bparam = self.initial_value()
        state_1 = tree_map(lambda a: a + 0.05, state)
        states = [state, state_1]
        bparam_1 = tree_map(lambda a: a + 0.05, bparam)
        bparams = [bparam, bparam_1]
        return states, bparams



if __name__ == '__main__':
    problem  = ResNetProblem()
    with open(problem.HPARAMS_PATH, "r") as hfile:
        hparams = json.load(hfile)
    mlflow.set_tracking_uri(hparams['meta']["mlflow_uri"])
    mlflow.set_experiment(hparams['meta']["name"])
    with mlflow.start_run(run_name=hparams['meta']["method"]+"-"+hparams["meta"]["optimizer"]) as run:
        resnet_params, bparam = problem.initial_value()
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
        one_batch = next(data_loader)
        print(one_batch[0].shape)
        num_batches = meta_mnist(batch_size=hparams["batch_size"], filter=hparams['filter'])["num_batches"]
        print(f"num of bathces: {num_batches}")
        print(resnet_params)
        compute_grad_fn = jit(grad(problem.objective, [0]))
        opt = OptimizerCreator(hparams["meta"]["optimizer"], learning_rate=hparams["natural_lr"]).get_optimizer()
        ma_loss = []
        for epoch in range(hparams["warmup_period"]):
            for b_j in range(num_batches):
                batch = next(data_loader)
                ae_grads = compute_grad_fn(resnet_params, batch)
                print(ae_grads)
                resnet_params.update({"params": opt.update_params(resnet_params["params"], ae_grads[0], step_index=epoch)})
                # bparam = opt.update_params(bparam, b_grads, step_index=epoch)
                loss = problem.objective(resnet_params, batch)
                ma_loss.append(loss)
                print(f"loss:{loss}  norm:{l2_norm(ae_grads)}")
            opt.lr = exp_decay(epoch, hparams["natural_lr"])
            mlflow.log_metrics({
                "train_loss": float(loss),
                "ma_loss": float(ma_loss[-1]),
                "learning_rate": float(opt.lr),
                "bparam": float(bparam[0]),
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

        val_loss = problem.objective(resnet_params, bparam, (test_images, test_labels))
        print(f"val loss: {val_loss, type(resnet_params)}")
        val_acc = problem.accuracy(resnet_params, bparam, (test_images, test_labels))
        print(f"val acc: {val_acc}")
        mlflow.log_metric("val_acc", float(val_acc))
        mlflow.log_metric("val_loss", float(val_loss))

        q = float(l2_norm(ae_grads[0]))
        if sw:
            sw.write([
                {'u': resnet_params},
                {'t': bparam},
                {'f': loss},
                {'q': q},
            ])
        else:
            print('sw none')
    with open(artifact_uri2 + 'params.pkl', 'wb') as file:
        pickle.dump(resnet_params, file)

    with open(artifact_uri2 + 'params.pkl', 'rb') as file:
        p = pickle.load(file)

    val_loss = problem.objective(p, bparam, (test_images, test_labels))
    print(f"val loss: {val_loss, type(resnet_params)}")
    val_acc = problem.accuracy(p, bparam, (test_images, test_labels))
    print(f"val acc: {val_acc}")



