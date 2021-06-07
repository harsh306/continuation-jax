from typing import Tuple

from jax import grad, jit
from jax.experimental.optimizers import l2_norm
from cjax.continuation.methods.corrector.base_corrector import Corrector
from cjax.optimizer.optimizer import OptimizerCreator
from cjax.utils.datasets import meta_mnist, get_preload_mnist_data, mnist
from cjax.utils.data_img_gamma import get_mnist_batch_alter, mnist_gamma
from cjax.utils.evolve_utils import running_mean, exp_decay
import math


class UnconstrainedCorrector(Corrector):
    """Minimize the objective using gradient based method."""

    def __init__(self, objective, concat_states, grad_fn, value_fn, accuracy_fn , hparams, dataset_tuple):
        self.concat_states = concat_states
        self._state = None
        self._bparam = None
        self.opt = OptimizerCreator(
            opt_string=hparams["meta"]["optimizer"], learning_rate=hparams["natural_lr"]
        ).get_optimizer()
        self.objective = objective
        self.accuracy_fn = accuracy_fn
        self.warmup_period = hparams["warmup_period"]
        self.hparams = hparams
        self.grad_fn = grad_fn
        self.value_fn = value_fn
        self._assign_states()
        if hparams["meta"]["dataset"] == "mnist":
            (self.train_images, self.train_labels,
             self.test_images, self.test_labels) = dataset_tuple
            if hparams["continuation_config"] == 'data':
                # data continuation
                self.data_loader = iter(
                    get_mnist_batch_alter(
                        self.train_images,
                        self.train_labels,
                        self.test_images,
                        self.test_labels,
                        alter=self._bparam,
                        batch_size=hparams["batch_size"],
                        resize=hparams["resize_to_small"],
                        filter=hparams["filter"]
                    )
                )
            else:
                # model continuation
                self.data_loader = iter(
                    get_preload_mnist_data(self.train_images,
                                           self.train_labels,
                                           self.test_images,
                                           self.test_labels,
                                           batch_size=hparams["batch_size"],
                                           resize=hparams["resize_to_small"],
                                           filter=hparams["filter"])
                )

            self.num_batches = meta_mnist(hparams["batch_size"], hparams["filter"])["num_batches"]
        else:
            self.data_loader = None
            self.num_batches = 1

    def _assign_states(self):
        self._state, self._bparam = self.concat_states

    def correction_step(self) -> Tuple:
        """Given the current state optimize to the correct state.

        Returns:
          (state: problem parameters, bparam: continuation parameter) Tuple
        """

        quality = 1.0
        ma_loss = []
        stop = False
        print("learn_rate", self.opt.lr)
        for k in range(self.warmup_period):
            for b_j in range(self.num_batches):
                batch = next(self.data_loader)
                grads = self.grad_fn(self._state, self._bparam, batch)
                self._state = self.opt.update_params(self._state, grads[0])
                quality = l2_norm(grads)
                value = self.value_fn(self._state, self._bparam, batch)
                ma_loss.append(value)
                self.opt.lr = exp_decay(k, self.hparams["natural_lr"])
                if self.hparams["local_test_measure"] == "norm_gradients":
                    if quality > self.hparams["quality_thresh"]:
                        pass
                        print(
                            f"quality {quality}, {self.opt.lr} ,{k}"
                        )
                    else:
                        stop = True
                        print(
                            f"quality {quality} stopping at , {k}th step"
                        )
                else:
                    if len(ma_loss) >= 20:
                        tmp_means = running_mean(ma_loss, 10)
                        if math.isclose(
                                tmp_means[-1],
                                tmp_means[-2],
                                abs_tol=self.hparams["loss_tol"],
                        ):
                            print(
                                f"stopping at , {k}th step"
                            )
                            stop = True
            if stop:
                print("breaking")
                break

        val_loss = self.value_fn(self._state, self._bparam, (self.test_images, self.test_labels))
        val_acc = self.accuracy_fn(self._state, self._bparam, (self.test_images, self.test_labels))
        return self._state, self._bparam, quality, value, val_loss, val_acc
