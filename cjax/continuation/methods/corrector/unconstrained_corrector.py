from typing import Tuple

from jax import grad, jit
from jax.experimental.optimizers import l2_norm
from cjax.continuation.methods.corrector.base_corrector import Corrector
from cjax.optimizer.optimizer import OptimizerCreator
from cjax.utils.datasets import get_mnist_data
from examples.torch_data import get_data

class UnconstrainedCorrector(Corrector):
    """Minimize the objective using gradient based method."""

    def __init__(
        self, objective, concat_states, grad_fn, value_fn, hparams
    ):
        self.concat_states = concat_states
        self._state = None
        self._bparam = None
        self.opt = OptimizerCreator(
            opt_string=hparams["meta"]["optimizer"], learning_rate=hparams["natural_lr"]
        ).get_optimizer()
        self.objective = objective
        self.warmup_period = hparams['warmup_period']
        self.hparams = hparams
        self.grad_fn = grad_fn
        self.value_fn = value_fn
        # self.data_loader = get_data(dataset=hparams["meta"]['dataset'],
        #                                        batch_size=hparams['batch_size'],
        #                                        num_workers=hparams['data_workers'],
        #                             train_only=True, test_only=False)
        self.data_loader = iter(get_mnist_data(batch_size=hparams['batch_size'],
                                               resize=hparams['resize_to_small']))

    def _assign_states(self):
        self._state, self._bparam = self.concat_states

    def correction_step(self) -> Tuple:
        """Given the current state optimize to the correct state.

        Returns:
          (state: problem parameters, bparam: continuation parameter) Tuple
        """
        self._assign_states()
        quality = 1.0
        for k in range(self.warmup_period):
            grads = self.grad_fn(self._state, self._bparam, next(iter(self.data_loader)))
            self._state = self.opt.update_params(self._state, grads[0])
            quality = l2_norm(grads)
        value = self.value_fn(self._state, self._bparam, next(iter(self.data_loader)))
        return self._state, self._bparam, quality, value
