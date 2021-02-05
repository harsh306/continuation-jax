from unittest import TestCase

from utils.math import pytree_dot, pytree_sub, pytree_array_equal, pytree_zeros_like
import jax.numpy as np
from examples.conv_nn.conv_nn import ConvNeuralNetwork
import copy


class TestPytree(TestCase):
    def setUp(self):
        self.x = [np.array([0.5, 0.5]), 0.5, {"beta": 0.5}]
        self.y = [np.array([0.5, 0.5]), 0.5, {"beta": 0.5}]
        self.z = [np.array([0.0, 0.0]), 0.0, {"beta": 0.0}]
        self.cnn_problem = ConvNeuralNetwork()
        self.flax_conv, self.bparams = self.cnn_problem.initial_value()
        self.flax_conv1 = copy.deepcopy(self.flax_conv)


class TestPytreeDot(TestPytree):
    def test_dot(self):
        self.assertEqual(1.0, pytree_dot(self.x, self.y))
        self.assertEqual(pytree_array_equal(self.flax_conv, self.flax_conv1), True)
        self.assertEqual(361.93063, pytree_dot(self.flax_conv, self.flax_conv1))


class TestPytreeSub(TestPytree):
    def test_sub(self):
        self.assertEqual(pytree_array_equal(self.z, pytree_sub(self.x, self.y)), True)
        self.assertEqual(pytree_array_equal(self.flax_conv, self.flax_conv1), True)
        self.assertEqual(
            pytree_array_equal(
                pytree_zeros_like(self.flax_conv),
                pytree_sub(self.flax_conv, self.flax_conv1),
            ),
            True,
        )
