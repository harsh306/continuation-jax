from unittest import TestCase

from cjax.utils.math_trees import *
import jax.numpy as np
from examples.conv_nn.conv_nn import ConvNeuralNetwork
import copy


class TestPytree(TestCase):
    def setUp(self):
        self.x = [np.array([0.5, 0.5]), 0.5, {"beta": 0.5}]
        self.y = [np.array([0.5, 0.5]), 0.5, {"beta": 0.5}]
        self.z = [np.array([0.0, 0.0]), 0.0, {"beta": 0.0}]
        self.x1 = [np.array([1.0, 1.0]), 1.0, {"beta": 1.0}]
        self.cnn_problem = ConvNeuralNetwork()
        self.flax_conv, self.bparams = self.cnn_problem.initial_value()
        self.flax_conv1 = copy.deepcopy(self.flax_conv)


class TestPytreeDot(TestPytree):
    def test_dot(self):
        self.assertEqual(1.0, pytree_dot(self.x, self.y))
        self.assertEqual(pytree_array_equal(self.flax_conv, self.flax_conv1), True)
        self.assertEqual(361.93063, pytree_dot(self.flax_conv, self.flax_conv1))


class TestPytreeOnesLike(TestPytree):
    def test_dot(self):
        self.assertEqual(pytree_array_equal(self.x1, pytree_ones_like(self.x)), True)


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
