from unittest import TestCase
import jax.numpy as np
import json
from cjax.continuation.methods.predictor.arc_secant_predictor import SecantPredictor

class TestSecantPredictor(TestCase):
    def setUp(self) -> None:
        self.x = [np.array([0.5, 0.5]), 0.5, {"beta": 0.5}]
        self.y = [np.array([0.5, 0.5]), 0.5, {"beta": 0.5}]
        self.z = [np.array([0.0, 0.0]), 0.0, {"beta": 0.0}]
        self.x1 = [np.array([1.0, 1.0]), 1.0, {"beta": 1.0}]
        self.bparam = [np.array([0.5])]
        concat_states = [
            (self.x, self.bparam),
            (self.z, self.bparam),
            None,
        ]
        with open("../../../../examples/autoencoder/hparams.json", 'r') as f:
            hparams = json.load(f)
        self.predictor  = SecantPredictor(concat_states, delta_s=0.02,
                                          prev_delta_s=0.03,omega=1.0,
                                          net_spacing_param=0.1,
                                          net_spacing_bparam=0.2, hparams=hparams)


class TestPredictionStep(TestSecantPredictor):
    def test_prediction_step(self):
        self.predictor._assign_states()
        print(self.predictor.state)
        self.predictor.prediction_step()
        print(self.predictor.state, self.predictor.bparam, self.predictor.hparams["delta_s"])
        self.assertEqual(1,1)
