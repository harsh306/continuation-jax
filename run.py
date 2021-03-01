"""
Main file to run contination on the user defined problem. Examples can be found in the examples/ directory.


Continuation is topological procedure to train a neural network. This module tracks all
the critical points or fixed points and dumps them to  output file provided in hparams.json file.

  Typical usage example:

  continuation = ContinuationCreator(
            problem=problem, hparams=hparams
        ).get_continuation_method()
        continuation.run()


"""
from cjax.continuation.creator.continuation_creator import ContinuationCreator
from examples.poly_nn.simple_neural_network import SimpleNeuralNetwork
from examples.conv_nn.conv_nn import ConvNeuralNetwork
from examples.autoencoder.autoencoder import PCATopologyAE
from examples.vae.autoencoder import TopologyVAE
from examples.data_cont_ae.autoencoder import DataTopologyAE
from examples.random_network.random_01 import RandomExp
from examples.conv_nn.resnet_50 import ResNet50Network
from examples.toy.vectror_pitchfork import PitchForkProblem, VectorPitchFork, QuadraticProblem, SigmoidFold
from examples.abstract_problem import ProblemWraper
import json
from jax.config import config
from datetime import datetime

config.update("jax_debug_nans", True)

# TODO: use **kwargs to reduce params

if __name__ == "__main__":
    problem = SigmoidFold()
    problem = ProblemWraper(problem)

    with open(problem.HPARAMS_PATH, "r") as hfile:
        hparams = json.load(hfile)
    start_time = datetime.now()

    if hparams["n_perturbs"] > 1:
        for perturb in range(hparams["n_perturbs"]):
            print(f"Running perturb {perturb}")
            continuation = ContinuationCreator(
                problem=problem, hparams=hparams, key=perturb
            ).get_continuation_method()
            continuation.run()
    else:
        continuation = ContinuationCreator(
            problem=problem, hparams=hparams
        ).get_continuation_method()
        continuation.run()

    end_time = datetime.now()
    print(f"Duration: {end_time-start_time}")
