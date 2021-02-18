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
from src.continuation.creator.continuation_creator import ContinuationCreator
from examples.poly_nn.simple_neural_network import SimpleNeuralNetwork
from examples.conv_nn.conv_nn import ConvNeuralNetwork
from examples.autoencoder.autoencoder import PCATopologyAE
from examples.random_network.random_01 import RandomExp
from examples.conv_nn.resnet_50 import ResNet50Network
from examples.pitchfork2d.vectror_pitchfork import PitchForkProblem, VectorPitchFork
from examples.abstract_problem import ProblemWraper
import json

# TODO: use **kwargs to reduce params

if __name__ == "__main__":
    HPARAMS_PATH = "examples/pitchfork2d/hparams.json"

    with open(HPARAMS_PATH, "r") as hfile:
        hparams = json.load(hfile)

    problem = PitchForkProblem()
    problem = ProblemWraper(problem)
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
