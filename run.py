from src.continuation.creator.continuation_creator import ContinuationCreator
from examples.poly_nn.simple_neural_network import SimpleNeuralNetwork
from examples.pitchfork2d.vectror_pitchfork import PitchForkProblem, VectorPitchFork
from examples.abstract_problem import ProblemWraper
import json

if __name__ == "__main__":
    HPARAMS_PATH = "examples/pitchfork2d/hparams.json"

    with open(HPARAMS_PATH, "r") as hfile:
        hparams = json.load(hfile)

    problem = VectorPitchFork()
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
