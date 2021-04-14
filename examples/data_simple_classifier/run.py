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
from examples.data_simple_classifier.data_classifier import DataContClassifier
from cjax.utils.abstract_problem import ProblemWraper
import json
from jax.config import config
from datetime import datetime
import mlflow
from cjax.utils.visualizer import pick_array, bif_plot

config.update("jax_debug_nans", True)

# TODO: use **kwargs to reduce params

if __name__ == "__main__":
    problem = DataContClassifier()
    print(f"problem name: {problem.__class__}")
    problem = ProblemWraper(problem)


    with open(problem.HPARAMS_PATH, "r") as hfile:
        hparams = json.load(hfile)
    mlflow.set_tracking_uri(hparams['meta']["mlflow_uri"])
    mlflow.set_experiment(hparams['meta']["name"])

    with mlflow.start_run(run_name=hparams['meta']["method"]+"-"+hparams["meta"]["optimizer"]) as run:
        mlflow.log_dict(hparams, artifact_file="hparams/hparams.json")
        mlflow.log_text("", artifact_file="output/_touch.txt")
        artifact_uri = mlflow.get_artifact_uri("output/")
        hparams["meta"]["output_dir"] = artifact_uri
        print(f"URI: {artifact_uri}")
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

        figure = bif_plot(hparams["meta"]["output_dir"], pick_array)
        mlflow.log_figure(figure, artifact_file="plots/fig.png")
