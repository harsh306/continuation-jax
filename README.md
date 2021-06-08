# continuation-jax : Continuaion Framework for lambda 
Continuation methods of Deep Neural Networks 
Tags: optimization, deep-learning, homotopy, bifurcation-analysis, continuation

[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black) 
[![PyPI version](https://badge.fury.io/py/continuation-jax.svg)](https://badge.fury.io/py/continuation-jax)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
![build](https://travis-ci.com/harsh306/continuation-jax.svg?branch=main)


#### Install using pip:
Package: https://pypi.org/project/continuation-jax/ 
```
pip install continuation-jax
```

#### Import

```python
import cjax
```

#### Math operations on Pytrees
```python
>>> import cjax
>>> from cjax.utils import math_trees
>>> math_trees.pytree_element_mul([2,3,5], 2)
[4, 6, 10]
>>> math_trees.pytree_sub([2,3,5], [1,1,1])
[DeviceArray(1, dtype=int32), DeviceArray(2, dtype=int32), DeviceArray(4, dtype=int32)]
>>> math_trees.pytree_zeros_like({'a':12, 'b':45, 'c':[1,1]})
{'a': 0, 'b': 0, 'c': [0, 0]}

```

#### Examples:
- Examples: https://github.com/harsh306/continuation-jax/tree/main/examples
- Sample Runner: https://github.com/harsh306/continuation-jax/blob/main/model_simple_classifier/run.py

```python
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
from examples.model_simple_classifier.model_classifier import ModelContClassifier
from cjax.utils.abstract_problem import ProblemWraper
import json
from jax.config import config
from datetime import datetime
import mlflow
from cjax.utils.visualizer import pick_array, bif_plot

config.update("jax_debug_nans", True)

# TODO: use **kwargs to reduce params

if __name__ == "__main__":
    problem = ModelContClassifier()
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
```

#### Note on Hyperparameters   

#### Papers:

#### Contact: 
`harshnpathak@gmail.com`
