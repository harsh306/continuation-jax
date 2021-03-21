# continuation-jax : Continuaion Framework for lambda 
Continuation methods of Deep Neural Networks 
Tags: optimization, deep-learning, homotopy, bifurcation-analysis, continuation

[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black) 
[![PyPI version](https://badge.fury.io/py/continuation-jax.svg)](https://badge.fury.io/py/continuation-jax)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)



#### Install using pip:
```
pip install continuation-jax
```

#### Import

```python
import cjax
help(cjax)
```

#### Simple Math on Pytrees
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
- Sample Runner: https://github.com/harsh306/continuation-jax/blob/main/run.py

```python
from cjax.continuation.creator.continuation_creator import ContinuationCreator
from examples.toy.vectror_pitchfork import SigmoidFold
from cjax.utils.abstract_problem import ProblemWraper
import json
from jax.config import config
from datetime import datetime
from cjax.utils.visualizer import bif_plot, pick_array

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
    
    bif_plot(hparams['output_dir'], pick_array, hparams['n_perturbs'])
```

#### Note on Hyperparameters   

#### Papers:


#### Contact: 
`harshnpathak@gmail.com`
