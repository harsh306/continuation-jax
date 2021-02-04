from examples.abstract_problem import AbstractProblem, ProblemWraper
from typing import Dict
from src.continuation.base_continuation import Continuation
from src.continuation.natural_continuation import NaturalContinuation
from src.continuation.arc_len_continuation import PseudoArcLenContinuation
from src.continuation.perturbed_arc_len_continuation import (
    PerturbedPseudoArcLenContinuation,
)


class ContinuationCreator:
    def __init__(self, problem: ProblemWraper, hparams: Dict, key=0):
        self.problem = problem
        self.hparams = hparams
        self.key = key

    def get_continuation_method(self) -> Continuation:
        if self.hparams["meta"]["method"] == "natural":
            state, bparam = self.problem.initial_value()
            return NaturalContinuation(
                state,
                bparam,
                counter=0,
                objective=self.problem.objective,
                output_file=self.hparams["meta"]["output_dir"] + "/version.json",
                hparams=self.hparams,
            )

        elif self.hparams["meta"]["method"] == "parc":
            states, bparams = self.problem.initial_values()
            state, bparam = states[0], bparams[0]
            state_0, bparam_0 = states[1], bparams[1]
            return PseudoArcLenContinuation(
                state,
                bparam,
                state_0,
                bparam_0,
                counter=0,
                objective=self.problem.objective,
                dual_objective=self.problem.dual_objective,
                lagrange_multiplier=self.hparams["lagrange_init"],
                output_file=self.hparams["meta"]["output_dir"] + "/version.json",
                hparams=self.hparams,
            )
        elif self.hparams["meta"]["method"] == "parc-perturb":
            states, bparams = self.problem.initial_values()
            state, bparam = states[0], bparams[0]
            state_0, bparam_0 = states[1], bparams[1]
            return PerturbedPseudoArcLenContinuation(
                state,
                bparam,
                state_0,
                bparam_0,
                counter=0,
                objective=self.problem.objective,
                dual_objective=self.problem.dual_objective,
                lagrange_multiplier=self.hparams["lagrange_init"],
                output_file=self.hparams["meta"]["output_dir"],
                hparams=self.hparams,
                key_state=self.key,
            )
        else:
            raise NotImplementedError
