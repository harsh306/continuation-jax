from cjax.utils.abstract_problem import ProblemWraper
from typing import Dict
from cjax.continuation.base_continuation import Continuation
from cjax.continuation.natural_continuation import NaturalContinuation
from cjax.continuation.secant_continuation import SecantContinuation
from cjax.continuation._arc_len_continuation import PseudoArcLenContinuation
from cjax.continuation._perturbed_arc_len_continuation import (
    PerturbedPseudoArcLenContinuation,
)
from cjax.continuation.perturbed_fixed_arclen_continuation import (
    PerturbedPseudoArcLenFixedContinuation,
)


class ContinuationCreator:
    """Continuation Factory to create the right objects on the fly.
    TODO: Use **kwargs to reduce the size of the constructors.
    """

    def __init__(self, problem: ProblemWraper, hparams: Dict, key=0, mlflow=None):
        self.problem = problem
        self.hparams = hparams
        self.key = key
        self.mlflow = mlflow

    def get_continuation_method(self) -> Continuation:
        """Creates the continuation object based on user arguments.

        options-
        natural: monotonic update of continuation parameter, followed by unconstrained solver
        secant: approximates tangent of the solution path to update both state and continuation parameter,
                followed by unconstrained solver
        parc: To be deprecated (TODO)
        parc-perturb: secant predictor, Langrange min-max updates for corrector with orthogonal constraint
        parc-fix-perturb: secant predictor, Langrange multiplier is fixed (hyperparamter)
                          updates for corrector with orthogonal constraint

        Returns:
            object: Continuation
        Raises:
            NotImplementedError
        """
        if self.hparams["meta"]["method"] == "natural":
            state, bparam = self.problem.initial_value()
            return NaturalContinuation(
                state,
                bparam,
                counter=0,
                objective=self.problem.objective,
                accuracy_fn=self.problem.accuracy_fn,
                hparams=self.hparams,
            )
        elif self.hparams["meta"]["method"] == "secant":
            states, bparams = self.problem.initial_values()
            state, bparam = states[1], bparams[1]
            state_0, bparam_0 = states[0], bparams[0]
            return SecantContinuation(
                state,
                bparam,
                state_0,
                bparam_0,
                counter=0,
                objective=self.problem.objective,
                accuracy_fn=self.problem.accuracy_fn,
                hparams=self.hparams,
            )
        elif self.hparams["meta"]["method"] == "parc":
            states, bparams = self.problem.initial_values()
            state, bparam = states[1], bparams[1]
            state_0, bparam_0 = states[0], bparams[0]
            return PseudoArcLenContinuation(
                state,
                bparam,
                state_0,
                bparam_0,
                counter=0,
                objective=self.problem.objective,
                dual_objective=self.problem.dual_objective,
                hparams=self.hparams,
            )
        elif self.hparams["meta"]["method"] == "parc-perturb":
            states, bparams = self.problem.initial_values()
            state, bparam = (
                states[1],
                bparams[1],
            )  # TODO: remove this hard-coding, basically use the previous two solutions.
            state_0, bparam_0 = states[0], bparams[0]
            return PerturbedPseudoArcLenContinuation(
                state,
                bparam,
                state_0,
                bparam_0,
                counter=0,
                objective=self.problem.objective,
                dual_objective=self.problem.dual_objective,
                hparams=self.hparams,
                key_state=self.key,
            )
        elif self.hparams["meta"]["method"] == "parc-fix-perturb":
            states, bparams = self.problem.initial_values()
            state, bparam = states[1], bparams[1]
            state_0, bparam_0 = states[0], bparams[0]
            return PerturbedPseudoArcLenFixedContinuation(
                state,
                bparam,
                state_0,
                bparam_0,
                counter=0,
                objective=self.problem.objective,
                dual_objective=self.problem.dual_objective,
                accuracy_fn = self.problem.accuracy_fn,
                hparams=self.hparams,
                key_state=self.key,
            )
        else:
            raise NotImplementedError
