from pygrank.algorithms.filters.abstract.filter import GraphFilter
from pygrank.core import (
    GraphSignal,
)
from pygrank.core.utils import obj2id
from pygrank.core import backend
from pygrank.algorithms.filters.krylov_utils import (
    krylov_base,
    krylov2original,
    krylov_error_bound,
)


class ClosedFormGraphFilter(GraphFilter):
    """Implements a graph filter described as an aggregation of graph signal diffusion certain number of hops away
    while weighting these by corresponding coefficients."""

    def __init__(
        self,
        krylov_dims: int = None,
        coefficient_type: str = "taylor",
        optimization_dict: dict = None,
        *args,
        **kwargs
    ):
        """
        Args:
            krylov_dims: Optional. Performs the Lanczos method to estimate filter outcome in the Krylov space
                of the graph with degree equal to the provided dimensions. This considerably speeds up filtering
                but ends up providing an *approximation* of true graph filter outcomes.
                If None (default) filters are computed in the node space, which can be slower but
                yields exact computations. Otherwise, a numeric value
                equal to the number of latent Krylov space dimensions is required.
            coefficient_type: Optional. If "taylor" (default), provided coefficients are considered
                to define a Taylor expansion. If "chebyshev", they are considered to be the coefficients of a Chebyshev
                expansion, which provides more robust errors but require normalized personalization. These approaches
                are **not equivalent** for the same coefficient values; changing this argument could cause adhoc
                filters to not work as indented.
            optimization_dict: Optional. If it is a dict, the filter keeps intermediate values that can help it
                avoid most (if not all) matrix multiplication when run again for the same graph signal. Setting this
                parameter to None (default) can save approximately **half the memory** the algorithm uses but
                slows down tuning iteration times to O(edges) instead of O(nodes). Note that the same dict needs to
                be potentially passed to multiple algorithms that take the same graph signal as input to see any
                improvement.
        """
        super().__init__(*args, **kwargs)
        self.krylov_dims = krylov_dims
        self.coefficient_type = coefficient_type.lower()
        self.optimization_dict = optimization_dict

    def references(self):
        refs = super().references()
        if self.coefficient_type == "chebyshev":
            refs.append("Chebyshev coefficients \\cite{yu2021chebyshev}")
        if self.krylov_dims is not None:
            refs.append(
                "Lanczos acceleration \\cite{susnjara2015accelerated} in the"
                + str(self.krylov_dims)
                + "-dimensional Krylov space"
            )
        if self.optimization_dict is not None:
            refs.append("dictionary-based hashing \\cite{krasanakis2022pygrank}")
        return refs

    def _start(self, M, personalization, ranks, *args, **kwargs):
        self.coefficient = None
        if self.coefficient_type == "chebyshev":
            self.prev_term = 0
        if self.krylov_dims is not None:
            V, H = krylov_base(M, personalization.np, int(self.krylov_dims))
            self.krylov_base = V
            self.krylov_H = H
            self.zero_coefficient = self.coefficient
            self.krylov_result = H * 0
            self.Mpower = backend.eye(int(self.krylov_dims))
            error_bound = krylov_error_bound(V, H, M, personalization.np)
            if error_bound > 0.01:
                raise Exception(
                    "Krylov approximation with estimated relative error "
                    + str(error_bound)
                    + " > 0.01 is too rough to be meaningful (try on lager graphs)"
                )
        else:
            self.ranks_power = personalization.np
            ranks.np = backend.repeat(0.0, backend.length(ranks.np))

    def _recursion(self, result, next_term, next_coefficient):
        if self.coefficient_type == "chebyshev":
            if self.convergence.iteration == 2:
                self.prev_term = next_term
            if self.convergence.iteration > 2:
                next_term = 2 * next_term - self.prev_term
                self.prev_term = next_term
                if self.coefficient == 0:
                    return result, next_term
            return result + next_term * next_coefficient, next_term
        elif self.coefficient_type == "taylor":
            if self.coefficient == 0:
                return result, next_term
            return result + next_term * next_coefficient, next_term
        else:
            raise Exception("Invalid coefficient type")

    def _prepare(self, personalization: GraphSignal):
        if self.optimization_dict is not None:
            personalization_id = obj2id(personalization)
            if personalization_id not in self.optimization_dict:
                self.optimization_dict[personalization_id] = dict()
            self.__active_dict = self.optimization_dict[personalization_id]
        else:
            self.__active_dict = None

    def _retrieve_power(self, ranks_power, M):
        if self.__active_dict is not None:
            if self.convergence.iteration not in self.__active_dict:
                self.__active_dict[self.convergence.iteration] = (
                    backend.conv(ranks_power, M)
                    if self.krylov_dims is None
                    else ranks_power @ M
                )
            return self.__active_dict[self.convergence.iteration]
        return (
            backend.conv(ranks_power, M)
            if self.krylov_dims is None
            else ranks_power @ M
        )

    def _step(self, M, personalization, ranks, *args, **kwargs):
        self.coefficient = self._coefficient(self.coefficient)
        if self.krylov_dims is not None:
            self.krylov_result, self.Mpower = self._recursion(
                self.krylov_result, self.Mpower, self.coefficient
            )
            ranks.np = krylov2original(
                self.krylov_base, self.krylov_result, int(self.krylov_dims)
            )
            self.Mpower = self._retrieve_power(self.Mpower, self.krylov_H)
        else:
            ranks.np, self.ranks_power = self._recursion(
                ranks.np, self.ranks_power, self.coefficient
            )
            self.ranks_power = self._retrieve_power(self.ranks_power, M)

    def _end(self, M, personalization, ranks, *args, **kwargs):
        if self.krylov_dims is not None:
            del self.krylov_base
            del self.krylov_H
        else:
            del self.ranks_power
        if self.coefficient_type == "chebyshev":
            del self.prev_term
        del self.coefficient
        del self.__active_dict

    def _coefficient(self, previous_coefficient: float) -> float:
        raise Exception(
            "Use a derived class of ClosedFormGraphFilter that implements the _coefficient method"
        )
