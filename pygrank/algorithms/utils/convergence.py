import numpy as np
from scipy.stats import norm
from timeit import default_timer as time
from pygrank.measures import Supervised, Mabs
from pygrank.core import backend


class ConvergenceManager:
    """ Used to keep previous iteration and generally manage convergence of variables. Graph filters
    automatically create instances of this class by passing on appropriate parameters.

    Examples:
        >>> convergence = ConvergenceManager()
        >>> convergence.start()
        >>> var = None
        >>> while not convergence.has_converged(var):
        >>>     ...
        >>>     var = ...
    """

    def __init__(self, tol=1.E-6, error_type=Mabs, max_iters=100):
        """
        Initializes a convergence manager with a provided tolerance level, error type and number of iterations.

        Args:
            tol: Numerical tolerance to determine the stopping point (algorithms stop if the "error" between
                consecutive iterations becomes less than this number). Default is 1.E-6 but for large graphs
                1.E-9 often yields more robust convergence points. If the provided value is less than the
                numerical precision of the backend `pygrank.epsilon()` then it is snapped to that value.
            error_type: Optional. How to calculate the "error" between consecutive iterations of graph signals.
                If "iters", convergence is reached at iteration *max_iters*-1 without throwing an exception.
                Default is `pygrank.Mabs`.
            max_iters: Optional. The number of iterations algorithms can run for. If this number is exceeded,
                an exception is thrown. This could help manage computational resources. Default value is 100,
                and exceeding this value with graph filters often indicates that either graphs have large diameters
                or that algorithms of choice converge particularly slowly.
        """
        self.tol = max(tol, backend.epsilon())
        self.error_type = error_type
        self.max_iters = max_iters
        self.iteration = 0
        self.last_ranks = None
        self._start_time = None
        self.elapsed_time = None

    def start(self, restart_timer=True):
        """
        Starts the convergence manager

        Args:
            restart_timer: Optional. If True (default) timing information, such as the number of iterations and wall
                clock time measurement, is reset. Otherwise, this only ensures that the convergence manager
                performs one iteration before starting comparing values with previous ones.
        """
        if restart_timer or self._start_time is None:
            self._start_time = time()
            self.elapsed_time = None
            self.iteration = 0
        self.last_ranks = None

    def has_converged(self, new_ranks):
        """
        Checks whether convergence has been achieved by comparing this iteration's backend array with the
        previous iteration's.

        Args:
            new_ranks: The iteration's backend array.
        """
        self.iteration += 1
        if self.iteration >= self.max_iters:
            if self.error_type == "iters":
                self.elapsed_time = time()-self._start_time
                return True
            raise Exception("Could not converge within "+str(self.max_iters)+" iterations")
        converged = False if self.last_ranks is None else self._has_converged(self.last_ranks, new_ranks)
        self.last_ranks = new_ranks
        self.elapsed_time = time()-self._start_time
        return converged

    def _has_converged(self, prev_ranks, ranks):
        if self.error_type == "iters":
            return False
        return self.error_type(prev_ranks)(ranks) <= self.tol

    def __str__(self):
        return str(self.iteration)+" iterations ("+str(self.elapsed_time)+" sec)"


class RankOrderConvergenceManager:
    def __init__(self, pagerank_alpha, confidence=0.98, criterion="rank_gap"):
        self.iteration = 0
        self._start_time = None
        self.elapsed_time = None
        self.accumulated_ranks = None
        self.pagerank_alpha = pagerank_alpha
        self.confidence = confidence
        self.criterion = criterion

    def start(self, restart_timer=True):
        if restart_timer or self._start_time is None:
            self._start_time = time()
            self.elapsed_time = None
            self.iteration = 0
            self.accumulated_ranks = 0

    def has_converged(self, new_ranks):
        new_ranks = np.array(new_ranks).squeeze()
        self.accumulated_ranks = (self.accumulated_ranks*self.iteration + new_ranks) / (self.iteration+1)
        self.iteration += 1
        converged = self.current_fraction_of_random_walks() >= self.needed_fraction_of_random_walks(new_ranks)
        self.elapsed_time = time()-self._start_time
        return converged

    def needed_fraction_of_random_walks(self, ranks):
        if self.criterion == "rank_gap":
            a = [rank for rank in ranks]
            order = np.argsort(a, kind='quicksort')
            gaps = [a[order[i + 1]] - a[order[i]] for i in range(len(order)-1) if a[order[i + 1]] != a[order[i]]]
            if len(gaps) < 2:
                return 1
            return 1-(max(gaps)-min(gaps)) / (norm.ppf(self.confidence) * np.std(gaps)*len(gaps))
        elif self.criterion == "fraction_of_walks":
            return self.confidence
        else:
            raise Exception("criterion can only be 'rank_gap' or 'fraction_of_walks'")

    def current_fraction_of_random_walks(self):
        sup_of_series_sum = -np.log(1 - self.pagerank_alpha)
        series_sum = 0
        power = 1
        for n in range(1, self.iteration+1):
            power *= self.pagerank_alpha
            series_sum += power / n  # this is faster than np.power(self.pagerank_alpha, n) / n
        return series_sum / sup_of_series_sum

    def __str__(self):
        return str(self.iteration)+" iterations ("+str(self.elapsed_time)+" sec)"
