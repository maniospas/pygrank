# Convergence

All graph filter constructors have a `convergence` argument that
indicates an object to help determine their convergence criteria, such as type of
error and tolerance for numerical convergence. If no such argument is passed
to the constructor, a `pygrank.ConvergenceManager` object
is automatically instantiated by borrowing whichever extra arguments it can
from those passed to algorithm constructors. These arguments can be:
- `tol` to indicate the numerical tolerance level required for convergence (default is 1.E-6).
- `error_type` to indicate how differences between two graph signals are computed. The default value is `pygrank.Mabs` but any other supervised [measure](#evaluation) that computes the differences between consecutive iterations can be used. The string "iters" can also be used to make the algorithm stop only when max_iters are reached (see below).
- `max_iters` to indicate the maximum number of iterations the algorithm can run for (default is 100). This quantity works as a safety net to guarantee algorithm termination. 

Sometimes, it suffices to reach a robust node rank order instead of precise 
values. To cover such cases we have implemented a different convergence criterion
``RankOrderConvergenceManager`` that stops 
at a robust node order [krasanakis2020stopping]. This criterion is specifically intended to be used with PageRank 
as the base ranking algorithm and needs to know that algorithm's diffusion
rate ``alpha``, which is passed as its first argument.

```python
import pygrank as pg

G, personalization = ...
alpha = 0.85
ordered_ranker = pg.PageRank(alpha=alpha, convergence=pg.RankOrderConvergenceManager(alpha))
ordered_ranker = pg.Ordinals(ordered_ranker)
ordered_ranks = ordered_ranker(G, personalization)
```

:bulb: Since the node order is more important than the specific rank values,
a post-processing step has been added throught the wrapping expression
``ordered_ranker = pg.Ordinals(ordered_ranker)`` to output rank order. 
