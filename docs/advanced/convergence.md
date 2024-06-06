# Convergence

All graph filter constructors have a `convergence` argument that
indicates an object to help determine their convergence criteria, such as type of
error and tolerance for numerical convergence. If no such argument is passed
to the constructor, a `pygrank.ConvergenceManager` object
is automatically instantiated by borrowing whichever extra arguments it can
from those passed to algorithm constructors. These arguments can be:

- *tol:* Indicates the numerical tolerance level required for convergence (default is 1.E-6).
- *error_type:* Indicates how differences between two graph signals are computed. The default value is `pygrank.Mabs` but any other supervised [measure](../basics/evaluation.md) that computes the differences between consecutive iterations can be used. The string "iters" can also be used to make the algorithm stop only when max_iters are reached (see below).
- *max_iters:* Indicates the maximum number of iterations the algorithm can run for (default is 100). This quantity works as a safety net to guarantee algorithm termination. 

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
ranker = pg.PageRank(alpha=alpha, convergence=pg.RankOrderConvergenceManager(alpha))
ordered_ranker = ranker >> pg.Ordinals()
ordered_ranks = ordered_ranker(G, personalization)
```

!!! info
    Since the node order was deemed more important than the specific rank values,
    a postprocessing step was added. 



# Demo

As a quick start, let us construct a graph 
and a set of nodes. The graph's class can be
imported either from the `networkx` library or from
`pygrank` itself. The two are in large part interoperable
and both can be parsed by our algorithms.
But our implementation is tailored to graph signal
processing needs and thus tends to be faster and consume
only a fraction of the memory.

```python
from pygrank import Graph

graph = Graph()
graph.add_edge("A", "B")
graph.add_edge("B", "C")
graph.add_edge("C", "D")
graph.add_edge("D", "E")
graph.add_edge("A", "C")
graph.add_edge("C", "E")
graph.add_edge("B", "E")
seeds = {"A", "B"}
```

We now run a personalized PageRank
to score the structural relatedness of graph nodes to the ones of the given set.
First, let us import the library:

```python
import pygrank as pg
```

For instructional purposes,
we experiment with (personalized) *PageRank*
and make it output the node order of ranks.

```python
ranker = pg.PageRank(alpha=0.85, tol=1.E-6, normalization="auto") >> pg.Ordinals()
ranks = ranker(graph, {v: 1 for v in seeds})
```

How much time did it take for the base ranker to converge?
(Depends on backend and device characteristics.)

```python
print(ranker.convergence)
# 19 iterations (0.0021852000063518062 sec)
```

Since for this example only the node order is important,
we can use a different way to specify convergence:

```python
convergence = pg.RankOrderConvergenceManager(pagerank_alpha=0.85, confidence=0.98) 
early_stop_ranker = pg.PageRank(alpha=0.85, convergence=convergence) >> pg.Ordinals()
ordinals = early_stop_ranker(graph, {v: 1 for v in seeds})
print(early_stop_ranker.convergence)
# 2 iterations (0.0005241000035312027 sec)
print(ordinals["B"], ordinals["D"], ordinals["E"])
# 3.0 5.0 4.0
```

Close to the previous results at a fraction of the time! For large graphs,
most ordinals would be near the ideal ones. Note that convergence time 
does not take into account the time needed to preprocess graphs.
