# Autotuning

Beyond the ability to compare node ranking algorithms,
we provide the ability to automatically tune node ranking 
algorithms or select the best ones with respect to optimizing a measure
based on the graph and personalization at hand. This process is abstracted
through a `pygrank.Tuner` base class, which wraps
any kind of node ranking algorithm. Ideally, this would wrap end-product
algorithms.

!!! warning
    Tuners differ from benchmarks in that they select node ranking algorithms
    on-the-fly based on input data. They may overfit even with train-validation-test splits.

An exhaustive list of ready-to-use tuners can be found [here](../generated/tuners.md).
After initialization with the appropriate
parameters, these can run with the same pattern as other node ranking algorithms.
Tuner instances with default arguments use commonly seen base settings.
For example, the following code separates training and evaluation
data of a provided personalization signal and then uses a tuner that
by default creates a `GenericGraphFilter` instance with ten parameters.

```python
import pygrank as pg
graph, personalization = ...
training, evaluation = pg.split(pg.to_signal(graph, personalization, training_samples=0.5))
scores_pagerank = pg.PageRank()(graph, training)
scores_tuned = pg.ParameterTuner()(graph, training)
auc_pagerank = pg.AUC(evaluation, exclude=training).evaluate(scores_pagerank)
auc_tuned = pg.AUC(evaluation, exclude=training).evaluate(scores_tuned)
assert auc_pagerank <= auc_tuned
# True
```

Specific algorithms can also be tuned on specific parameter values, given
a method to instantiate the algorithm from a given set of parameters
(at worst, a lambda expression). For example, the following code defines and runs
a tuner with the same training personalization of the
previous example. The tuner finds the optimal alpha value of personalized
PageRank that optimizes NDCG (tuners optimize AUC be default if no measure is provided).

```python
import pygrank as pg
graph, personalization = ...
algorithm_from_params = lambda params: pg.PageRank(alpha=params[0])
scores_tuned = pg.ParameterTuner(algorithm_from_params, 
                                     max_vals=[0.99], 
                                     min_vals=[0.5],
                                     measure=pg.NDCG).tune(personalization)
```


Graph convolutions are the most computationally-intensive operations
node ranking algorithms employ, as their running time scales linearly with the 
number of network edges (instead of nodes). However, when tuners
aim to optimize algorithms involving graph filters extending the
`ClosedFormGraphFilter` class, graph filtering is decomposed into 
weighted sums of naturally occurring
Krylov space base elements {*M<sup>n</sup>p*, *n=0,1,...*}.
To speed up computation time (by many times in some settings) `pygrank`
provides the ability to save the generation of this Krylov space base
so that future runs do *not* recompute it, effectively removing the need
to perform graph convolutions all but once for each personalization.

!!! info
    This speedup can be applied outside of tuners too;
    explicitly pass a graph signal object to node ranking algorithms.

To enable this behavior, a dictionary needs to be passed to closed form
graph filter constructors through an `optimization_dict` argument.
In most cases, tuners are responsible for delegating additional arguments
to default algorithms and this can be achieved with the following code.

```python
graph, personalization = ...
optimization_dict = dict()
tuner = pg.ParameterTuner(error_type="iters", 
                              num_parameters=20,
                              max_iters=20,
                              optimization_dict=optimization_dict)
scores = tuner(graph, personalization)
```

!!! warning
    Similarly to the `assume_immutability=True` option
    for preprocessors, the optimization dictionary requires that graphs signals are not altered in
    the interim, although it is possible to clear signal values.
    Furthermore, using optimization dictionaries multiplies (e.g. at least doubles)
    the amount of used memory, which the system may run out of for large graphs.
    To remove allocated memory, keep a reference to the dictionary and clear
    it afterwards with `optimization_dict.clear()`.

!!! info
    The default algorithms constructed by tuners (if none are provided) use
    *pygrank.SelfClearDict* instead of a normal dictionary. This clears other entries when
    a new personalization is inserted, therefore avoiding memory bloat.
