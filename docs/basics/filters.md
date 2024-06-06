# Graph Filters

Graph filters are algorithms that spread the node values stored in graph signals
through graphs by diffusing them through edges for several hops of different weights.
This process produces new graph signals. The original graph signal is called 
the *personalization*, and often
indicates the likelihood that nodes have a certain property, such as being members
of a structural or metadata community. Graph filters refine these
initial estimates by providing improved (probability) scores for all nodes.
Below we present a typical node ranking pipeline that starts from a known personalization,
applies a graph filter, potentially postprocesses its outcome, and eventually arrives at new node values.

<img src="../pipeline.png" alt="pipeline" style="width: 60%;">

Filters are created based on a constructor that takes as input several keyword
arguments affecting how they work. An exhaustive list of ready-to-use graph filters 
and their constructors
found [here](../generated/filters.md).
More complicated node ranking algorithms can be obtained by applying postprocessors on
filters. This is covered in the [next section](postprocessors.md).
After its initialization, a filter `alg` can run
with one of the following two patterns (these are interchangeable):

* `ranks = alg(graph, personalization)`
* `alg(pg.to_signal(graph, personalization))`


As an example, let us define an personalized PageRank filter. If the personalization is
binary (i.e. all nodes have initial scores either 0 or 1) this algorithm
is equivalent to a stochastic Markov process where it starts from the nodes
with initial scores 1, iteratively jumps to neighbors randomly, and has
a fixed probability *1-alpha* to restart. Node scores capture the probabilities 
of arriving at each node.

We use a restart probability at each step *1-alpha=0.01* and will
perform "col" (column-wise) normalization of the adjacency matrix to make
jumps to neighbors have equal probabilities (the alternative is "symmetric"
normalization, where the probabilities of moving between two nodes are the
same for both movement directions). We will also stop the algorithm at numerical
tolerance 1.E-9. Smaller tolerances are more accurate in exactly solving
each algorithm's exact outputs but take longer to converge.

```python
import pygrank as pg
algorithm = pg.PageRank(alpha=0.99, normalization="col", tol=1.E-9)
```

Having defined this algorithm, we will now use the graph `G` and graph signal
`signal` generated in the previous section. Passing these through the pipeline
while ignoring any postprocessing for the time being can be done as:

```python
scores = algorithm(graph, signal)
# Exception: ('Could not converge within 100 iterations')
```

The code threw an exception, because for alpha values near 1 and high tolerance
PageRank is slow to converge. Convergence speed is further reduced by the graph being
sparsely connected (this does not happen for graphs with higher average node
degrees). To address this issue, we can either set a laxer numerical
tolerance or simply provide a larger number of iterations the algorithm is allowed
to run for. For the sake of demonstration, we chose the second solution and allow
the algorithm to run for up to 2,000 iterations:

```python
algorithm = pg.PageRank(alpha=0.99, normalization="col", tol=1.E-9, max_iters=2000)
scores = algorithm(graph, signal)
print(scores)
# [('A', 0.25613418536078547), ('B', 0.12678642237010243), ('C', 0.2517487443382047), ('D', 0.24436832596280528), ('E', 0.12096232196810223)]
```

We can see that both 'A' and 'C' end up with the higher scores,
which are approximately 0.25. 'D' forms a circle with these
in the graph's structure and thus, by merit of being structurally close,
is scored closely to these two as 0.24. Finally, the other two nodes
assume lower values.

In the above code, we could also pass to the `rank` method
the dictionary `{'A':1, 'C': 2}` in place
of the signal and the package would make the conversion internally.
Alternatively, if a graph signal is already defined,
the graph could be omitted, as shown next. We stress that this is possible
only because the graph signal holds a reference to the graph it is tied to
and directly inputting other kinds of primitives would throw an error message.

```python
scores = algorithm(signal)
```

We now examine the structural relatedness of various nodes to the personalization:
```python
print(scores)
# [('A', 0.25613418536078547), ('B', 0.12678642237010243), ('C', 0.2517487443382047), ('D', 0.24436832596280528), ('E', 0.12096232196810223)]
```

!!! info
    Most filters are low-pass (they reduce graph eigenvalues) and thus smooth out the 
    personalization through the graph's structure.
