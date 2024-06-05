# Graph Preprocessing

Graph filters all use the same default graph normalization scheme
that performs symmetric (i.e. Laplacian-like) normalization 
for undirected graphs and column-wise normalization that
follows a true probabilistic formulation of transition probabilities
for directed graphs, such as `DiGraph` instances. The type of
normalization can be specified by passing a `normalization`
argument to constructors of ranking algorithms. This parameter can 
assume values of:
* *"auto"* for the above-described default behavior
* *"col"* for column-wise normalization
* *"symmetric"* for symmetric normalization
* *"none"* for avoiding any normalization, for example because edge weights already hold the normalization.

In combination to the above types of normalization, ranking
algorithms can be made to perform the renormalization trick
often employed by graph neural networks,
which shrinks their spectrum by adding self-loops to nodes
before extracting the adjacency matrix and its normalization.
To enable this behavior, you can use `renormalization=True`
alongside any other `normalization` argument.

In all cases, adjacency matrix normalization involves the
computationally intensive operation of converting the graph 
into a scipy sparse matrix each time  the `rank(G, personalization)`
method of ranking algorithms is called. The `pygrank` package
provides a way to avoid recomputing the normalization
during large-scale experiments by the same algorithm for 
the same graphs by passing an argument `assume_immutability=True`
to the algorithms's constructor, which indicates that
the the graph does not change between runs of the algorithm
and hence computes the normalization only once for each given
graph, a process known as hashing.

:warning: Hashing only uses the Python object's hash method, 
so a different instance of the same graph will recompute the 
normalization if it points at a different memory location.

:warning: Do not alter graph objects after passing them to
`rank(...)` methods of algorithms with
`assume_immutability=True` for the first time. If altering the
graph is necessary midway through your code, create a copy
instance with one of *networkx*'s in-built methods and
edit that one.

For example, hashing the outcome of graph normalization to
speed up multiple calls to the same graph can be achieved
as per the following code:
```python
import pygrank as pg
graph, personalization1, personalization2 = ...
algorithm = pg.PageRank(alpha=0.85, normalization="col", assume_immutability=True)
ranks1 = algorithm(graph, personalization1)
ranks2 = algorithm(graph, personalization2) # does not re-compute the normalization
```

Sometimes, many different algorithms are applied on the
same graph. In this case, to prevent each one
from recomputing the hashing already calculated by others,
they can be made to share the same normalization method. This 
can be done by using a shared instance of the 
normalization preprocessing `pg.preprocessor`, 
which can be passed as the `preprocessor` argument of ranking algorithm
constructors. In this case, the `normalization`, `renormalization` 
and `assume_immutability`
arguments should be passed to the preprocessor and will be ignored by the
constructors (what would otherwise happen is that the constructors
would create a prerpocessor with these arguments).

Basically, when the default value `preprocessor=None` is passed to ranking algorithm
constructors, these create a new preprocessing instance
with the `normalization`, `renormalization` and `assume_immutability`
values passed
to their constructor. These two arguments are completely ignored
if a preprocessor instance is passed to the ranking algorithm.
Direct use of these arguments without needing to instantiate a
preprocessor was demonstrated in the previous code example.

Using the same outcome of graph preprocessing 
to speed up multiple rank calls to the same graph by
different ranking algorithms can be done as:
```python
import pygrank as pg
graph, personalization1, personalization2 = ...
pre = pg.preprocessor(normalization="col", assume_immutability=True)
algorithm1 = pg.PageRank(alpha=0.85, preprocessor=pre)
algorithm2 = pg.HeatKernel(alpha=0.85, preprocessor=pre)
ranks1 = algorithm1(graph, personalization1)
ranks2 = algorithm2(graph, personalization2) # does not re-compute the normalization
```

:bulb: When benchmarking, in the above code you can call `pre(graph)`
before the first `rank(...)` call to make sure that that call
does not also perform the first normalization whose outcome will
be hashed and immediately retrieved by subsequent calls.
