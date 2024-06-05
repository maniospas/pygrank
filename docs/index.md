# pygrank

Fast node ranking algorithms on large graphs.

## Quickstart

1.&nbsp;Install the library with `pip install pygrank`, import it,
and construct a node ranking algorithm
(incrementally apply postprocessors with `>>`). 
There are many components and parameters to find 
good configurations; [autotuning](advanced/autotuning.md) may be helpful.

```python
import pygrank as pg

hk5 = pg.HeatKernel(t=5, normalization="symmetric", renormalize=True)  # a graph filter
hk5_advanced = hk5 >> pg.SeedOversampling() >> pg.Sweep() >> pg.Normalize("max") 
```

2.&nbsp;Automatically load a graph and a community of nodes with some shared attribute. 
You can also use a `networkx` graph. 
Then run the algorithm to get a graph signal that maps nodes to scores, where scores indicate
structural proximity to community members.

```python
_, graph, community = next(pg.load_datasets_one_community(["EUCore"]))
personalization = {node: 1.0 for node in community}  # binary or stochastic membership, missing scores are zero

scores = hk5_advanced(graph, personalization)  # returns a dict-like pg.GraphSignal
print(scores)  # {'0': 0.3154503251398683, '1': 0.26661671252340463, '2': 0.03700150026429704, ... }
```

3.&nbsp;Evaluate scores; here we use a stochastic generalization of the unsupervised Conductance measure (that
can parse scores).

```python
measure = pg.Conductance()  # an evaluation measure
pg.benchmark_print_line("My conductance", measure(scores))  # pretty
print("Cite this algorithm as:", hk5_advanced.cite())
```


## Citation

Don't forget to cite all algorithms you are using. Find implemented papers [here](tips/citations.md).
This is the citation for `pygrank` only:

```
@article{krasanakis2022pygrank,
  author       = {Emmanouil Krasanakis, Symeon Papadopoulos, Ioannis Kompatsiaris, Andreas Symeonidis},
  title        = {pygrank: A Python Package for Graph Node Ranking},
  journal      = {SoftwareX},
  year         = 2022,
  month        = oct,
  doi          = {10.1016/j.softx.2022.101227},
  url          = {https://doi.org/10.1016/j.softx.2022.101227}
}
```