# Quickstart

## 1. Install and import
Install the library using `pip install pygrank` and import it. 
Construct a node ranking algorithm from a graph filter by incrementally 
applying postprocessors with the `>>` symbol. There are many components 
and parameters available, and [autotuning](autotuning.md) can help find good configurations.
In the example below there is a Heat Kernel filter that emphasizes diffusion
of initial node values to nodes `t` hops away. Additional parameters controlling
the graph's adjacency matrix normalization are provided, and two strategies to
improve the identification of structural communities are applied.

```python
import pygrank as pg

hk5 = pg.HeatKernel(t=5, normalization="symmetric", renormalize=True)  # a graph filter
hk5_advanced = hk5 >> pg.SeedOversampling() >> pg.Sweep() >> pg.Normalize("max") 
```

## 2. Load a graph and community
Automatically load a graph alongside a community of nodes with a shared attribute. 
To work with your own data instead of examples that run immediately, 
create a custom `networkx` graph to analyse. 
Run the algorithm to get a graph signal that maps nodes to scores indicating 
structural proximity to community members.

```python
_, graph, community = next(pg.load_datasets_one_community(["eucore"]))
personalization = {node: 1.0 for node in community}  # binary or stochastic membership, missing scores are zero

scores = hk5_advanced(graph, personalization)  # returns a dict-like pg.GraphSignal
print(scores)  # {'0': 0.3154503251398683, '1': 0.26661671252340463, '2': 0.03700150026429704, ... }
```

## 3. Evaluate
Evaluate the scores using a stochastic generalization of the unsupervised conductance measure.
This is the same as normal conductance but allows a stochastic interpretation of community
membership when results are not crisp binary values. You may also obtain a short sentence on how
to describe a constructed algorithm in scientific manuscripts. This includes several citations.
Instead of manually computing one measure, `pygrank` also
provides benchmarks that consider many datasets, measures, and algorithms.

```python
measure = pg.Conductance()  # an evaluation measure
pg.benchmark_print_line("My conductance", measure(scores))  # pretty print
print("Cite this algorithm as:", hk5_advanced.cite())
```
