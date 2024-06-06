# Postprocessors

Filter outcomes of graph often require additional processing steps, for example to perform
normalization, improve their quality, or apply fairness constraints.
Postptprocessors wrap base filters to improve their outcomes, and the result
node ranking algorithms are called as if they were still filters. The filters
can either be supplied to construcors, or the postprocessors may be initialized
from the rest of their arguments and applied onto filters afterwards with the 
functional chain pattern `algorithm = filter >> postprocessor`.


A list of ready-to-use postprocessors can be
found [here](../generated/postprocessors.md). Simpler ones perform
normalization, for example to enforce the maximal or the sum 
of node scores to be 1. There also exist thresholding schemes, which can be used
for binary community detection, as well as methods to make node
comparisons non-parametric by transforming scores to ordinalities.

More complex postprocessing mechanisms involve re-running the 
base filters with augmented personalization. This happens both for
seed oversampling postprocessors, which aim to augment node scores
by providing more example nodes, and for fairness-aware posteriors,
which aim to make node scores adhere to some fairness constraint, 
such as disparate impact.

Let us consider a simple toy scenario where we want the graph signal outputted
by a filter to always be normalized so that its largest node score is one. For
graph `G`, signal `signal` and filter `alg`, 
and can use the postprocessor `Normalize("max")`. For convenience, simpler postprocessors 
like this one supply a method to transform graph signals like so:

```python
scores = alg(graph, signal)
normalized_scores = pg.Normalize("max").transform(scores)
print(list(normalized_scores.items()))
# [('A', 1.0), ('B', 0.4950000024069947), ('C', 0.9828783455187619), ('D', 0.9540636897749238), ('E', 0.472261528845582)]
```

The pattern that works for **all** postprocessors
is to wrap base algorithms, like in the following equivalent example:


```python
nalg = alg >> pg.Normalize("max")  # also valid pg.Normalize("max", alg) or pg.Normalize(alg, "max")
nscores = nalg(graph, signal)
print(nscores)
# [('A', 1.0), ('B', 0.4950000024069947), ('C', 0.9828783455187619), ('D', 0.9540636897749238), ('E', 0.472261528845582)]
```

We can add more steps, such as
an element-wise exponential transformation of scores
before normalization:

```python
nealg = alg >> pg.Transformer(pf.exp) >> pg.Normalize("max")
nescores = nealg(graph, signal)
print(nescores)
# [('A', 1.0), ('B', 0.8786683440755908), ('C', 0.9956241609824301), ('D', 0.9883030876536782), ('E', 0.8735657648099558)]
```
