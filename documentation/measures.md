# :scroll: List of Measures
*This file is automatically generated with `docgenerator.py`.*

The following measures can be imported from the package `pygrank.measures`.
Constructor details are provided, including arguments inherited from and passed to parent classes.
All of them can be used through the code patterns presented at the library'personalization [documentation](documentation.md#evaluation).  
1. [Time](#measure-time)
2. [AM](#measurecombination-am)
3. [GM](#measurecombination-gm)
4. [AUC](#supervised-auc)
5. [Accuracy](#supervised-accuracy)
6. [Cos](#supervised-cos)
7. [CrossEntropy](#supervised-crossentropy)
8. [Dot](#supervised-dot)
9. [KLDivergence](#supervised-kldivergence)
10. [MKLDivergence](#supervised-mkldivergence)
11. [Mabs](#supervised-mabs)
12. [MannWhitneyParity](#supervised-mannwhitneyparity)
13. [MaxDifference](#supervised-maxdifference)
14. [NDCG](#supervised-ndcg)
15. [PearsonCorrelation](#supervised-pearsoncorrelation)
16. [SpearmanCorrelation](#supervised-spearmancorrelation)
17. [pRule](#supervised-prule)
18. [Conductance](#unsupervised-conductance)
19. [Density](#unsupervised-density)
20. [Modularity](#unsupervised-modularity)

### <kbd>Measure</kbd> Time

An abstract class that can be passed to benchmark experiments to indicate that they should report running time 
of algorithms. Instances of this class have no functionality. 

### <kbd>MeasureCombination</kbd> AM

Combines several measures through their arithmetic mean. 
Instantiates a combination of several measures. More measures with their own weights and threhsolded range 
can be added with the `add(measure, weight=1, min_val=-inf, max_val=inf)` method. 

Args: 
 * *measures:* Optional. An iterable of measures to combine. If None (default) no new measure is added. 
 * *weights:* Optional. A iterable of floats with which to weight the measures provided by the previous argument. The concept of weighting depends on how measures are aggregated, but it corresponds to an importance value placed on each measure. If None (default), provided measures are all weighted by 1. 
 * *thresholds:* Optional. A tuple of [min_val, max_val] with which to bound measure outcomes. If None (default) provided measures 

### <kbd>MeasureCombination</kbd> GM

Combines several measures through their geometric mean. 
Instantiates a combination of several measures. More measures with their own weights and threhsolded range 
can be added with the `add(measure, weight=1, min_val=-inf, max_val=inf)` method. 

Args: 
 * *measures:* Optional. An iterable of measures to combine. If None (default) no new measure is added. 
 * *weights:* Optional. A iterable of floats with which to weight the measures provided by the previous argument. The concept of weighting depends on how measures are aggregated, but it corresponds to an importance value placed on each measure. If None (default), provided measures are all weighted by 1. 
 * *thresholds:* Optional. A tuple of [min_val, max_val] with which to bound measure outcomes. If None (default) provided measures 

### <kbd>Supervised</kbd> AUC

Wrapper for sklearn.metrics.auc evaluation. 
Initializes the supervised measure with desired graph signal outcomes. 

Args: 
 * *known_ranks:* The desired graph signal outcomes. 
 * *exclude:* Optional. An iterable (e.g. list, map, networkx graph, graph signal) whose items/keys are traversed to determine which nodes to ommit from the evaluation, for example because they were used for training. If None (default) the measure is evaluated on all graph nodes. You can safely set the `self.exclude` property at any time to alter this original value. Prefer using this behavior to avoid overfitting measure assessments. 

### <kbd>Supervised</kbd> Accuracy

Computes the accuracy as 1- mean absolute differences between given and known ranks. 
Initializes the supervised measure with desired graph signal outcomes. 

Args: 
 * *known_ranks:* The desired graph signal outcomes. 
 * *exclude:* Optional. An iterable (e.g. list, map, networkx graph, graph signal) whose items/keys are traversed to determine which nodes to ommit from the evaluation, for example because they were used for training. If None (default) the measure is evaluated on all graph nodes. You can safely set the `self.exclude` property at any time to alter this original value. Prefer using this behavior to avoid overfitting measure assessments. 

### <kbd>Supervised</kbd> Cos

Computes the cosine similarity between given and known ranks 
Initializes the supervised measure with desired graph signal outcomes. 

Args: 
 * *known_ranks:* The desired graph signal outcomes. 
 * *exclude:* Optional. An iterable (e.g. list, map, networkx graph, graph signal) whose items/keys are traversed to determine which nodes to ommit from the evaluation, for example because they were used for training. If None (default) the measure is evaluated on all graph nodes. You can safely set the `self.exclude` property at any time to alter this original value. Prefer using this behavior to avoid overfitting measure assessments. 

### <kbd>Supervised</kbd> CrossEntropy

Computes a cross-entropy loss of given vs known ranks. 
Initializes the supervised measure with desired graph signal outcomes. 

Args: 
 * *known_ranks:* The desired graph signal outcomes. 
 * *exclude:* Optional. An iterable (e.g. list, map, networkx graph, graph signal) whose items/keys are traversed to determine which nodes to ommit from the evaluation, for example because they were used for training. If None (default) the measure is evaluated on all graph nodes. You can safely set the `self.exclude` property at any time to alter this original value. Prefer using this behavior to avoid overfitting measure assessments. 

### <kbd>Supervised</kbd> Dot

Computes the dot similarity between given and known ranks 
Initializes the supervised measure with desired graph signal outcomes. 

Args: 
 * *known_ranks:* The desired graph signal outcomes. 
 * *exclude:* Optional. An iterable (e.g. list, map, networkx graph, graph signal) whose items/keys are traversed to determine which nodes to ommit from the evaluation, for example because they were used for training. If None (default) the measure is evaluated on all graph nodes. You can safely set the `self.exclude` property at any time to alter this original value. Prefer using this behavior to avoid overfitting measure assessments. 

### <kbd>Supervised</kbd> KLDivergence

Computes the KL-divergence of given vs known ranks. 
Initializes the supervised measure with desired graph signal outcomes. 

Args: 
 * *known_ranks:* The desired graph signal outcomes. 
 * *exclude:* Optional. An iterable (e.g. list, map, networkx graph, graph signal) whose items/keys are traversed to determine which nodes to ommit from the evaluation, for example because they were used for training. If None (default) the measure is evaluated on all graph nodes. You can safely set the `self.exclude` property at any time to alter this original value. Prefer using this behavior to avoid overfitting measure assessments. 

### <kbd>Supervised</kbd> MKLDivergence

Computes the KL-divergence of given vs known ranks. 
Initializes the supervised measure with desired graph signal outcomes. 

Args: 
 * *known_ranks:* The desired graph signal outcomes. 
 * *exclude:* Optional. An iterable (e.g. list, map, networkx graph, graph signal) whose items/keys are traversed to determine which nodes to ommit from the evaluation, for example because they were used for training. If None (default) the measure is evaluated on all graph nodes. You can safely set the `self.exclude` property at any time to alter this original value. Prefer using this behavior to avoid overfitting measure assessments. 

### <kbd>Supervised</kbd> Mabs

Computes the mean absolute error between ranks and known ranks. 
Initializes the supervised measure with desired graph signal outcomes. 

Args: 
 * *known_ranks:* The desired graph signal outcomes. 
 * *exclude:* Optional. An iterable (e.g. list, map, networkx graph, graph signal) whose items/keys are traversed to determine which nodes to ommit from the evaluation, for example because they were used for training. If None (default) the measure is evaluated on all graph nodes. You can safely set the `self.exclude` property at any time to alter this original value. Prefer using this behavior to avoid overfitting measure assessments. 

### <kbd>Supervised</kbd> MannWhitneyParity

Performs two-tailed Mann-Whitney U-test to check that the scores of sensitive-attributed nodes do not exhibit 
higher or lower values compared to the rest. To do this, the test's U statistic is transformed so that value 
1 indicate that the probability of sensitive-attributed nodes exhibiting higher values is the same as 
for lower values (50%). Value 0 indicates that either the probability of exhibiting only higher or only lower 
values is 100%. 
Known ranks correspond to the binary sensitive attribute checking whether nodes are sensitive. 
Initializes the supervised measure with desired graph signal outcomes. 

Args: 
 * *known_ranks:* The desired graph signal outcomes. 
 * *exclude:* Optional. An iterable (e.g. list, map, networkx graph, graph signal) whose items/keys are traversed to determine which nodes to ommit from the evaluation, for example because they were used for training. If None (default) the measure is evaluated on all graph nodes. You can safely set the `self.exclude` property at any time to alter this original value. Prefer using this behavior to avoid overfitting measure assessments. 

### <kbd>Supervised</kbd> MaxDifference

Computes the maximum absolute error between ranks and known ranks. 
Initializes the supervised measure with desired graph signal outcomes. 

Args: 
 * *known_ranks:* The desired graph signal outcomes. 
 * *exclude:* Optional. An iterable (e.g. list, map, networkx graph, graph signal) whose items/keys are traversed to determine which nodes to ommit from the evaluation, for example because they were used for training. If None (default) the measure is evaluated on all graph nodes. You can safely set the `self.exclude` property at any time to alter this original value. Prefer using this behavior to avoid overfitting measure assessments. 

### <kbd>Supervised</kbd> NDCG

Provides evaluation of NDCG@k score between given and known ranks. 
Initializes the PageRank scheme parameters. 

Args: 
 * *k:* Optional. Calculates NDCG@k. If None (default), len(known_ranks) is used. 
 * *known_ranks:* The desired graph signal outcomes. 
 * *exclude:* Optional. An iterable (e.g. list, map, networkx graph, graph signal) whose items/keys are traversed to determine which nodes to ommit from the evaluation, for example because they were used for training. If None (default) the measure is evaluated on all graph nodes. You can safely set the `self.exclude` property at any time to alter this original value. Prefer using this behavior to avoid overfitting measure assessments. 

### <kbd>Supervised</kbd> PearsonCorrelation

Computes the Pearson correlation coefficient between given and known ranks. 
Initializes the supervised measure with desired graph signal outcomes. 

Args: 
 * *known_ranks:* The desired graph signal outcomes. 
 * *exclude:* Optional. An iterable (e.g. list, map, networkx graph, graph signal) whose items/keys are traversed to determine which nodes to ommit from the evaluation, for example because they were used for training. If None (default) the measure is evaluated on all graph nodes. You can safely set the `self.exclude` property at any time to alter this original value. Prefer using this behavior to avoid overfitting measure assessments. 

### <kbd>Supervised</kbd> SpearmanCorrelation

Computes the Spearman correlation coefficient between given and known ranks. 
Initializes the supervised measure with desired graph signal outcomes. 

Args: 
 * *known_ranks:* The desired graph signal outcomes. 
 * *exclude:* Optional. An iterable (e.g. list, map, networkx graph, graph signal) whose items/keys are traversed to determine which nodes to ommit from the evaluation, for example because they were used for training. If None (default) the measure is evaluated on all graph nodes. You can safely set the `self.exclude` property at any time to alter this original value. Prefer using this behavior to avoid overfitting measure assessments. 

### <kbd>Supervised</kbd> pRule

Computes an assessment of stochastic ranking fairness by obtaining the fractional comparison of average scores 
between sensitive-attributed nodes and the rest the rest. 
Values near 1 indicate full fairness (statistical parity), whereas lower values indicate disparate impact. 
Known ranks correspond to the binary sensitive attribute checking whether nodes are sensitive. 
Usually, pRule > 80% is considered fair. 
Initializes the supervised measure with desired graph signal outcomes. 

Args: 
 * *known_ranks:* The desired graph signal outcomes. 
 * *exclude:* Optional. An iterable (e.g. list, map, networkx graph, graph signal) whose items/keys are traversed to determine which nodes to ommit from the evaluation, for example because they were used for training. If None (default) the measure is evaluated on all graph nodes. You can safely set the `self.exclude` property at any time to alter this original value. Prefer using this behavior to avoid overfitting measure assessments. 

### <kbd>Unsupervised</kbd> Conductance

Graph conductance (information flow) of ranks. 
Assumes a fuzzy set of subgraphs whose nodes are included with probability proportional to their ranks, 
as per the formulation of [krasanakis2019linkauc] and calculates E[outgoing edges] / E[internal edges] of 
the fuzzy rank subgraph. 
If ranks assume binary values, E[.] becomes set size and this calculates the induced subgraph Conductance. 
Initializes the Conductance measure. 

Args: 
 * *graph:* Optional. The graph on which to calculate the measure. If None (default) it is automatically extracted from graph signals passed for evaluation. 
 * *max_rank:* Optional. The maximum value ranks can assume. To maintain a probabilistic formulation of conductance, this can be greater but not less than the maximum rank during evaluation. Default is 1. 

Example:

```python 
>>> import pygrank as pg 
>>> graph, seed_nodes, algorithm = ... 
>>> algorithm = pg.Normalize(algorithm) 
>>> ranks = algorithm.rank(graph, seed_nodes) 
>>> conductance = pg.Conductance().evaluate(ranks) 
```


### <kbd>Unsupervised</kbd> Density

Extension of graph density that can account for ranks. 
Assumes a fuzzy set of subgraphs whose nodes are included with probability proportional to their ranks, 
as per the formulation of [krasanakis2019linkauc] and calculates E[internal edges] / E[possible edges] of 
the fuzzy rank subgraph. 
If ranks assume binary values, E[.] becomes set size and this calculates the induced subgraph Density. 
Initializes the Density measure. 

Args: 
 * *graph:* Optional. The graph on which to calculate the measure. If None (default) it is automatically extracted from graph signals passed for evaluation. 

Example:

```python 
>>> import pygrank as pg 
>>> graph, seed_nodes, algorithm = ... 
>>> ranks = algorithm.rank(graph, seed_nodes) 
>>> conductance = pg.Density().evaluate(ranks) 
```


### <kbd>Unsupervised</kbd> Modularity
 