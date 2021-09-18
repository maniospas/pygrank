# :scroll: List of Postprocessors
*This file is automatically generated with `docgenerator.py`.*

The following postprocessors can be imported from the package `pygrank.algorithms.postprocess`.
Constructor details are provided, including arguments inherited from and passed to parent classes.
All of them can be used through the code patterns presented at the library'personalization [documentation](documentation.md#postprocessors).  
1. [AdHocFairness](#postprocessor-adhocfairness)
2. [BoostedSeedOversampling](#postprocessor-boostedseedoversampling)
3. [FairPersonalizer](#postprocessor-fairpersonalizer)
4. [FairWalk](#postprocessor-fairwalk)
5. [MabsMaintain](#postprocessor-mabsmaintain)
6. [Normalize](#postprocessor-normalize)
7. [Ordinals](#postprocessor-ordinals)
8. [SeedOversampling](#postprocessor-seedoversampling)
9. [Sweep](#postprocessor-sweep)
10. [Tautology](#postprocessor-tautology)
11. [Threshold](#postprocessor-threshold)
12. [Transformer](#postprocessor-transformer)

### <kbd>Postprocessor</kbd> AdHocFairness

Adjusts node scores so that the sum of sensitive nodes is moved closer to the sum of non-sensitive ones based on 
ad hoc literature assumptions about how unfairness is propagated in graphs. 
Initializes the fairness-aware postprocessor. 

Args: 
 * *ranker:* The base ranking algorithm. 
 * *method:* The method with which to adjust weights. If "O" (default) an optimal gradual adjustment is performed [tsioutsiouliklis2020fairness]. If "B" node scores are weighted according to whether the nodes are sensitive, so that the sum of sensitive node scores becomes equal to the sum of non-sensitive node scores [tsioutsiouliklis2020fairness].  If "fairwalk" the graph is pre-processed so that, when possible, walks are equally probable to visit sensitive or non-sensitive nodes at non-restarting iterations [rahman2019fairwalk]. 
 * *eps:* A small value to consider rank redistribution to have converged. Default is 1.E-12. 

### <kbd>Postprocessor</kbd> BoostedSeedOversampling

Iteratively performs seed oversampling and combines found ranks by weighting them with a Boosting scheme. 
Initializes the class with a base ranker and the boosting scheme'personalization parameters. 

Attributes: 
 * *ranker:* The base ranker instance. 
 * *objective:* Optional. Can be either "partial" (default) or "naive". 
 * *oversample_from_iteration:* Optional. Can be either "previous" (default) to oversample the ranks of the previous iteration or "original" to always ovesample the given personalization. 
 * *weight_convergence:* Optional.  A ConvergenceManager that helps determine whether the weights placed on boosting iterations have converged. If None (default), initialized with ConvergenceManager(error_type=pyrgank.MaxDifference, tol=0.001, max_iters=100) 

Example:

```python 
>>> import pygrank as pg 
>>> graph, seed_nodes = ... 
>>> algorithm = pg.BoostedSeedOversampling(pg.PageRank(alpha=0.99)) 
>>> ranks = algorithm.rank(graph, personalization={1 for v in seed_nodes}) 
```


### <kbd>Postprocessor</kbd> FairPersonalizer

A personalization editing scheme that aims to edit graph signal priors (i.e. personalization) to produce 
disparate 
Instantiates a personalization editing scheme that trains towards optimizing 
retain_rank_weight*error_type(original scores, editing-induced scores) 
+ pRule_weight*min(induced score pRule, target_pRule) 

Args: 
 * *ranker:* The base ranking algorithm. 
 * *target_pRule:* Up to which value should pRule be improved. pRule values greater than this are not penalized further. 
 * *retain_rank_weight:* Can be used to penalize deviations from original posteriors due to editing. Use the default value 1 unless there is a specific reason to scale the error. Higher values correspond to tighter maintenance of original posteriors, but may not improve fairness as much. 
 * *pRule_weight:* Can be used to penalize low pRule values. Either use the default value 1 or, if you want to place most emphasis on pRule maximization (instead of trading-off between fairness and posterior preservation) 10 is a good empirical starting point. 
 * *error_type:* The supervised measure used to penalize deviations from original posterior scores. pygrank.KLDivergence (default) uses is used in [krasanakis2020prioredit]. pygrank.Error is used by the earlier [krasanakis2020fairconstr]. The latter does not induce fairness as well on average, but is sometimes better for specific graphs. 
 * *parameter_buckets:* How many sets of parameters to be used to . Default is 1. More parameters could be needed to to track, but running time scales **exponentially** to these (with base 4). 
 * *max_residual:* An upper limit on how much the original personalization is preserved, i.e. a fraction of it in the range [0, max_residual] is preserved. Default is 1 and is introduced by [krasanakis2020prioredit], but 0 can be used for exact replication of [krasanakis2020fairconstr]. 

### <kbd>Postprocessor</kbd> FairWalk
 

### <kbd>Postprocessor</kbd> MabsMaintain

Forces node ranking posteriors to have the same mean absolute value as prior inputs. 

### <kbd>Postprocessor</kbd> Normalize

Normalizes ranks by dividing with their maximal value. 
Initializes the class with a base ranker instance. Args are automatically filled in and 
re-ordered if at least one is provided. 

Args: 
 * *ranker:* The base ranker instance. A Tautology() ranker is created if None (default) was specified. 
 * *method:* Divide ranks either by their "max" (default) or by their "sum" or make the lie in the "range" [0,1] by subtracting their mean before diving by their max. 

Example:

```python 
>>> import pygrank as pg 
>>> graph, personalization, algorithm = ... 
>>> algorithm = pg.Normalize(0.5, algorithm) # sets ranks >= 0.5 to 1 and lower ones to 0 
>>> ranks = algorithm.rank(graph, personalization) 
```


Example (same outcome, simpler one-liner):

```python 
>>> import pygrank as pg 
>>> graph, personalization, algorithm = ... 
>>> ranks = pg.Normalize(0.5).transform(algorithm.rank(graph, personalization)) 
```


### <kbd>Postprocessor</kbd> Ordinals

Converts ranking outcome to ordinal numbers. 
The highest rank is set to 1, the second highest to 2, etc. 
Initializes the class with a base ranker instance. 

Args: 
 * *ranker:* Optional. The base ranker instance. A Tautology() ranker is created if None (default) was specified. 

Example:

```python 
>>> import pygrank as pg 
>>> graph, personalization, algorithm = ... 
>>> algorithm = pg.Ordinals(algorithm) 
>>> ranks = algorithm.rank(graph, personalization) 
```


Example (same outcome, simpler one-liner):

```python 
>>> import pygrank as pg 
>>> graph, personalization, algorithm = ... 
>>> ranks = pg.Ordinals().transform(algorithm.rank(graph, personalization)) 
```


### <kbd>Postprocessor</kbd> SeedOversampling

Performs seed oversampling on a base ranker to improve the quality of predicted seeds. 
Initializes the class with a base ranker. 

Attributes: 
 * *ranker:* The base ranker instance. 
 * *method:* Optional. Can be "safe" (default) to oversample based on the ranking scores of a preliminary base ranker run or "neighbors" to oversample the neighbors of personalization nodes. 

Example:

```python 
>>> import pygrank as pg 
>>> graph, seed_nodes = ... 
>>> algorithm = pg.SeedOversampling(pg.PageRank(alpha=0.99)) 
>>> ranks = algorithm.rank(graph, personalization={1 for v in seed_nodes}) 
```


### <kbd>Postprocessor</kbd> Sweep

Applies a sweep procedure that divides personalized node ranks by corresponding non-personalized ones. 
Initializes the sweep procedure. 

Args: 
 * *ranker:* The base ranker instance. 
 * *uniform_ranker:* Optional. The ranker instance used to perform non-personalized ranking. If None (default) the base ranker is used. 

Example:

```python 
>>> import pygrank as pg 
>>> graph, personalization, algorithm = ... 
>>> algorithm = pg.Sweep(algorithm) # divides node scores by uniform ranker'personalization non-personalized outcome 
>>> ranks = algorithm.rank(graph, personalization 
```


Example with different rankers:

```python 
>>> import pygrank as pg 
>>> graph, personalization, algorithm, uniform_ranker = ... 
>>> algorithm = pg.Sweep(algorithm, uniform_ranker=uniform_ranker) 
>>> ranks = algorithm.rank(graph, personalization) 
```


Example (same outcome):

```python 
>>> import pygrank as pg 
>>> graph, personalization, uniform_ranker, algorithm = ... 
>>> ranks = pg.Threshold(uniform_ranker).transform(algorithm.rank(graph, personalization)) 
```


### <kbd>Postprocessor</kbd> Tautology

Returns ranks as-are. 
Can be used as a baseline against which to compare other postprocessors. 

### <kbd>Postprocessor</kbd> Threshold

Converts ranking outcome to binary values based on a threshold value. 
Initializes the Threshold postprocessing scheme. Args are automatically filled in and 
re-ordered if at least one is provided. 

Args: 
 * *ranker:* Optional. The base ranker instance. A Tautology() ranker is created if None (default) was specified. 
 * *threshold:* Optional. The minimum numeric value required to output rank 1 instead of 0. If "gap" (default) then its value is automatically determined based on the maximal percentage increase between consecutive ranks. 

Example:

```python 
>>> import pygrank as pg 
>>> graph, personalization, algorithm = ... 
>>> algorithm = pg.Threshold(algorithm, 0.5) # sets ranks >= 0.5 to 1 and lower ones to 0 
>>> ranks = algorithm.rank(graph, personalization) 
```


Example (same outcome):

```python 
>>> import pygrank as pg 
>>> graph, personalization, algorithm = ... 
>>> ranks = pg.Threshold(0.5).transform(algorithm.rank(graph, personalization)) 
```


### <kbd>Postprocessor</kbd> Transformer

Applies an element-by-element transformation on a graph signal based on a given expression. 
Initializes the class with a base ranker instance. Args are automatically filled in and 
re-ordered if at least one is provided. 

Args: 
 * *ranker:* Optional. The base ranker instance. A Tautology() ranker is created if None (default) was specified. 
 * *expr:* Optional. A lambda expression to apply on each element. The transformer will automatically try to apply it on the backend array representation of the graph signal first, so prefer pygrank's backend functions for faster computations. For example, backend.exp (default) should be preferred instead of math.exp, because the former can directly parse numpy arrays, tensors, etc. 

Example:

```python 
>>> import pygrank as pg 
>>> graph, personalization, algorithm = ... 
>>> r1 = pg.Normalize(algorithm, "sum").rank(graph, personalization) 
>>> r2 = pg.Transformer(algorithm, lambda x: x/pg.sum(x)).rank(graph, personalization) 
>>> print(pg.Mabs(r1)(r2)) 
```
