# Tuners
The following tuning mechanisms can be imported from the package `pygrank.algorithms.autotune`.
Constructor details are provided, including arguments inherited from and passed to parent classes.
 
## <span class="component">AlgorithmSelection</span>
<b class="parameters">Extends</b><br> *Tuner*<br><b class="parameters">About</b><br>
Selects the best among a list of node ranking algorithms by randomly masking the personalization and trying 
to reconstruct the masked portion. The constructor instantiates the tuning mechanism. 
<br><b class="parameters">Parameters</b>

 * *rankers:* An iterable of node ranking algorithms to chose from. Try to make them share a preprocessor for more efficient computations. If None (default), the filters obtained from pygrank.benchmark.create_demo_filters().values() are used instead. 
 * *measure:* Callable to constuct a supervised measure with given known node scores and an iterable of excluded scores. 
 * *fraction_of_training:* A number in (0,1) indicating how to split provided graph signals into training and validaton ones by randomly sampling training nodes to meet the required fraction of all graph nodes. Numbers outside this range can also be used (not recommended without specific reason) per the conventions of `pygrank.split(...)`. Default is 0.8. 
 * *combined_prediction:* If True (default), after the best version of algorithms is determined, the whole personalization is used to produce the end-result. Otherwise, only the training portion of the training-validation split is used. 
 * *tuning_backend:* Specifically switches to a designated backend for the tuning process before restoring the previous one to perform the actual ranking. If None (default), this functionality is ignored. 

<b class="parameters">Example</b>
```python 
import pygrank as pg 
graph, personalization = ... 
tuner = pg.AlgorithmSelection(pg.create_demo_filters().values(), measure=pg.AUC, deviation_tol=0.01) 
ranks = tuner.rank(graph, personalization) 
```
Example (with more filters):
```python 
import pygrank as pg 
graph, personalization = ... 
algorithms = pg.create_variations(pg.create_many_filters(tol=1.E-9), pg.create_many_variation_types()) 
tuner = pg.AlgorithmSelection(algorithms.values(), measure=pg.AUC, deviation_tol=0.01) 
ranks = tuner.rank(graph, personalization) 
```
## <span class="component">HopTuner</span>
<b class="parameters">Extends</b><br> *Tuner*<br><b class="parameters">About</b><br>
Tunes a GenericGraphFilter specific measure by splitting the personalization 
in training and test sets and measuring the similarity of hops at given number of steps 
away. <br>:warning: **This is an experimental approach.**<br> The constructor instantiates the tuning mechanism. 
<br><b class="parameters">Parameters</b>

 * *ranker_generator:* A callable that constructs a ranker based on a list of parameters. If None (default) then a pygrank.algorithms.learnable.GenericGraphFilter is constructed with automatic normalization and assuming immutability (this is the most common setting). These parameters can be overriden and other ones can be passed to the algorithm'personalization constructor simply by including them in kwargs. 
 * *measure:* Callable to constuct a supervised measure with given known node scores. 
 * *basis:* Can use either the "Krylov" or the "Arnoldi" orthonormal basis of the krylov space. The latter does not produce a ranking algorithm. 
 * *tuning_backend:* Specifically switches to a designted backend for the tuning process before restoring the previous one to perform the actual ranking. If None (default), this functionality is ignored. 
 * *tunable_offset:* If None, no offset is added to estimated parameters. Otherwise, a supervised measure generator (e.g. a supervised measure class) can be passed. Default is `pygrank.AUC`. 
 * *kwargs:* Additional arguments are passed to the automatically instantiated GenericGraphFilter. 

<b class="parameters">Example</b>
```python 
import pygrank as pg 
graph, personalization = ... 
tuner = pg.HopTuner(measure=AUC) 
ranks = tuner.rank(graph, personalization) 
```
## <span class="component">ParameterTuner</span>
<b class="parameters">Extends</b><br> *Tuner*<br><b class="parameters">About</b><br>
Tunes a parameterized version of node ranking algorithms under a specific measure by splitting the personalization 
in training and test sets. The tuning mechanism is fully described in [krasanakis2022autogf]. The constructor instantiates the tuning mechanism. 
<br><b class="parameters">Parameters</b>

 * *ranker_generator:* A callable that constructs a ranker based on a list of parameters. If None (default) then a pygrank.algorithms.learnable.GenericGraphFilter is constructed with automatic normalization and assuming immutability (this is the most common setting). These parameters can be overriden and other ones can be passed to the algorithm'personalization constructor by including them in kwargs. 
 * *measure:* Callable to constuct a supervised measure with given known node scores and an iterable of excluded scores. 
 * *fraction_of_training:* A number in (0,1) indicating how to split provided graph signals into training and validaton ones by randomly sampling training nodes to meet the required fraction of all graph nodes. Numbers outside this range can also be used (not recommended without specific reason) per the conventions of `pygrank.split(...)`. Default is 0.5. 
 * *cross_validate:* Averages the optimal parameters along a specified number of validation splits. Default is 1. 
 * *combined_prediction:* If True (default), after the best version of algorithms is determined, the whole personalization is used to produce the end-result. Otherwise, only the training portion of the training-validation split is used. 
 * *tuning_backend:* Specifically switches to a designted backend for the tuning process before restoring the previous one to perform the actual ranking. If None (default), this functionality is ignored. 
 * *optimizer:* The optimizer of choice to use. Default is `pygrank.algorithms.autotune.optimization.optimize`, but other methods can be used such as Default is `pygrank.algorithms.autotune.optimization.evolutionary_optimizer`. Parameters to the optimizer need to be passed via kwargs. 
 * *kwargs:* Additional arguments can be passed to pygrank.algorithms.autotune.optimization.optimize. Otherwise, the respective arguments are retrieved from the variable *default_tuning_optimization*, which is crafted for fast convergence of the default ranker_generator. Arguments passable to the ranker_generator are also passed to it. Make sure to declare both the upper **and** the lower bounds of parameter values. 

<b class="parameters">Example</b>
```python 
import pygrank as pg 
graph, personalization = ... 
tuner = pg.ParameterTuner(measure=AUC, deviation_tol=0.01) 
ranks = tuner.rank(graph, personalization) 
```
Example to tune pagerank'personalization float parameter alpha in the range [0.5, 0.99]:
```python 
import pygrank as pg 
graph, personalization = ... 
tuner = pg.ParameterTuner(lambda params: pg.PageRank(alpha=params[0]), 
measure=AUC, deviation_tol=0.01, max_vals=[0.99], min_vals=[0.5]) 
ranks = tuner.rank(graph, personalization) 
```
