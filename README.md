# pygrank
Recommendation algorithms for large graphs.

## Installation
```
pip install pygrank
```

## Usage
###### Run a PageRank algorithm with seed oversampling
```python
import networkx as nx
from pygrank.algorithms.pagerank import PageRank as Ranker
from pygrank.algorithms.oversampling import SeedOversampling as Oversampler

G = nx.Graph()
seeds = list()
... # insert graph nodes and select some of them as seeds (e.g. see tests.py)

algorithm = Oversampler(Ranker(alpha=0.85, tol=1.E-6, max_iters=100)) # these are the default values
ranks = algorithm.rank(G, {v: 1 for v in seeds})
```

###### Run a PageRank algorithm and make it converge to a robust node order
```python
import networkx as nx
from pygrank.algorithms.pagerank import PageRank as Ranker
from pygrank.algorithms.utils import RankOrderConvergenceManager

G = nx.Graph()
seeds = list()
... # insert graph nodes and select some of them as seeds (e.g. see tests.py)
alpha = 0.85

algorithm = Ranker(alpha=alpha, convergence=RankOrderConvergenceManager(alpha))
ranks = algorithm.rank(G, {v: 1 for v in seeds})
```

###### Hash the outcome of graph normalization to speed up multiple calls to the same graph
```python
import networkx as nx
from pygrank.algorithms.pagerank import PageRank as Ranker
from pygrank.algorithms.utils import preprocessor

G = nx.Graph()
seeds1 = list()
seeds2 = list()
... # insert graph nodes and select some of them as seeds (e.g. see tests.py)

algorithm = Ranker(alpha=0.8, to_scipy=preprocessor(normalization="col", assume_immutability=True))
ranks = algorithm.rank(G, {v: 1 for v in seeds1})
ranks = algorithm.rank(G, {v: 1 for v in seeds2}) # does not re-compute the normalization
```

###### How to evaluate with an unsupervised metric
```python
from pygrank.algorithms.postprocess import Normalize
from pygrank.metrics.unsupervised import Conductance

G, ranks = ... # calculate as per the first example
normalized_ranks = Normalize().rank(ranks)

metric = Conductance(G)
print(metric.evaluate(normalized_ranks))
```

###### How to evaluate with a supervised metric
```python
from pygrank.metrics.supervised import AUC
import pygrank.metrics.utils

G, seeds, algorithm = ... # as per the first example
seeds, ground_truth = pygrank.metrics.utils.split_groups(seeds, fraction_of_training=0.5)

pygrank.metrics.utils.remove_group_edges_from_graph(G, ground_truth)
ranks = algorithm.rank(G, {v: 1 for v in seeds})

metric = AUC({v: 1 for v in ground_truth})
print(metric.evaluate(ranks))
```

###### How to evaluate multiple ranks
```python
import networkx as nx
from pygrank.algorithms.pagerank import PageRank as Ranker
from pygrank.algorithms.postprocess import Normalize as Normalizer
from pygrank.algorithms.oversampling import BoostedSeedOversampling as Oversampler
from pygrank.metrics.unsupervised import Conductance
from pygrank.metrics.supervised import AUC
from pygrank.metrics.multigroup import MultiUnsupervised, MultiSupervised, LinkAUC
import pygrank.metrics.utils

# Construct data
G = nx.Graph()
groups = {}
groups["group1"] = list()
... 

# Split to training and test data
training_groups, test_groups = pygrank.metrics.utils.split_groups(groups)
pygrank.metrics.utils.remove_group_edges_from_graph(G, test_groups)

# Calculate ranks and put them in a map
algorithm = Normalizer(Oversampler(Ranker(alpha=0.99)))
ranks = {group_id: algorithm.rank(G, {v: 1 for v in group}) 
        for group_id, group in training_groups.items()}
        

# Evaluation with Conductance
conductance = MultiUnsupervised(Conductance, G)
print(conductance.evaluate(ranks))

# Evaluation with LinkAUC
link_AUC = LinkAUC(G, pygrank.metrics.utils.to_nodes(test_groups))
print(link_AUC.evaluate(ranks))

# Evaluation with AUC
auc = MultiSupervised(AUC, pygrank.metrics.utils.to_seeds(test_groups))
print(auc.evaluate(ranks))
        
```


## References
```
@article{krasanakis2020unsupervised,
  title={Unsupervised evaluation of multiple node ranks by reconstructing local structures},
  author={Krasanakis, Emmanouil and Papadopoulos, Symeon and Kompatsiaris, Yiannis},
  journal={Applied Network Science},
  volume={5},
  number={1},
  pages={1--32},
  year={2020},
  publisher={Springer}
}
```
```
@unpublished{krasanakis2020stopping,
  title={Stopping Personalized PageRank without an Error Tolerance Parameter},
  author={Krasanakis, Emmanouil and Papadopoulos, Symeon and Kompatsiaris, Ioannis},
  year={2020},
  note = {unpublished}
}
```
```
@article{krasanakis2019boosted,
  title={Boosted seed oversampling for local community ranking},
  author={Krasanakis, Emmanouil and Schinas, Emmanouil and Papadopoulos, Symeon and Kompatsiaris, Yiannis and Symeonidis, Andreas},
  journal={Information Processing \& Management},
  pages={102053},
  year={2019},
  publisher={Elsevier}
}
```
```
@inproceedings{krasanakis2019linkauc,
  title={LinkAUC: Unsupervised Evaluation of Multiple Network Node Ranks Using Link Prediction},
  author={Krasanakis, Emmanouil and Papadopoulos, Symeon and Kompatsiaris, Yiannis},
  booktitle={International Conference on Complex Networks and Their Applications},
  pages={3--14},
  year={2019},
  organization={Springer}
}
```
```
@inproceedings{krasanakis2018venuerank,
  title={VenueRank: Identifying Venues that Contribute to Artist Popularity.},
  author={Krasanakis, Emmanouil and Schinas, Emmanouil and Papadopoulos, Symeon and Kompatsiaris, Yiannis and Mitkas, Pericles A},
  booktitle={ISMIR},
  pages={702--708},
  year={2018}
}
```