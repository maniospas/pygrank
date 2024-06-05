# Evaluation
There is often the need to assess the ability of node ranking algorithms to 
produce desired outcomes. To this end, `pygrank` provides a suite of measures
which can be used to assess node ranking algorithms, as well as comprehensive
code interfaces with which to quickly set up experiments of any scale to assess
the efficacy of new practices.

Most measures are either supervised in that they compare graph signal posteriors
with some known ground truth or unsupervised in that they explore whether
posteriors satisfy a desired property, such as low conductance or density.
Multiple measures can also be aggregated through the `pygrank.AM` and 
`pygrank.GM` classes, which respectively perform arithmetic and geometric
averaging of measure outcomes.

An exhaustive list of measures can be
found [here](../generated/measures.md). After initialization with the appropriate
parameters, these can be used interchangeably in the above example.

### Datasets
`pygrank` provides a variety of datasets to be automatically downloaded
and imported, but synthetically generated datasets have also been
[generated for pygrank](https://github.com/maniospas/pygrank-datasets).
To help researchers provide appropriate citations,
we provide messages in the error console pointing to respective sources.
Researchers can also add their own datasets by placing them in the loading
directory (usually a `data` directory in their project, alongside automatically
downloaded datasets). Please visit the repository of datasets
[generated for pygrank](https://github.com/maniospas/pygrank-datasets) for
a description of conventions needed to create new datasets - these follow
the pairs.txt and groups.txt conventions of the SNAP repository.

A comprehensive list of all datasets which are automatically downloaded by the project 
and their available types of data can 
be found [here](datasets.md). A lists of all dataset names can be obtained 
programmatically with the method `downloadable_small_datasets()`, but 
you can also use `downloadable_small_datasets()` to limit experiments
on datasets that run fastly (within seconds instead of minutes or tens of minutes)
in modern machines.

Given a list of datasets, these can be inputted to loader methods that
iteratively yield the outcome of loading. Five methods with the ability to load
incrementally more information are provided, where datasets lacking that information
are ommited:

1. `load_datasets_graph` Yields respective dataset graphs.
2. `load_datasets_one_community` Yields tuples of dataset names, graphs and node lists, where the lists correspond to one of the (structural or metadata) communities of graph nodes.
3. `load_datasets_all_communities` Yields the same tuples as before, but also traverses all possible datasets node communities (thus, there are many loading outcomes for each dataset). Community identifiers are appended to dataset names.
4. `load_datasets_multiple_communities` Yields tuples of dataset names, graphs and hashmaps, where the latter map community identifiers to lists of nodes. 
5. `load_feature_dataset` Yields respective tuples of graphs, features and node labels, where the last two are organized into numpy arrays whose lines correspond to graph nodes (based on the order the graphs are traversed).

:warning: To make sure that any kind of experiment is performed only adequately many data, communities with less than
1% of graph nodes are ommitted from loaders 1-4.

All the above loaders take as an argument a list of datasets and, if convenient,
a secondary argument of a directory location `path="data"`
(take care *not* to add a trailing slash) 
in which to download or load the datasets from. Loaders are iterables and thus they need to be re-defined to traverse
through datasets again. For example, the following code can be used to load datasets for overlapping community detection
given that each node community should be experimented on separately.

```python
import pygrank as pg
datasets = pg.downloadable_small_datasets()
print(datasets)
# ['citeseer', 'eucore', 'graph5', 'graph9', 'bigraph']
for dataset, graph, group in pg.load_datasets_all_communities(datasets):
    print(dataset, ":", len(group), "community members", len(graph), "nodes",  graph.number_of_edges(), "edges")
# REQUIRED CITATION: Please visit the url https://linqs.soe.ucsc.edu/data for instructions on how to cite the dataset citeseer in your research
# citeseer0 : 596 community members 3327 nodes 4676 edges
# citeseer1 : 668 community members 3327 nodes 4676 edges
# citeseer2 : 701 community members 3327 nodes 4676 edges
...
```


### Benchmarks
`pygrank` offers the ability to conduct benchmark experiments that compare
node ranking algorithms and parameters on a wide range of graphs. For example,
a simple way to obtain some fastly-running algorithms and small datasets and
compare them under the AUC measure would be per:

```python
import pygrank as pg
dataset_names = pg.downloadable_small_datasets()
print(dataset_names)
# ['citeseer', 'eucore']
algorithms = pg.create_demo_filters()
print(algorithms.keys())
# dict_keys(['PPR.85', 'PPR.9', 'PPR.99', 'HK3', 'HK5', 'HK7'])
loader = pg.load_datasets_one_community(dataset_names)
pg.benchmark_print(pg.benchmark(algorithms, loader, pg.AUC))
#                	 PPR.85  	 PPR.9  	 PPR.99  	 HK3  	 HK5  	 HK7 
# citeseer       	 .89     	 .89    	 .89     	 .89  	 .89  	 .89 
# eucore         	 .85     	 .71    	 .71     	 .91  	 .89  	 .83 
# graph9         	 1.00    	 1.00   	 1.00    	 1.00 	 1.00 	 1.00
# bigraph        	 .96     	 .77    	 .77     	 1.00 	 .98  	 .86 
# REQUIRED CITATION: Please visit the url https://linqs.soe.ucsc.edu/data for instructions on how to cite the dataset citeseer in your research
# REQUIRED CITATION: Please visit the url https://snap.stanford.edu/data/email-Eu-core.html for instructions on how to cite the dataset eucore in your research
# REQUIRED CITATION: Please visit the url https://github.com/maniospas/pygrank-datasets for instructions on how to cite the dataset graph5 in your research
# REQUIRED CITATION: Please visit the url https://github.com/maniospas/pygrank-datasets for instructions on how to cite the dataset graph9 in your research
# REQUIRED CITATION: Please visit the url https://github.com/maniospas/pygrank-datasets for instructions on how to cite the dataset bigraph in your research
```

In the above scheme customly-defined algorithms could also be added
or used in place of the `algorithms` dictionary.
For example, we could add an automatically-tuned
algorithm (more on these later) with default parameters per the following code and
then re-run experiments to compare this with alternatives.

```python
algorithms["Tuned"] = pg.ParameterTuner()
```

:warning: To run a new series of benchmark experiments, a new loader needs to be created.

