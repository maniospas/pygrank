# Overview

<img src="../architecture.png" alt="architecture" style="width: 40%;float: right;">

At the core of `pygrank` lies the concept of *graph signals*, which map graph nodes to scores. 
Supervised and unsupervised measures evaluate the predictive/ranking quality
of graph signals. 

Node ranking algorithms can be defined. These start from *graph filters*, 
which diffuse the scores of nodes to neighbors that are connected to them. 
The output of these filters can be processed further with additional components. 
Finally, benchmarking experiments compare algorithms.

Here is a glossary of common concepts sorted alphabetically:


| Term                   | Explanation                                                                                                                                                                                                                                                                    |
|------------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Graph filter           | A strategy of diffusing node scores through graphs through edges.                                                                                                                                                                                                              |
| Graph neural network   | A combination of traditional neural networks and node ranking algorithms.                                                                                                                                                                                                      |
| Graph signal           | Scores assigned to nodes. Each node's score typically assumes values in the range \[0,1\].                                                                                                                                                                                     |
| Node ranking algorithm | An algorithm that starts with a graph and personalization and outputs graph signals that boast higher quality than the personalization, for example by making predictions for all graph ndes. Typically consists of combinations of graph filters with postprocessing schemes. |
| Personalization        | The graph signal inputted in grap filters. This is also known as graph signal priors or the personalization vector.                                                                                                                                                            |
| Seeds                  | Example nodes that are known to belong to a community.                                                                                                                                                                                                                         |
| Tuning                 | A process of determining algorithm hyper-parameters that optimize some evaluation measure.                                                                                                                                                                                     |
