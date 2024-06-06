# Graph Signals

<img src="../graph_signal.png" alt="graph signal" style="width: 40%;float: right;">

A *graph signal* is a way to organize numerical values 
that correspond to the nodes of a graph. Signals are used as the
inputs and outputs of node ranking algorithms, though calls to
the latter are for convenience overloaded to automatically 
construct signals if different arguments are provided.

Here is a simple graph that includes nodes
'A' and 'C' with values of 3 and 2 respectively.
A graph signal with these values and 0 for other nodes
can be created as:

```python
import pygrank as pg
import networkx as nx

graph = nx.Graph()
graph.add_edge('A', 'B')
graph.add_edge('A', 'C')
graph.add_edge('C', 'D')
graph.add_edge('D', 'E')
signal = pg.to_signal(graph, {'A': 3, 'C': 2})

print(signal['A'], signal['B'])  # 3.0 0.0
```

Formats to indicate the node values with which to construct signals per `pg.to_signal(graph, obj)` are:

| Format Type                                               | Description                                                                                                                                                                                        | Example                      |
|-----------------------------------------------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|------------------------------|
| Maps of node values                                       | Assume all other missing elements to represent zero values.                                                                                                                                        | `obj={'A': 3, 'C': 2}`       |
| Numpy arrays, lists or tensors.                           | Represent numerical values for each graph node, where nodes are organized per their traversal order in the graph's iterator. If tensors are provided, most computations remain backpropagate-able. | `obj=np.array([3, 0, 2, 0])` |
| List or set of nodes with fewer elements than graph nodes | Assigns value of one to specified nodes. Other nodes obtain zero values.                                                                                                                           | `obj=['A', 'C']`             |
| `None`                                                    | Interpreted as a signal of ones.                                                                                                                                                                   | `obj=None`                   |


!!! info
    Signal values can be accessed 
    through the `signal.np` attribute, which out-of-the-box holds a `numpy` array.
    Different data types may be held, depending on the *current*
    backend; switching backends after a signal is obtained will yield
    a representation in the new backend's preferred format. 

Arithmetic operations defined by the running backend
are also directly applicable to signals by implying the `np` attribute,
like below. All operations involving signals should occur on the same
graph, and include a respective assertion.

```python
signal = signal / pg.sum(signal)
print([(k,v) for k,v in signal.items()])  # [('A', 0.6), ('B', 0.0), ('C', 0.4), ('D', 0.0), ('E', 0.0)]
```
