import pygrank as pg
import networkx as nx
import random
import tqdm

_, graph, group = next(pg.load_datasets_one_community(["citeseer"]))
algorithm = pg.SymmetricAbsorbingRandomWalks() >> pg.SeedOversampling("neighbors") >> pg.Sweep()

tprs = list()
ppvs = list()
f1s = list()
for node in tqdm.tqdm(list(graph)):
    #training, test = pg.split(pg.to_signal(graph, list(group)), 1.1)
    neighbors = list(graph.neighbors(node))
    if len(neighbors) < 3:
        continue
    training = pg.to_signal(graph, {node: 1})
    test = pg.to_signal(graph, {neighbor: 1 for neighbor in neighbors})
    for neighbor in random.sample(neighbors, 1):
        graph.remove_edge(node, neighbor)
    top5 = (training >> algorithm)*(1-training) >> pg.Top(10) >> pg.Threshold()
    ppvs.append(pg.PPV(test, exclude=training)(top5))
    tprs.append(pg.TPR(test, exclude=training)(top5))
    prec = pg.PPV(test, exclude=training)(top5)
    rec = pg.TPR(test, exclude=training)(top5)
    f1s.append(pg.safe_div(2*prec*rec, prec+rec))
    for neighbor in graph.neighbors(node):
        graph.add_edge(node, neighbor)

print(f"f1 {sum(f1s) / len(f1s):.3f}\t prec {sum(ppvs) / len(ppvs):.3f}\t rec {sum(tprs)/len(tprs):.3f}\t")