import pygrank as pg

for name, graph, _ in pg.load_datasets_one_community(["facebook01"]):
    pg.benchmark_print_line(name, graph.number_of_nodes(), graph.number_of_edges())
