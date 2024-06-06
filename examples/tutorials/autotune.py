import pygrank as pg

_, graph, group = pg.load_one("eucore")
signal = pg.to_signal(graph, group)

training, evaluation = pg.split(signal, training_samples=0.5)  # applies different masks

scores_pagerank = pg.PageRank(max_iters=1000)(training)
scores_tuned = pg.ParameterTuner()(training)

measure = pg.AUC(evaluation, exclude=training)
pg.benchmark_print_line("Pagerank", measure(scores_pagerank))
pg.benchmark_print_line("Tuned", measure(scores_tuned))
