import pygrank as pg


datasets = ["EUCore", "DBLP"]
pre = pg.preprocessor(assume_immutability=True, normalization="symmetric")
algorithms = {
    "ppr0.85": pg.PageRank(alpha=0.85, to_scipy=pre, max_iters=1000000, tol=1.E-9),
    "ppr0.99": pg.PageRank(alpha=0.99, to_scipy=pre, max_iters=1000000, tol=1.E-9),
    "hk3": pg.HeatKernel(t=3, to_scipy=pre, max_iters=1000000, tol=1.E-9),
    "hk7": pg.HeatKernel(t=5, to_scipy=pre, max_iters=1000000, tol=1.E-9),
    "tuned": pg.ParameterTuner(to_scipy=pre, max_iters=1000000, tol=1.E-9),
}
#algorithms = benchmark.create_variations(algorithms, {"": pg.Tautology, "+SO": pg.SeedOversampling})
#loader = pg.dataset_loader(datasets)
#pg.supervised_benchmark(algorithms, loader, "time", verbose=True)

loader = pg.dataset_loader(datasets)
pg.supervised_benchmark(algorithms, loader, pg.AUC, verbose=True)