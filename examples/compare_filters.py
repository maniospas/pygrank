import pygrank as pg
datasets = ["EUCore", "Amazon"]
pre = pg.preprocessor(assume_immutability=True, normalization="symmetric")
algs = {"ppr.85": pg.PageRank(.85, preprocessor=pre, tol=1.E-9, max_iters=1000),
        "ppr.99": pg.PageRank(.99, preprocessor=pre, tol=1.E-9, max_iters=1000),
        "hk3": pg.HeatKernel(3, preprocessor=pre, tol=1.E-9, max_iters=1000),
        "hk5": pg.HeatKernel(5, preprocessor=pre, tol=1.E-9, max_iters=1000),
        }
loader = pg.load_datasets_one_community(datasets)
algs = algs | pg.create_variations(algs, {"+Sweep": pg.Sweep})
algs["tuned"] = pg.ParameterTuner(preprocessor=pre, tol=1.E-9, max_iters=1000)
algs["tuned+Sweep"] = pg.ParameterTuner(ranker_generator=lambda params: pg.Sweep(pg.GenericGraphFilter(params, preprocessor=pre, tol=1.E-9, max_iters=1000)))
pg.benchmark_print(pg.benchmark(algs, loader, pg.AUC, fraction_of_training=.5))
