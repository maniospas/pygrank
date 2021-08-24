import pygrank as pg
import io


def test_benchmark_print():
    assert pg.benchmark.utils._fraction2str(0.1) == ".10"
    assert pg.benchmark.utils._fraction2str(0.00001) == "0"
    assert pg.benchmark.utils._fraction2str(1) == "1.00"
    loader = pg.load_datasets_one_community(["graph9", "bigraph"])
    console = pg.benchmark_print(pg.benchmark(pg.create_demo_filters(), loader),
                                 out=io.StringIO(""), err=None).getvalue()
    loader = pg.load_datasets_one_community(["graph9", "bigraph"])
    ret = pg.benchmark_dict(pg.benchmark(pg.create_demo_filters(), loader))
    assert isinstance(ret, dict)
    assert len(ret) == 2
    assert isinstance(ret["graph9"], dict)
    assert (len(str(ret)) - len(console)) < (len(str(ret)) + len(console))/5


def test_unsupervised_vs_auc():
    def loader():
        return pg.load_datasets_one_community(["graph9"])
    algorithms = pg.create_variations(pg.create_many_filters(), pg.create_many_variation_types())
    time_scores = pg.benchmark_scores(pg.benchmark(algorithms, loader(), "time"))
    assert sum(time_scores) > 0

    measures = {"AUC": lambda ground_truth, exlude: pg.MultiSupervised(pg.AUC, ground_truth, exlude),
                "NDCG": lambda ground_truth, exlude: pg.MultiSupervised(pg.NDCG, ground_truth, exlude),
                "Density": lambda graph: pg.MultiUnsupervised(pg.Density, graph),
                "Modularity": lambda graph: pg.MultiUnsupervised(pg.Modularity, graph),
                "CCcos": lambda graph: pg.ClusteringCoefficient(graph, similarity="cos"),
                "CCdot": lambda graph: pg.ClusteringCoefficient(graph, similarity="dot"),
                "LinkAUCcos": lambda graph: pg.LinkAssessment(graph, similarity="cos"),
                "LinkAUCdot": lambda graph: pg.LinkAssessment(graph, similarity="dot"),
                "HopAUCcos": lambda graph: pg.LinkAssessment(graph, similarity="cos", hops=2),
                "HopAUCdot": lambda graph: pg.LinkAssessment(graph, similarity="dot", hops=2),
                }

    scores = {measure: pg.benchmark_scores(pg.benchmark(algorithms, loader(), measures[measure])) for measure in measures}
    supervised = {"AUC", "NDCG"}
    for measure in measures:
        if measure not in supervised:
            print(measure, pg.SpearmanCorrelation(scores["AUC"], scores[measure]))



def test_benchmarks(self):
    datasets = pg.downloadable_small_datasets()
    pre = pg.preprocessor(assume_immutability=True, normalization="symmetric")
    algorithms = {
        "ppr0.85": pg.PageRank(alpha=0.85, preprocessor=pre, max_iters=10000, tol=1.E-9),
        "ppr0.99": pg.PageRank(alpha=0.99, preprocessor=pre, max_iters=10000, tol=1.E-9),
        "hk3": pg.HeatKernel(t=3, preprocessor=pre, max_iters=10000, tol=1.E-9),
        "hk5": pg.HeatKernel(t=5, preprocessor=pre, max_iters=10000, tol=1.E-9),
        "tuned": pg.ParameterTuner(preprocessor=pre, max_iters=10000, tol=1.E-9),
    }
    # algorithms = benchmark.create_variations(algorithms, {"": pg.Tautology, "+SO": pg.SeedOversampling})
    # loader = pg.load_datasets_one_community(datasets)
    # pg.benchmark(algorithms, loader, "time", verbose=True)

    loader = pg.load_datasets_one_community(datasets)
    pg.benchmark_print(pg.benchmark(algorithms, loader, pg.AUC, fraction_of_training=.8))
    loader = pg.load_datasets_one_community(datasets)
    pg.benchmark_print(pg.benchmark(algorithms, loader, pg.Conductance, fraction_of_training=.8))



def test_load_datasets_all_communities(self):
    self.assertGreater(len(list(pg.load_datasets_all_communities(["citeseer"]))), 1)

def test_dataset_generation(self):
    self.assertEquals(len(pg.downloadable_datasets()), len(pg.datasets))

    def test_multigroup(self):
        datasets = pg.downloadable_small_datasets()
        pre = pg.preprocessor(assume_immutability=True, normalization="symmetric")
        algorithms = {
            "ppr0.85": pg.PageRank(alpha=0.85, preprocessor=pre, max_iters=10000, tol=1.E-9),
            "ppr0.99": pg.PageRank(alpha=0.99, preprocessor=pre, max_iters=10000, tol=1.E-9),
            "hk3": pg.HeatKernel(t=3, preprocessor=pre, max_iters=10000, tol=1.E-9),
            "hk5": pg.HeatKernel(t=5, preprocessor=pre, max_iters=10000, tol=1.E-9),
        }
        loader = pg.load_datasets_multiple_communities(datasets)
        pg.benchmark_print(pg.benchmark(algorithms, loader,
                                        lambda ground_truth, exlude: pg.MultiSupervised(pg.AUC, ground_truth, exlude)))
        loader = pg.load_datasets_multiple_communities(datasets)
        pg.benchmark_print(pg.unsupervised_benchmark(algorithms, loader,
                                                   lambda graph: pg.LinkAssessment(graph, max_positive_samples=200, max_negative_samples=200)))
        loader = pg.load_datasets_multiple_communities(datasets)
        pg.benchmark_print(pg.unsupervised_benchmark(algorithms, loader,
                                                   lambda graph: pg.LinkAssessment(graph, hops=2, similarity="dot", max_positive_samples=200, max_negative_samples=200)))
        loader = pg.load_datasets_multiple_communities(datasets)
        pg.benchmark_print(pg.unsupervised_benchmark(algorithms, loader,
                                                   lambda graph: pg.ClusteringCoefficient(graph)))
        loader = pg.load_datasets_multiple_communities(datasets)
        pg.benchmark_print(pg.unsupervised_benchmark(algorithms, loader,
                                                   lambda graph: pg.ClusteringCoefficient(graph, similarity="dot")))
        loader = pg.load_datasets_multiple_communities(datasets)
        pg.benchmark_print(pg.unsupervised_benchmark(algorithms, loader,
                                                   lambda graph: pg.MultiUnsupervised(pg.Conductance, graph)))