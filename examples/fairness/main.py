import pygrank as pg
import setting

print("----------- PPR.85 -----------")
filter = pg.PageRank(
    0.85,
    max_iters=10000,
    tol=1.0e-9,
    assume_immutability=True,
    normalization="symmetric",
)
setting.experiment(
    filter,
    ["citeseer"],
    pg.AUC,
    [0.1, 0.2, 0.3, 0.4, 0.5],
    fix_personalization=True,
    repeats=2,
    sensitive_group=100,
)
