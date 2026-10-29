"""Does adding SAM's exudate evidence to the grader's energy improve the grade?  (no GPU, no API)

    python compose_eval.py exudate_features_test.csv npz_event [label_column] [area|area+spots]

Energy per grade g (lower = more compatible):
    grader alone        E(g) = -z_g                                    z = saved grader logits
    grader, recalibrated E(g) = -(t * z_g + b_g)                       control: does refitting alone help?
    grader + exudates   E(g) = -(t * z_g + b_g + w_g . phi(area))      phi = [any exudate, log area]
The two fitted versions are scored with leave-one-out, so no image is graded by a model fitted on itself.
Comparing the last two isolates what the exudate evidence adds beyond recalibration.
"""
import sys

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import log_softmax
from sklearn.metrics import cohen_kappa_score

K = 5


def phi(feats, kind="area"):
    a = np.nan_to_num(feats.exudate_area.values, nan=0.0)
    f = np.c_[(a > 1e-4).astype(float), np.log10(1e-4 + a) + 2.0]
    if kind == "area+spots":   # many small spots look like exudates; one or two big blobs look like reflections
        f = np.c_[f, np.log1p(np.nan_to_num(feats.exudate_spots.values, nan=0.0)) / 3.0]
    return f


def fit(z, f, y, use_evidence, reg=1.0):
    nf = f.shape[1] if use_evidence else 0

    def unpack(p):
        return p[0], p[1:1 + K], p[1 + K:].reshape(K, nf)

    def loss(p):
        t, b, w = unpack(p)
        s = t * z + b + (f @ w.T if nf else 0)
        return -log_softmax(s, 1)[np.arange(len(y)), y].mean() + reg * (np.sum(b ** 2) + np.sum(w ** 2)) / len(y)

    p0 = np.r_[1.0, np.zeros(K), np.zeros(K * nf)]
    t, b, w = unpack(minimize(loss, p0, method="L-BFGS-B").x)
    return lambda z2, f2: t * z2 + b + (f2 @ w.T if nf else 0)


def loo_predict(z, f, y, use_evidence):
    pred = np.empty(len(y), int)
    for i in range(len(y)):
        keep = np.arange(len(y)) != i
        score = fit(z[keep], f[keep], y[keep], use_evidence)
        pred[i] = score(z[i:i + 1], f[i:i + 1]).argmax()
    return pred


def metrics(y, p):
    low, high = y <= 1, y >= 2
    return dict(acc=(p == y).mean(), qwk=cohen_kappa_score(y, p, weights="quadratic"),
                over_call=(p[low] >= 2).mean(), under_call=(p[high] <= 1).mean())


def main(features_csv, npz_dir, label_col="label", kind="area"):
    feats = pd.read_csv(features_csv)
    f, y_feat, y_eval = phi(feats, kind), feats.label.values, feats[label_col].values
    rows = []
    for model in ["ebl", "ce"]:
        for seed in [42, 43, 44, 45, 46]:
            d = np.load(f"{npz_dir}/{model}/seed{seed}.npz")
            z, y = d["idrid_logits"].astype(float), d["idrid_labels"]
            assert (y == y_feat).all(), "feature rows are not in the order of the saved logits"
            y = y_eval
            for name, pred in [("grader alone", z.argmax(1)),
                               ("grader, recalibrated", loo_predict(z, f, y, False)),
                               ("grader + exudates", loo_predict(z, f, y, True))]:
                rows.append(dict(model=model.upper(), seed=seed, method=name, **metrics(y, pred)))
    df = pd.DataFrame(rows)
    order = ["grader alone", "grader, recalibrated", "grader + exudates"]
    summary = df.groupby(["model", "method"])[["acc", "qwk", "over_call", "under_call"]].mean().reindex(order, level=1)
    print(summary.round(3).to_string())
    w = df.pivot_table(index="seed", columns=["model", "method"], values=["acc", "qwk", "over_call"])
    for model in ["EBL", "CE"]:
        for k, better in [("acc", 1), ("qwk", 1), ("over_call", -1)]:
            dlt = w[k][model]["grader + exudates"] - w[k][model]["grader, recalibrated"]
            print(f"{model} {k:9s} exudates vs recalibrated: {dlt.mean():+.3f}  "
                  f"({((dlt * better) > 0).sum()}/5 better, {(dlt == 0).sum()} ties)")
    df.to_csv("compose_eval.csv", index=False)


if __name__ == "__main__":
    main(*sys.argv[1:5])
