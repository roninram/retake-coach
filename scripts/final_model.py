"""Build the final coach model from cached DINOv2 features (no GPU, no API).

    python scripts/final_model.py --features quality/features.npz --build quality/build_features.npz \
        --labels Challenge2_labels.xlsx --out models/coach_model.npz

Inputs come from notebooks 1 and 3: DeepDRiD training/validation features and labels, and test, IDRiD,
widefield and everyday-picture features. Challenge2_labels.xlsx is DeepDRiD's test-label file
(regular_fundus_images/Online-Challenge1&2-Evaluation/).

Every choice is made on validation photos: regularization per head, problem thresholds (Youden), and the retake
threshold (send back at most 25% of good validation photos, declared before scoring the test set).
The test set is scored only for the report.
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import coach_core as cc  # noqa: E402

C_PER_HEAD = {"retake": 0.003, "focus": 0.01, "aim": 0.01, "obstruction": 0.01}   # best validation AUROC
MAX_GOOD_SENT_BACK = 0.25


def targets(d):
    return {"retake": (d["Overall quality"] == 0).values, "focus": (d["Clarity"] <= 6).values,
            "aim": (d["Field definition"] <= 6).values, "obstruction": (d["Artifact"] >= 6).values}


def auc_ci(y, s, n=2000):
    r, b = np.random.default_rng(1), []
    for _ in range(n):
        i = r.integers(0, len(y), len(y))
        if y[i].min() != y[i].max():
            b.append(roc_auc_score(y[i], s[i]))
    return np.percentile(b, [2.5, 97.5])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", required=True)
    ap.add_argument("--build", required=True)
    ap.add_argument("--labels", required=True)
    ap.add_argument("--out", default="models/coach_model.npz")
    a = ap.parse_args()

    F, G = np.load(a.features, allow_pickle=True), np.load(a.build, allow_pickle=True)
    dd, te = pd.DataFrame(F["dd_labels"]), pd.read_excel(a.labels)
    X, C = F["dd"], F["dd_checks"]
    tr, va = (dd.split == "train").values, (dd.split == "val").values
    Y, Yt = targets(dd), targets(te)
    Yv = {k: v[va] for k, v in Y.items()}

    coach = cc.Coach.fit(X[tr], C[tr], {k: v[tr] for k, v in Y.items()}, C=C_PER_HEAD)
    Pv, Pt = coach.probs(X[va], C[va]), coach.probs(G["te_X"], G["te_C"])
    thr = [np.quantile(Pv[~Yv["retake"], 0], 1 - MAX_GOOD_SENT_BACK)]
    for j, h in enumerate(cc.PROBLEMS, 1):
        fpr, tpr, th = roc_curve(Yv[h], Pv[:, j])
        thr.append(th[np.argmax(tpr - fpr)])
    coach.thr = np.array(thr)

    rows = []
    for j, h in enumerate(cc.Coach.HEADS):
        y, p, t = Yt[h], Pt[:, j], coach.thr[j]
        lo, hi = auc_ci(y, p)
        rows.append({"check": h, "test photos with it": f"{y.sum()}/{len(y)}", "AUROC test": round(roc_auc_score(y, p), 3),
                     "95% CI": f"{lo:.2f}-{hi:.2f}", "AUROC val": round(roc_auc_score(Yv[h], Pv[:, j]), 3),
                     "sharpness only (test)": round(roc_auc_score(y, -G["te_C"][:, 0]), 3),
                     "sensitivity": round((p[y] >= t).mean(), 2), "specificity": round((p[~y] < t).mean(), 2)})
    print("== Held-out DeepDRiD test (400 photos) ==")
    print(pd.DataFrame(rows).to_string(index=False))

    V = coach.verdict(G["te_X"], G["te_C"])
    flag, bad = np.array([v["retake"] for v in V]), Yt["retake"]
    truth = [{k for k in cc.PROBLEMS if Yt[k][i]} for i in range(len(te))]
    print(f"\nRetake asked for {flag.mean():.0%} of test photos: caught {(flag & bad).sum()}/{bad.sum()} not-good-enough "
          f"photos, sent back {(flag & ~bad).sum()}/{(~bad).sum()} good ones.")
    caught = [i for i in np.where(flag & bad)[0] if truth[i]]
    print(f"Caught photos with a labelled problem (n={len(caught)}): top named problem real "
          f"{np.mean([V[i]['problems'][0] in truth[i] for i in caught]):.0%}")
    for k in cc.PROBLEMS:
        idx = [i for i in np.where(flag)[0] if truth[i] == {k}]
        if idx:
            print(f"  only {k:11s}: top named problem right {np.mean([V[i]['problems'][0] == k for i in idx]):.0%} (n={len(idx)})")
    Vi = coach.verdict(G["idr_X"], G["idr_C"])
    print(f"IDRiD photos picked for grading (another camera): retake asked for {np.mean([v['retake'] for v in Vi]):.0%}")

    # guard: DeepDRiD training + 200 unlabeled IDRiD photos; threshold = 99th percentile on other fundus photos
    coach.set_guard(np.r_[X[tr], G["idr_X"][:200]], np.r_[X[va], G["idr_X"][200:300]])
    fw = np.array([n == "retina" for n in G["ev_names"]])
    ok = lambda Z: (coach.energy(Z) <= coach.g_thr).mean()  # noqa: E731
    print(f"Guard: accepts DeepDRiD test {ok(G['te_X']):.0%}, IDRiD 300+ {ok(G['idr_X'][300:]):.0%}, web fundus "
          f"{ok(G['ev_X'][fw]):.0%}; turns away widefield {1 - ok(G['uwf_X']):.0%}, everyday + black frame {1 - ok(G['ev_X'][~fw]):.0%}")
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    coach.save(a.out)
    print(f"saved {a.out}")


if __name__ == "__main__":
    main()
