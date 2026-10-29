"""Calibration, error detection and dataset-shift detection from saved per-seed outputs.

Save one .npz per seed from your existing training code, for each loss:
    runs/ebl/seed42.npz, runs/ebl/seed43.npz, ...   runs/ce/seed42.npz, ...
with these arrays (N x 5 outputs, N labels on the 0-4 scale):
    idrid_logits, idrid_labels        IDRiD test split (in-distribution)
    aptos_logits, aptos_labels        APTOS-2019 external split (shifted)
    val_logits, val_labels            optional: IDRiD validation split, for temperature scaling

If a method's arrays hold per-class energies (lower = more likely), pass --energies for it.

    pip install numpy scipy scikit-learn pandas
    python energy_eval.py runs/ebl runs/ce --energies runs/ebl
"""
import argparse, glob, os

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from scipy.special import log_softmax, logsumexp
from sklearn.metrics import cohen_kappa_score, roc_auc_score


def ece(p, y, bins=15):
    conf, correct = p.max(1), p.argmax(1) == y
    b = np.minimum((conf * bins).astype(int), bins - 1)
    return sum(abs(correct[b == i].mean() - conf[b == i].mean()) * (b == i).mean()
               for i in range(bins) if (b == i).any())


def fit_temperature(s, y):
    nll = lambda t: -log_softmax(s / t, 1)[np.arange(len(y)), y].mean()
    return minimize_scalar(nll, bounds=(0.05, 20), method="bounded").x


def confidences(s):
    """Higher = more confident. The gap and free energy are the energy view; MSP is the softmax baseline."""
    top2 = np.sort(s, 1)[:, -2:]
    return {"msp": np.exp(log_softmax(s, 1)).max(1),
            "energy_gap": top2[:, 1] - top2[:, 0],
            "neg_free_energy": logsumexp(s, 1)}


def split_metrics(s, y, T=None):
    logp = log_softmax(s, 1)
    p, pred = np.exp(logp), logp.argmax(1)
    m = dict(acc=(pred == y).mean(),
             qwk=cohen_kappa_score(y, pred, weights="quadratic"),
             ece=ece(p, y),
             brier=((p - np.eye(s.shape[1])[y]) ** 2).sum(1).mean(),
             nll=-logp[np.arange(len(y)), y].mean())
    if T is not None:
        m["ece_temp_scaled"] = ece(np.exp(log_softmax(s / T, 1)), y)
    wrong = pred != y
    if 0 < wrong.sum() < len(y):  # can low confidence pick out the wrong predictions?
        for k, c in confidences(s).items():
            m[f"error_auroc_{k}"] = roc_auc_score(wrong, -c)
    return m


def evaluate_run(path, energies):
    d = np.load(path)
    sign = -1.0 if energies else 1.0
    s_in, s_out = sign * d["idrid_logits"], sign * d["aptos_logits"]
    T = fit_temperature(sign * d["val_logits"], d["val_labels"]) if "val_logits" in d else None
    row = {"temperature": np.nan if T is None else T}
    row.update({f"idrid_{k}": v for k, v in split_metrics(s_in, d["idrid_labels"], T).items()})
    row.update({f"aptos_{k}": v for k, v in split_metrics(s_out, d["aptos_labels"], T).items()})
    # Can the score tell an APTOS image from an IDRiD one? 0.5 = no better than chance.
    is_shifted = np.r_[np.zeros(len(s_in)), np.ones(len(s_out))]
    c_in, c_out = confidences(s_in), confidences(s_out)
    # Same test on grade-0 images only, so a different grade mix between datasets can't explain it.
    g0 = np.r_[d["idrid_labels"] == 0, d["aptos_labels"] == 0]
    for k in c_in:
        c = -np.r_[c_in[k], c_out[k]]
        row[f"shift_auroc_{k}"] = roc_auc_score(is_shifted, c)
        if 0 < is_shifted[g0].sum() < g0.sum():
            row[f"shift_auroc_grade0_{k}"] = roc_auc_score(is_shifted[g0], c[g0])
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("methods", nargs="+", help="one folder of seed*.npz per method")
    ap.add_argument("--energies", nargs="*", default=[], help="folders whose arrays are energies, not logits")
    ap.add_argument("--out", default="energy_eval.csv")
    args = ap.parse_args()

    rows = []
    for folder in args.methods:
        for path in sorted(glob.glob(os.path.join(folder, "seed*.npz"))):
            energies = os.path.normpath(folder) in {os.path.normpath(e) for e in args.energies}
            row = evaluate_run(path, energies)
            rows.append({"method": os.path.basename(os.path.normpath(folder)),
                         "seed": os.path.splitext(os.path.basename(path))[0], **row})
    df = pd.DataFrame(rows)
    df.to_csv(args.out, index=False)

    summary = df.drop(columns="seed").groupby("method").agg(["mean", "std"]).T
    pd.set_option("display.width", 160)
    print(summary.round(3).to_string())
    print(f"\n{len(df)} runs; per-seed results in {args.out}")


if __name__ == "__main__":
    main()
