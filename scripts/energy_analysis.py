"""What the guard's energy tells, from cached features ($0, no GPU, no API).

    python scripts/energy_analysis.py --build quality/build_features.npz --labels Challenge2_labels.xlsx \
        --model models/coach_model.npz --out results/energy_report.txt

The guard's energy is how far a photo's DINOv2 features sit from a Gaussian fitted on ordinary fundus photos
(Mahalanobis distance divided by the number of dimensions). That Gaussian was fitted on DeepDRiD training photos and
200 IDRiD photos, good and bad alike, with no quality labels (scripts/final_model.py). This script asks:
1. How the energy separates fundus photos from everything else.
2. Whether the same label-free energy already ranks bad photos higher, on the held-out DeepDRiD test set.
3. Whether it tells when the trained checks are wrong.
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import coach_core as cc  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--build", required=True, help="quality/build_features.npz from notebook 3")
    ap.add_argument("--labels", required=True, help="DeepDRiD Challenge2_labels.xlsx (test labels)")
    ap.add_argument("--model", default="models/coach_model.npz")
    ap.add_argument("--out", default="results/energy_report.txt")
    a = ap.parse_args()

    coach = cc.Coach.load(a.model)
    G = np.load(a.build, allow_pickle=True)
    te = pd.read_excel(a.labels)
    lines = []
    say = lambda s="": (print(s), lines.append(s))  # noqa: E731

    web = np.array([n == "retina" for n in G["ev_names"]])
    groups = [("DeepDRiD test fundus photos", G["te_X"]), ("IDRiD photos 300+ (not used to fit the guard)", G["idr_X"][300:]),
              ("fundus photo from the web", G["ev_X"][web]), ("DeepDRiD widefield photos", G["uwf_X"]),
              ("everyday pictures and a black frame", G["ev_X"][~web])]
    say(f"== 1. Energy by kind of input (limit {coach.g_thr:.2f}; above it the photo is turned away) ==")
    rows = []
    for name, Z in groups:
        e = coach.energy(Z)
        rows.append(dict(input=name, photos=len(e), median=round(float(np.median(e)), 2), lowest=round(float(e.min()), 2),
                         highest=round(float(e.max()), 2), turned_away=f"{(e > coach.g_thr).mean():.0%}"))
    say(pd.DataFrame(rows).to_string(index=False))
    if "cor_X" in G.files:
        e = coach.energy(G["cor_X"])
        say("\nIDRiD photos with synthetic damage, median energy by kind of damage: "
            + ", ".join(f"{k} {v:.2f}" for k, v in pd.Series(e).groupby(G["cor_group"]).median().items()))

    E = coach.energy(G["te_X"])
    P = coach.probs(G["te_X"], G["te_C"])
    labels = {"not good enough overall": (te["Overall quality"] == 0).values, "focus (clarity <= 6)": (te["Clarity"] <= 6).values,
              "aim (field definition <= 6)": (te["Field definition"] <= 6).values, "obstruction (artifact >= 6)": (te["Artifact"] >= 6).values}
    say("\n== 2. Label-free quality: does the energy rank bad photos higher? (DeepDRiD test, 400 photos) ==")
    rows = [dict(problem=k, photos_with_it=int(y.sum()), AUROC_energy_alone=round(roc_auc_score(y, E), 3),
                 AUROC_trained_check=round(roc_auc_score(y, P[:, j]), 3)) for j, (k, y) in enumerate(labels.items())]
    say(pd.DataFrame(rows).to_string(index=False))
    say("The energy never saw a quality label; the trained checks did.")

    V = coach.verdict(G["te_X"], G["te_C"])
    bad = labels["not good enough overall"]
    retake = np.array([v["retake"] for v in V])
    wrong = retake != bad
    margin = np.abs(cc._logit(P[:, 0]) - cc._logit(coach.thr[0]))
    say("\n== 3. Does the energy tell when the retake decision is wrong? ==")
    say(f"Decisions that disagree with DeepDRiD's label: {wrong.sum()} of {len(wrong)}. "
        f"AUROC for spotting them: energy {roc_auc_score(wrong, E):.2f}; the check's own distance from its threshold "
        f"{roc_auc_score(wrong, -margin):.2f}. 0.50 is chance.")
    q = pd.qcut(E, 4, labels=["lowest quarter", "second", "third", "highest quarter"])
    say(pd.DataFrame(dict(energy=q, wrong=wrong, bad=bad, retake=retake)).groupby("energy", observed=True)
        .agg(photos=("wrong", "size"), not_good_enough=("bad", "mean"), sent_back=("retake", "mean"), decision_wrong=("wrong", "mean"))
        .round(2).to_string())

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"\nsaved {a.out}")


if __name__ == "__main__":
    main()
