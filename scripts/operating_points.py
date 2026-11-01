"""How many not-good-enough photos the retake decision catches, against how many good photos it sends back.
From cached features: $0, no GPU, no API.

    python scripts/operating_points.py --features quality/features.npz --build quality/build_features.npz \
        --labels Challenge2_labels.xlsx --model models/coach_model.npz --out results/operating_points.txt

The retake decision compares the overall score with one line. Today's line was set to send back at most 25% of good
*validation* photos. Each other line here is chosen the same way, on validation photos only: the line that catches a
target share of not-good-enough validation photos. Then the 400 test photos are scored once per line, with 95%
bootstrap intervals. The test set chooses nothing.
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import coach_core as cc  # noqa: E402

TARGETS = (0.80, 0.85, 0.90, 0.95)
BAD_PER_100 = 20        # an example clinic mix: ungradable rates of 12-24% are reported (see the README's opening)


def rates(p, y, thr):
    return (p[y] >= thr).mean(), (p[~y] >= thr).mean()


def interval(p, y, thr, n=2000, seed=1):
    r, b = np.random.default_rng(seed), []
    for _ in range(n):
        i = r.integers(0, len(y), len(y))
        b.append(rates(p[i], y[i], thr))
    return np.percentile(np.array(b), [2.5, 97.5], axis=0)          # rows: low, high; columns: caught, good sent back


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", required=True, help="quality/features.npz (DeepDRiD training and validation)")
    ap.add_argument("--build", required=True, help="quality/build_features.npz (DeepDRiD test)")
    ap.add_argument("--labels", required=True, help="DeepDRiD Challenge2_labels.xlsx (test labels)")
    ap.add_argument("--model", default="models/coach_model.npz")
    ap.add_argument("--out", default="results/operating_points.txt")
    a = ap.parse_args()

    coach = cc.Coach.load(a.model)
    if coach.rule != "overall":
        sys.exit(f"This model asks for a retake by the rule '{coach.rule}'; this script covers the 'overall' rule only.")
    F, G = np.load(a.features, allow_pickle=True), np.load(a.build, allow_pickle=True)
    dd, te = pd.DataFrame(F["dd_labels"]), pd.read_excel(a.labels)
    va = (dd.split == "val").values
    pv = coach.probs(F["dd"][va], F["dd_checks"][va])[:, 0]
    yv = (dd["Overall quality"] == 0).values[va]
    pt = coach.probs(G["te_X"], G["te_C"])[:, 0]
    yt = (te["Overall quality"] == 0).values

    lines = [("today's line", float(coach.thr[0]))] + [(f"catch {t:.0%}", float(np.quantile(pv[yv], 1 - t))) for t in TARGETS]
    rows, points = [], []
    for name, thr in lines:
        (vc, vs), (tc, ts) = rates(pv, yv, thr), rates(pt, yt, thr)
        (lc, ls), (hc, hs) = interval(pt, yt, thr)
        good = 100 - BAD_PER_100
        rows.append({"line": name, "score line": round(thr, 3), "val: caught/good back": f"{vc:.0%} / {vs:.0%}",
                     "test: caught": f"{tc:.0%} ({lc:.0%}-{hc:.0%})", "test: good sent back": f"{ts:.0%} ({ls:.0%}-{hs:.0%})",
                     "per 100: retakes": round(BAD_PER_100 * tc + good * ts), "per 100: bad kept": round(BAD_PER_100 * (1 - tc))})
        points.append((name, ts, tc))

    out = [f"== Retake line trade-off: DeepDRiD test, {yt.sum()} not-good-enough and {(~yt).sum()} good photos ==",
           "Each line is chosen on validation photos only: today's sends back at most 25% of good validation photos, and",
           "'catch N%' catches N% of not-good-enough validation photos. Test results have 95% bootstrap intervals.",
           pd.DataFrame(rows).to_string(index=False),
           f"\n'per 100' applies the test rates to an example clinic mix of {BAD_PER_100} not-good-enough photos in 100 "
           f"(reported ungradable rates are 12-24%): retakes asked, and bad photos kept."]
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "w") as f:
        f.write("\n".join(out) + "\n")
    print("\n".join(out))
    plot(pt, yt, points, os.path.splitext(a.out)[0] + ".png")
    print(f"\nsaved {a.out} and its figure")


def plot(pt, yt, points, path):
    """The test trade-off curve with each validation-chosen line on it."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import roc_curve

    ink, ink2, grid, line, now = "#0b0b0b", "#52514e", "#e4e3df", "#2a78d6", "#eb6834"
    fpr, tpr, _ = roc_curve(yt, pt)
    fig, ax = plt.subplots(figsize=(6.4, 4.6), dpi=160)
    fig.patch.set_facecolor("#fcfcfb")
    ax.set_facecolor("#fcfcfb")
    ax.plot(100 * fpr, 100 * tpr, color=line, lw=2, zorder=2)
    for i, (name, x, y) in enumerate(points):         # labels alternate below-right and above-left of the curve
        today, below = i == 0, i % 2 == 0
        ax.scatter(100 * x, 100 * y, s=64, color=now if today else line, edgecolor="#fcfcfb", linewidth=2, zorder=3)
        ax.annotate(name, (100 * x, 100 * y), xytext=(8, -15) if below else (-8, 9), textcoords="offset points",
                    ha="left" if below else "right", fontsize=9, color=ink if today else ink2,
                    fontweight="bold" if today else "normal")
    ax.text(99, 3, "Each dot is a retake line chosen on validation photos.\nTable: results/operating_points.txt",
            ha="right", va="bottom", fontsize=8, color=ink2)
    ax.set_xlim(0, 100), ax.set_ylim(0, 102)
    ax.set_xlabel("Good photos sent back for a retake (%)", color=ink2, fontsize=9.5)
    ax.set_ylabel("Not-good-enough photos caught (%)", color=ink2, fontsize=9.5)
    ax.set_title("Retake line trade-off on 400 DeepDRiD test photos", color=ink, fontsize=11, loc="left", pad=10)
    ax.grid(color=grid, lw=0.8), ax.set_axisbelow(True)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.tick_params(colors=ink2, labelsize=8.5, length=0)
    fig.tight_layout()
    fig.savefig(path, facecolor=fig.get_facecolor())
    plt.close(fig)


if __name__ == "__main__":
    main()
