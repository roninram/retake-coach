"""Export per-seed logits from ebl_fundus checkpoints, for energy_eval.py.

Run from inside the ebl_fundus folder (prior work: it supplies dataset.py and model.py) after training:
    python /path/to/export_logits.py --aptos_csv aptos/train.csv --aptos_dir aptos/train_images

Writes npz/<ce|ebl>/seed<N>.npz holding logits and labels for the IDRiD validation split, the IDRiD
test split and APTOS-2019. The model outputs logits (energy = -logit), so run energy_eval.py WITHOUT --energies.

APTOS-2019 labelled images (Kaggle competition data, use under its rules; not redistributed here):
https://www.kaggle.com/competitions/aptos2019-blindness-detection/data
"""
import argparse, glob, sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, "src")
from dataset import get_dataloaders, get_transforms  # noqa: E402  (ebl_fundus code)
from model import build_model  # noqa: E402

NAMES = {"softmax": "ce", "ebm_margin": "ebl", "ebm_margin_hard": "ebl_hard"}


class Aptos(Dataset):
    def __init__(self, df, root, ext):
        self.df, self.root, self.ext, self.tf = df.reset_index(drop=True), Path(root), ext, get_transforms("test")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        r = self.df.iloc[i]
        return self.tf(Image.open(self.root / f"{r.id_code}{self.ext}").convert("RGB")), int(r.diagnosis)


def materialize(loader):
    """Decode and transform every image once, so 10 checkpoints don't re-read thousands of PNGs."""
    xs, ys = zip(*loader)
    return torch.cat(xs).half(), torch.cat(ys).numpy()


@torch.no_grad()
def logits_of(model, X, device, bs=64):
    return torch.cat([model(X[i:i + bs].to(device).float()).cpu() for i in range(0, len(X), bs)]).numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="runs", help="folder of training runs, one per loss and seed")
    ap.add_argument("--data_dir", default="data/idrid")
    ap.add_argument("--aptos_csv", required=True, help="CSV with id_code, diagnosis")
    ap.add_argument("--aptos_dir", required=True)
    ap.add_argument("--aptos_ext", default=".png")
    ap.add_argument("--aptos_n", type=int, default=None, help="stratified sample size (default: every image)")
    ap.add_argument("--out", default="npz")
    a = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    aptos = pd.read_csv(a.aptos_csv)
    if a.aptos_n:
        aptos, _ = train_test_split(aptos, train_size=a.aptos_n, stratify=aptos.diagnosis, random_state=0)
    idrid = get_dataloaders(str(Path(a.data_dir) / "splits.csv"), str(Path(a.data_dir) / "images"), batch_size=32)
    print(f"preparing images: IDRiD val/test + {len(aptos)} APTOS")
    data = {"val": materialize(idrid["val"]), "idrid": materialize(idrid["test"]),
            "aptos": materialize(DataLoader(Aptos(aptos, a.aptos_dir, a.aptos_ext), batch_size=64, num_workers=2))}

    written = set()
    for path in sorted(glob.glob(f"{a.runs}/*/best.pt")):
        ck = torch.load(path, map_location=device, weights_only=False)
        assert [int(c) for c in ck["class_names"]] == list(range(5)), f"{path}: classes {ck['class_names']}"
        dest = Path(a.out) / NAMES[ck["args"]["loss"]] / f"seed{ck['args']['seed']}.npz"
        if dest in written:
            sys.exit(f"{path} and an earlier run both map to {dest}: keep one run per loss and seed in --runs")
        written.add(dest)
        model = build_model(num_classes=5, pretrained=False).to(device).eval()
        model.load_state_dict(ck["model_state"])
        out = {}
        for k, (X, y) in data.items():
            out[f"{k}_logits"], out[f"{k}_labels"] = logits_of(model, X, device), y
        dest.parent.mkdir(parents=True, exist_ok=True)
        np.savez(dest, **out)
        acc = {k: (out[f"{k}_logits"].argmax(1) == out[f"{k}_labels"]).mean() for k in ("idrid", "aptos")}
        print(f"{dest}  IDRiD test acc {acc['idrid']:.3f} | APTOS acc {acc['aptos']:.3f}")


if __name__ == "__main__":
    main()
