"""Score SAM 3.1 text-prompted lesion masks against IDRiD expert annotations.

Data: IDRiD "A. Segmentation" subset (81 fundus images with pixel masks for
microaneurysms, haemorrhages, hard and soft exudates):
https://ieee-dataport.org/open-access/indian-diabetic-retinopathy-image-dataset-idrid

    pip install openai meta-sam-parser pillow numpy scipy pandas
    export MODEL_API_KEY=...
    python sam_lesions.py --root "A. Segmentation" --split test --out sam_out

Lesions are only a few pixels wide once SAM shrinks a 4288x2848 photo, so the image is cut into tiles
(--tile, default 1024) and each tile is one request. Plain visual prompts are used: in the pilot, medical terms
("hard exudate") returned nothing where "yellow spot" found exudates at 98.7% precision.
Each tile x prompt is one request ($2.50 per 1,000); the run prints the number of requests.
Predicted masks are cached as PNGs, so re-runs are free.
"""
import argparse, base64, glob, io, os
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
from PIL import Image
from scipy import ndimage
from openai import OpenAI
from meta_sam_parser import CompletedOutcome, decode_mask_to_raster, image_segmentation_format

Image.MAX_IMAGE_PIXELS = None
LESIONS = {  # IDRiD mask code -> plain visual prompts (SAM 3.1 matches everyday concepts, not medical terms)
    "MA": ["tiny red dot", "small red dot"],
    "HE": ["dark red spot", "red blot"],
    "EX": ["yellow spot", "bright yellow patch"],
    "SE": ["white fluffy spot", "pale patch"],
}
client = OpenAI(base_url="https://api.meta.ai/v1", api_key=os.environ.get("MODEL_API_KEY"))


def sam_call(img, prompt, min_conf=0.0):
    """One SAM 3.1 request: union of every match for one concept, as a bool array at the image's own size."""
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=95)
    url = "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()

    resp = client.responses.create(
        model="sam-3.1",
        input=[{"type": "message", "role": "user", "content": [
            {"type": "input_text", "text": prompt},
            {"type": "input_image", "image_url": url}]}],
        metadata={"mask_encoding": "one_bit", "include_confidence": "true"},
    )
    parser = image_segmentation_format().create_parser()
    parser.push(resp.output_text or "")  # empty text = zero matches, a valid answer
    records = parser.finish(CompletedOutcome()).result.records

    w, h = img.size
    out = np.zeros((h, w), bool)
    for r in records:
        if r.kind != "mask" or (r.confidence is not None and r.confidence < min_conf):
            continue
        m = np.frombuffer(decode_mask_to_raster(r.mask), np.uint8).reshape(r.mask.height, r.mask.width)
        x0, y0, x1, y1 = (int(v) for v in (r.bounds.left, r.bounds.top, r.bounds.right, r.bounds.bottom))
        if m.shape != (y1 - y0, x1 - x0):  # masks are box-local; resize if the raster is coarser
            m = np.array(Image.fromarray(m * 255).resize((x1 - x0, y1 - y0), Image.NEAREST))
        out[y0:y1, x0:x1] |= m > 0
    return out


def tiles(img, tile, overlap=64):
    """Tile boxes covering the image, skipping tiles that are almost all black corner outside the retina."""
    W, H = img.size
    step = tile - overlap
    boxes = [(x, y, min(x + tile, W), min(y + tile, H))
             for y in range(0, max(H - overlap, 1), step) for x in range(0, max(W - overlap, 1), step)]
    gray = np.asarray(img.convert("L"))
    return [b for b in boxes if (gray[b[1]:b[3], b[0]:b[2]] > 20).mean() > 0.1]


def sam_mask(img, prompt, tile=0, min_conf=0.0, workers=8):
    """Whole image (tile=0) or tiled. Returns the mask and the number of requests made."""
    if not tile:
        return sam_call(img, prompt, min_conf), 1
    W, H = img.size
    out, boxes = np.zeros((H, W), bool), tiles(img, tile)
    with ThreadPoolExecutor(workers) as ex:
        for b, m in zip(boxes, ex.map(lambda b: sam_call(img.crop(b), prompt, min_conf), boxes)):
            out[b[1]:b[3], b[0]:b[2]] |= m
    return out, len(boxes)


def gt_mask(root, split_dir, img_id, code, shape):
    hits = glob.glob(os.path.join(root, "*Groundtruths", split_dir, "*", f"{img_id}_{code}.tif"))
    if not hits:  # IDRiD leaves the file out when that lesion type is absent
        return np.zeros(shape, bool)
    a = np.array(Image.open(hits[0]))
    return (a.max(-1) if a.ndim == 3 else a) > 0


def score(pred, gt):
    """Pixel overlap plus lesion-level counts (a GT lesion counts as found if any predicted pixel touches it)."""
    tp, fp, fn = (pred & gt).sum(), (pred & ~gt).sum(), (~pred & gt).sum()
    gt_lab, n_gt = ndimage.label(gt)
    pr_lab, n_pr = ndimage.label(pred)
    found = len(np.unique(gt_lab[pred & gt]))
    false_blobs = n_pr - len(np.unique(pr_lab[pred & gt]))
    return dict(dice=2 * tp / (2 * tp + fp + fn) if gt.any() or pred.any() else np.nan,
                precision=tp / (tp + fp) if pred.any() else np.nan,
                recall=tp / (tp + fn) if gt.any() else np.nan,
                gt_lesions=n_gt, lesions_found=found, false_blobs=false_blobs,
                gt_present=bool(gt.any()), pred_present=bool(pred.any()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help='the IDRiD "A. Segmentation" folder')
    ap.add_argument("--split", choices=["train", "test"], default="test")
    ap.add_argument("--out", default="sam_out")
    ap.add_argument("--tile", type=int, default=1024, help="tile size in pixels; 0 sends the whole image")
    ap.add_argument("--min-conf", type=float, default=0.0)
    ap.add_argument("--limit", type=int, default=None, help="first N images only, for a pilot run")
    ap.add_argument("--prompts", type=int, default=1, help="prompts per lesion type (1 or 2)")
    ap.add_argument("--max-requests", type=int, default=60, help="hard cap: stop before exceeding this many requests")
    args = ap.parse_args()

    split_dir = "a. Training Set" if args.split == "train" else "b. Testing Set"
    images = sorted(glob.glob(os.path.join(args.root, "*Original Images", split_dir, "*.jpg")))[: args.limit]
    os.makedirs(os.path.join(args.out, "masks"), exist_ok=True)

    rows, calls, stopped = [], 0, False
    for path in images:
        if stopped:
            break
        img_id = os.path.splitext(os.path.basename(path))[0]
        img = Image.open(path).convert("RGB")
        for code, prompts in LESIONS.items():
            gt = gt_mask(args.root, split_dir, img_id, code, img.size[::-1])
            for k, prompt in enumerate(prompts[: args.prompts]):
                cache = os.path.join(args.out, "masks", f"{img_id}_{code}_p{k}_t{args.tile}.png")
                if os.path.exists(cache):
                    pred = np.array(Image.open(cache)) > 0
                else:
                    need = len(tiles(img, args.tile)) if args.tile else 1
                    if calls + need > args.max_requests:
                        print(f"stopping: the next prompt needs {need} requests and the cap is {args.max_requests}")
                        stopped = True
                        break
                    pred, n = sam_mask(img, prompt, args.tile, args.min_conf)
                    calls += n
                    Image.fromarray(pred).save(cache)
                rows.append(dict(image=img_id, lesion=code, prompt=prompt, **score(pred, gt)))
                print(img_id, code, f"{prompt!r}", f"dice={rows[-1]['dice']:.3f}", f"| requests so far {calls}")
            if stopped:
                break

    df = pd.DataFrame(rows)
    tag = f"{args.split}_t{args.tile}"
    df.to_csv(os.path.join(args.out, f"sam_scores_{tag}.csv"), index=False)
    has = df[df.gt_present]
    summary = has.groupby(["lesion", "prompt"]).agg(
        images=("image", "count"), dice=("dice", "mean"), precision=("precision", "mean"),
        recall=("recall", "mean"), lesions_found=("lesions_found", "sum"),
        gt_lesions=("gt_lesions", "sum"), false_blobs=("false_blobs", "sum"))
    summary["lesion_recall"] = summary.lesions_found / summary.gt_lesions
    # images with no GT lesion of that type: how often does SAM still claim one?
    summary["false_alarm_rate"] = df[~df.gt_present].groupby(["lesion", "prompt"]).pred_present.mean()
    print("\n", summary.round(3).to_string())
    print(f"\n{calls} SAM requests this run (about ${calls * 0.0025:.2f}); cached masks cost nothing")
    summary.to_csv(os.path.join(args.out, f"sam_summary_{tag}.csv"))


if __name__ == "__main__":
    main()
