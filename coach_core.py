"""Retake coach: is this fundus photo good enough to grade, and if not, what should the photographer fix?

It judges photo quality only and makes no diagnosis.

- Detection: Meta's DINOv2 (facebook/dinov2-base, frozen) runs locally. Four small logistic heads read its
  features plus three simple image checks: not good enough, focus, aim, obstruction. The heads are trained on
  DeepDRiD's photo-quality labels (CC BY-SA 4.0, https://github.com/deepdrdoc/DeepDRiD).
- Guard: an energy on the same features (a Gaussian model of ordinary fundus photos, i.e. Mahalanobis distance)
  flags inputs that do not look like a regular fundus photo, such as a widefield image or a random picture.
- Where: the disc and macula locator (DINOv2 patch tokens) gives aim directions; the obstruction map (the
  obstruction head's own per-patch split, on DINOv3 patch tokens) says where a shadow or glare is.
- Instructions: standard sentences built from those measured facts, on this computer.
- Coach's note: Muse Spark (Meta Model API) writes a short note at the end of a session from the session's counts in
  words; no photo is sent. A hard cap on requests and dollars applies; past the cap, or on any error, a fixed summary
  is used. (Wording: Muse Spark can also rewrite a single instruction; the app does not use that, see the README.)
"""
import base64
import io
import json
import re
from collections import Counter

import numpy as np
from PIL import Image
from scipy import ndimage

S = 448
PROBLEMS = ("focus", "aim", "obstruction")
DESCRIBE = {
    "focus": "out of focus or hazy: the small vessels are not sharp",
    "aim": "badly aimed: the optic disc and the macula are not both near the centre of the field",
    "obstruction": "partly covered: eyelid or eyelash shadow, glare, reflections or dust over the retina",
}
TEMPLATES = {
    "focus": "Refocus until the small vessels look sharp, then retake.",
    "aim": "Re-aim so the view is centred on the macula or on the optic disc, then retake.",
    "obstruction": "Ask the patient to open wide (lift the eyelid if needed), wipe the lens, then retake.",
}
GENERIC = "Not good enough to grade: check focus, aim and eyelids, then retake."
MINOR = {"focus": "Focus could be sharper", "aim": "The aim is a little off centre", "obstruction": "Part of the view is shaded"}


def keep_text(minor, aim_text=None):
    if not minor:
        return "Good enough to grade. Move on to the next photo."
    where = aim_text.split(". ")[0] if (minor[0] == "aim" and aim_text and not aim_text.startswith(("Centre", "Re-aim"))) else MINOR[minor[0]]
    return f"Good enough to grade. {where}; this photo is still usable."
NOT_FUNDUS = "This does not look like a regular fundus photo. Check the camera mode or the file."
MEAN, STD = np.array([0.485, 0.456, 0.406]), np.array([0.229, 0.224, 0.225])


# ---------- image preparation (identical to the pipeline the heads were trained on) ----------

def _open(src):
    if isinstance(src, Image.Image):
        return src.convert("RGB")
    if isinstance(src, np.ndarray):
        return Image.fromarray(src).convert("RGB")
    return Image.open(src).convert("RGB")


def load_square(src, geom=False):
    """Crop to the retina, pad to a square, resize to 448 px. Returns float32 in [0, 1]. src: path, PIL image or uint8 array.
    geom=True also returns how original pixel coordinates map into the square (see to_square)."""
    img = _open(src)
    W0, H0 = img.size
    img.thumbnail((1024, 1024))
    a = np.asarray(img)
    ys, xs = np.nonzero(a.max(-1) > 15)
    y0 = x0 = 0
    if len(ys):                                   # an all-black frame keeps its full size
        y0, x0 = ys.min(), xs.min()
        a = a[y0:ys.max() + 1, x0:xs.max() + 1]
    h, w = a.shape[:2]
    side = max(h, w)
    sq = np.zeros((side, side, 3), np.uint8)
    sq[(side - h) // 2:(side - h) // 2 + h, (side - w) // 2:(side - w) // 2 + w] = a
    x = np.asarray(Image.fromarray(sq).resize((S, S), Image.BILINEAR)).astype(np.float32) / 255
    if not geom:
        return x
    return x, dict(sx=img.size[0] / W0, sy=img.size[1] / H0, x0=int(x0), y0=int(y0),
                   px=(side - w) // 2, py=(side - h) // 2, side=side)


def to_square(X, Y, g):
    """Original pixel (X, Y) -> position in the square photo, as fractions (0..1) of its width and height."""
    return (X * g["sx"] - g["x0"] + g["px"]) / g["side"], (Y * g["sy"] - g["y0"] + g["py"]) / g["side"]


def retina_mask(x):
    return x.max(-1) > 15 / 255


def simple_checks(x):
    """Sharpness (Laplacian variance), brightness and contrast of the green channel inside the retina."""
    g, m = x[..., 1], ndimage.binary_erosion(retina_mask(x), iterations=8)
    if m.sum() < 100:                             # nearly black frame: measure the whole picture
        m = np.ones_like(m)
    return [ndimage.laplace(g)[m].var(), g[m].mean(), g[m].std()]


class Encoder:
    """Frozen DINOv2 or DINOv3: class token + mean patch token (2 x 768 numbers per photo for the base models).
    Register tokens (DINOv3 has 4) sit between the class token and the patches and are skipped."""

    def __init__(self, name="facebook/dinov2-base", device=None, dtype=None):
        import torch
        from transformers import AutoModel
        self.torch, self.name = torch, name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = dtype or (torch.float16 if self.device == "cuda" else torch.float32)
        self.model = AutoModel.from_pretrained(name).to(self.device, self.dtype).eval()
        ps = self.model.config.patch_size
        self.patch = ps[0] if isinstance(ps, (list, tuple)) else ps

    def __call__(self, xs, bs=32, patches=False):
        """patches=True also returns the patch tokens as a (n, g, g, 768) float16 grid (g = 448 / patch size)."""
        out, grids = [], []
        g = S // self.patch
        with self.torch.no_grad():
            for i in range(0, len(xs), bs):
                b = self.torch.tensor((np.stack(xs[i:i + bs]) - MEAN) / STD).permute(0, 3, 1, 2)
                h = self.model(pixel_values=b.to(self.device, self.dtype)).last_hidden_state
                p = h[:, h.shape[1] - g * g:]                       # patch tokens are the last g*g
                out.append(self.torch.cat([h[:, 0], p.mean(1)], 1).float().cpu())
                if patches:
                    grids.append(p.reshape(len(b), g, g, -1).half().cpu())
        X = self.torch.cat(out).numpy()
        return (X, self.torch.cat(grids).numpy()) if patches else X


def describe(srcs, encoder, bs=32, workers=8):
    """Photos -> (DINOv2 features, simple checks). Loads in parallel, encodes in batches."""
    from concurrent.futures import ThreadPoolExecutor
    X, C = [], []
    for i in range(0, len(srcs), bs):
        with ThreadPoolExecutor(workers) as ex:
            xs = list(ex.map(load_square, srcs[i:i + bs]))
        X.append(encoder(xs, bs))
        C += [simple_checks(x) for x in xs]
    return np.concatenate(X), np.array(C)


# ---------- detector heads and energy guard ----------

def gaussian(X):
    """Mean and whitening matrix of a shrunk (Ledoit-Wolf) Gaussian over feature rows."""
    from sklearn.covariance import LedoitWolf
    lw = LedoitWolf().fit(np.asarray(X, np.float64))
    lam, V = np.linalg.eigh(lw.covariance_)                    # whiten from the covariance itself:
    lam = np.maximum(lam, lam.max() * 1e-6)                    # tiny directions are floored, never inverted to inf
    return lw.location_, V / np.sqrt(lam)


def energy(X, mean, white):
    return (((X - mean) @ white) ** 2).sum(1) / X.shape[1]


def camera_component(X_cam, name, factor=1.1):
    """A Gaussian over one camera's own photos. Its limit is the largest leave-one-out energy among them, times factor,
    so each photo is judged by a model that never saw it."""
    X_cam = np.asarray(X_cam, np.float64)
    assert len(X_cam) >= 5, "add at least 5 ordinary photos from the camera"
    loo = []
    for k in range(len(X_cam)):
        m, w = gaussian(np.delete(X_cam, k, 0))
        loo.append(energy(X_cam[k:k + 1], m, w)[0])
    m, w = gaussian(X_cam)
    return dict(name=name, mean=m, white=w, thr=float(max(loo) * factor))


def _sigmoid(z):
    return 1 / (1 + np.exp(-np.clip(z, -50, 50)))


def _logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


class Coach:
    HEADS = ("retake",) + PROBLEMS
    RULES = ("overall", "problems", "both")   # what triggers a retake: the overall head, any problem head, or both

    def __init__(self, mu, sd, W, b, thr, g_mean=None, g_white=None, g_thr=None, rule="overall", locator=None,
                 backbone="facebook/dinov2-base"):
        assert rule in self.RULES
        self.mu, self.sd, self.W, self.b, self.thr, self.rule = mu, sd, W, b, np.asarray(thr, float), rule
        self.backbone = backbone                 # the encoder whose features these heads read
        self.g_mean, self.g_white, self.g_thr = g_mean, g_white, g_thr
        self.cameras = []                         # onboarded cameras: dicts of name, mean, white, thr
        self.locator = locator                   # optional Locator; saved in its own file (use attach_locator)

    def attach_locator(self, locator):
        """Use the locator only if it was fitted on the same encoder's patch tokens."""
        if locator.backbone != self.backbone:
            print(f"locator reads {locator.backbone}, heads read {self.backbone}: aim directions off")
            return False
        self.locator = locator
        return True

    @classmethod
    def fit(cls, X, checks, labels, C=0.05):
        """labels: dict head -> bool array; C: one value or a dict head -> value (smaller = stronger regularization).
        Thresholds start at 0.5; set coach.thr from validation photos."""
        from sklearn.linear_model import LogisticRegression
        mu, sd = checks.mean(0), checks.std(0) + 1e-12
        F = np.c_[X, (checks - mu) / sd]
        W, b = [], []
        for h in cls.HEADS:
            clf = LogisticRegression(C=C[h] if isinstance(C, dict) else C, max_iter=5000).fit(F, labels[h])
            W.append(clf.coef_[0])
            b.append(clf.intercept_[0])
        return cls(mu, sd, np.array(W), np.array(b), np.full(len(cls.HEADS), 0.5))

    def probs(self, X, checks):
        """(n, 4) probabilities in HEADS order."""
        return _sigmoid(np.c_[X, (checks - self.mu) / self.sd] @ self.W.T + self.b)

    def set_guard(self, X_ref, X_cal, quantile=0.99):
        """Energy guard fitted on ordinary fundus photos (X_ref); threshold = the 99th percentile on other ones (X_cal)."""
        self.g_mean, self.g_white = gaussian(X_ref)
        self.g_thr = float(np.quantile(self.energy(X_cal), quantile))
        return self

    def energy(self, X):
        """Mahalanobis energy per feature dimension: about 1 for typical fundus photos, large for unfamiliar inputs."""
        return energy(X, self.g_mean, self.g_white)

    def add_camera(self, X_cam, name, factor=1.1):
        """Onboard a new camera from its own ordinary photos, no labels: a second Gaussian over just those photos.
        A photo passes the guard if either the general model or a camera's model finds it familiar."""
        self.cameras.append(camera_component(X_cam, name, factor))
        return self

    def familiar(self, X):
        """(n,) bool: does each photo look like a regular fundus photo to the general model or to an onboarded camera?"""
        if self.g_mean is None:
            return np.ones(len(X), bool)
        ok = self.energy(X) <= self.g_thr
        for c in self.cameras:
            ok |= energy(X, c["mean"], c["white"]) <= c["thr"]
        return ok

    def verdict(self, X, checks, patches=None, max_problems=2):
        """patches (from Encoder(..., patches=True)) adds measured disc and macula positions when a locator is set."""
        P = self.probs(X, checks)
        E = self.energy(X) if self.g_mean is not None else np.zeros(len(X))
        OK = self.familiar(X)
        L = self.locator.locate(patches) if (self.locator is not None and patches is not None) else [None] * len(X)
        out = []
        for p, e, loc, ok in zip(P, E, L, OK):
            fundus = bool(ok)
            margin = _logit(p[1:]) - _logit(self.thr[1:])          # how far each problem head is past its threshold
            overall, any_problem = bool(p[0] >= self.thr[0]), bool((margin > 0).any())
            retake = {"overall": overall, "problems": any_problem, "both": overall and any_problem}[self.rule]
            order = [int(i) for i in np.argsort(-margin)]
            problems = [PROBLEMS[i] for i in order if margin[i] > 0][:max_problems] if retake else []
            if retake and not problems:                              # always give one concrete thing to fix
                problems = [PROBLEMS[order[0]]]
            v = dict(looks_like_fundus=fundus, retake=retake and fundus, problems=problems if fundus else [], minor=[],
                     p_not_good=float(p[0]), scores={k: float(v) for k, v in zip(PROBLEMS, p[1:])}, energy=float(e))
            if loc is not None and fundus:
                dd = self.locator.dd
                # a kept photo gets an aim note only when the aim head and the measured positions both say off centre
                both_found = loc["disc"] is not None and loc["macula"] is not None
                if not retake and margin[1] > 0 and both_found and aim_advice(loc, dd) is not None:
                    v["minor"] = ["aim"]
                v.update(disc=loc["disc"], macula=loc["macula"], dd=dd, positions=positions_text(loc, dd),
                         aim_text=aim_instruction(loc, dd) if "aim" in v["problems"] + v["minor"] else None,
                         **eye_and_field(loc, dd))
            out.append(v)
        return out

    def save(self, path):
        g = {} if self.g_mean is None else dict(g_mean=self.g_mean, g_white=self.g_white.astype(np.float32), g_thr=self.g_thr)
        for i, c in enumerate(self.cameras):
            g.update({f"cam{i}_name": np.array(c["name"]), f"cam{i}_mean": c["mean"], f"cam{i}_white": c["white"].astype(np.float32),
                      f"cam{i}_thr": c["thr"]})
        np.savez(path, mu=self.mu, sd=self.sd, W=self.W, b=self.b, thr=self.thr, heads=np.array(self.HEADS),
                 rule=np.array(self.rule), backbone=np.array(self.backbone), **g)

    @classmethod
    def load(cls, path):
        z = np.load(path)
        assert tuple(z["heads"]) == cls.HEADS
        g = dict(g_mean=z["g_mean"], g_white=z["g_white"].astype(np.float64), g_thr=float(z["g_thr"])) if "g_mean" in z else {}
        c = cls(z["mu"], z["sd"], z["W"], z["b"], z["thr"], rule=str(z["rule"]) if "rule" in z else "overall",
                backbone=str(z["backbone"]) if "backbone" in z else "facebook/dinov2-base", **g)
        i = 0
        while f"cam{i}_mean" in z:
            c.cameras.append(dict(name=str(z[f"cam{i}_name"]), mean=z[f"cam{i}_mean"], white=z[f"cam{i}_white"].astype(np.float64),
                                  thr=float(z[f"cam{i}_thr"])))
            i += 1
        return c


# ---------- where are the optic disc and the macula? ----------

class Locator:
    """A linear map from DINOv2 patch tokens to two heatmaps (optic disc, fovea), fitted by ridge regression on
    IDRiD's expert-marked centres. Positions are fractions (0..1) of the square photo; dd is one disc diameter
    in the same units, so distances can be read the way DeepDRiD's field-definition score reads them."""
    NAMES = ("disc", "macula")

    def __init__(self, W, peak_min, dd, backbone="facebook/dinov2-base"):
        self.W, self.peak_min, self.dd = np.asarray(W, np.float32), np.asarray(peak_min, float), float(dd)
        self.backbone = backbone

    @staticmethod
    def design(T):
        """(n, g, g, d) tokens -> (n, g*g, d+1) with a constant column for the bias."""
        T = np.asarray(T, np.float32).reshape(T.shape[0], -1, T.shape[-1])
        return np.concatenate([T, np.ones(T.shape[:2] + (1,), np.float32)], -1)

    def heatmaps(self, T, bs=32):
        g = T.shape[1]
        return np.concatenate([(self.design(T[i:i + bs]) @ self.W).reshape(-1, g, g, len(self.NAMES))
                               for i in range(0, len(T), bs)])

    def locate(self, T):
        out = []
        for hm in self.heatmaps(T):
            g, found = hm.shape[0], {}
            for k, name in enumerate(self.NAMES):
                m = ndimage.uniform_filter(hm[..., k], 3)
                i, j = np.unravel_index(np.argmax(m), m.shape)
                i0, i1, j0, j1 = max(i - 1, 0), min(i + 2, g), max(j - 1, 0), min(j + 2, g)
                win = np.clip(hm[i0:i1, j0:j1, k], 0, None) + 1e-9          # sub-patch refinement
                ii, jj = np.mgrid[i0:i1, j0:j1]
                u, v = ((jj * win).sum() / win.sum() + 0.5) / g, ((ii * win).sum() / win.sum() + 0.5) / g
                found[name] = (float(u), float(v)) if m[i, j] >= self.peak_min[k] else None
                found[name + "_peak"] = float(m[i, j])
            out.append(found)
        return out

    def save(self, path):
        np.savez(path, W=self.W, peak_min=self.peak_min, dd=self.dd, backbone=np.array(self.backbone))

    @classmethod
    def load(cls, path):
        z = np.load(path)
        return cls(z["W"], z["peak_min"], float(z["dd"]), str(z["backbone"]) if "backbone" in z else "facebook/dinov2-base")



# ---------- where is the obstruction? the obstruction head's own patch map ----------

DINOV3 = "facebook/dinov3-vitb16-pretrain-lvd1689m"


def patch_mask(x, g):
    """Which of the g x g patches are mostly retina, not the black surround."""
    k = S // g
    return retina_mask(x).reshape(g, k, g, k).mean((1, 3)) > 0.5


def smooth_on(r, m, sigma=1.0):
    """Average each patch with its retina neighbours only, so the black rim does not bleed in."""
    num = ndimage.gaussian_filter(np.where(m, r, 0.0), sigma)
    den = ndimage.gaussian_filter(m.astype(float), sigma)
    return np.where(m, num / np.maximum(den, 1e-6), 0.0)


def _overlay_png(area, light=False):
    """The marked area as a see-through blue layer with a white edge, the size of the square photo. A large area gets a
    lighter fill so the photo underneath stays visible."""
    edge = area & ~ndimage.binary_erosion(area, iterations=2)
    rgba = np.zeros(area.shape + (4,), np.uint8)
    rgba[area] = (40, 140, 255, 45 if light else 105)
    rgba[edge] = (255, 255, 255, 230)
    buf = io.BytesIO()
    Image.fromarray(rgba, "RGBA").save(buf, "PNG", optimize=True)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


class ObstructionMap:
    """Where the obstruction is, read from the obstruction head itself.
    Part of the head's score is its weight on the mean patch token, so that part splits exactly into one number per
    patch. Each patch is compared with the same place on a typical clean photo (DeepDRiD artifact score 0), and patches
    above what 1% of clean patches reach are marked. On DeepDRiD validation photos with DINOv3, the marked share follows
    the artifact score (Spearman 0.81), and the strongest spot lands on a pasted shadow 85% and a pasted glare 80% of
    the time (chance 7%); results/obstruction_map_report.txt."""

    HINT_SHARE = 0.054   # measured: 95% of artifact-0 DeepDRiD validation photos mark less (obstruction_map_build_report.txt)

    def __init__(self, w, base, thr, backbone=DINOV3, min_share=0.0, hint_share=None):
        self.w, self.base, self.thr = np.asarray(w, np.float32), np.asarray(base, np.float32), float(thr)
        self.backbone, self.min_share = backbone, float(min_share)   # smaller marked areas are what clean photos show
        # on a retake for another reason, a marked area larger than this is shown too: clear photos rarely mark this much
        self.hint_share = float(self.HINT_SHARE if hint_share is None else hint_share)

    @staticmethod
    def head_weights(coach):
        """The obstruction head's weights on the mean patch token (features are [class token, mean patch, 3 checks])."""
        n = (coach.W.shape[1] - 3) // 2
        return np.asarray(coach.W[Coach.HEADS.index("obstruction"), n:2 * n], np.float32)

    def raw(self, grids):
        """(..., g, g, d) patch tokens -> (..., g, g) share of the obstruction score from each patch."""
        return np.asarray(grids, np.float32) @ self.w

    @classmethod
    def fit(cls, w, maps, masks, backbone, quantile=99.0, share_quantile=None):
        """maps: raw maps of clean photos (n, g, g); masks: their retina patches.
        Sets the typical map and the level a patch must pass (what 1% of clean patches reach). share_quantile, if given,
        also sets the smallest marked area worth reporting (e.g. 95: what 95% of clean photos stay under). The app runs
        the map only after the obstruction check has asked for a retake, so by default there is no such floor."""
        masks = np.asarray(masks)
        base = np.median(maps, 0)
        smoothed = np.array([smooth_on(m - base, pm) for m, pm in zip(maps, masks)])
        thr = max(float(np.percentile(smoothed[masks], quantile)), 1e-6)
        shares = [((sm > thr) & pm).sum() / max(1, pm.sum()) for sm, pm in zip(smoothed, masks)]
        return cls(w, base, thr, backbone, float(np.percentile(shares, share_quantile)) if share_quantile else 0.0,
                   hint_share=float(np.percentile(shares, 95)))

    def strength(self, x, raw):
        """Each patch's push toward 'something is in the way', relative to a clean photo, and the retina patches."""
        pm = patch_mask(x, raw.shape[0])
        return smooth_on(raw - self.base, pm), pm

    def locate(self, x, grid, exclude=None):
        """x: the square photo; grid: its (g, g, d) patch tokens from the map's backbone.
        Returns where (side of the picture, 'centre' or 'most'), tone (darker, brighter or None), share and an overlay.
        exclude: (u, v, r) leaves out patches within r of (u, v), e.g. around the macula, which is naturally darker."""
        return self.locate_raw(x, self.raw(grid), exclude)

    def locate_raw(self, x, raw, exclude=None):
        """The same from an already computed raw map (g, g)."""
        s, pm = self.strength(x, raw)
        g = s.shape[0]
        marked = (s > self.thr) & pm
        if exclude:
            c = (np.arange(g) + 0.5) / g
            marked &= np.hypot(c[None, :] - exclude[0], c[:, None] - exclude[1]) > exclude[2]
        share = float(marked.sum() / max(1, pm.sum()))
        out = dict(where=None, tone=None, share=share, overlay=None)
        if not marked.any() or share <= self.min_share:
            return out
        lab, _ = ndimage.label(marked)                  # the marked area that holds the strongest spot
        top = np.unravel_index(np.argmax(np.where(marked, s, -np.inf)), s.shape)
        cy, cx = ndimage.center_of_mass(lab == lab[top])
        u, v = (cx + 0.5) / g, (cy + 0.5) / g
        out["where"] = "most" if share >= 0.5 else "centre" if np.hypot(u - 0.5, v - 0.5) < 0.15 else side_of(u, v)
        ret = retina_mask(x)
        up = np.array(Image.fromarray(s.astype(np.float32)).resize((S, S), Image.BILINEAR))
        k = S / g                                          # the marked patches at photo size, with rounded edges
        area = ndimage.gaussian_filter(np.array(Image.fromarray(marked.astype(np.uint8) * 255).resize((S, S), Image.NEAREST)) / 255.0,
                                       k / 2) > 0.5
        area &= ret
        rest, lum = ret & ~area, x.mean(-1)
        if area.any() and rest.sum() > 0.1 * ret.sum():          # tone of the strongest quarter of the area, against the rest
            core = area & (up >= np.percentile(up[area], 75))
            ratio = lum[core].mean() / max(float(np.median(lum[rest])), 1e-6)
            out["tone"] = "darker" if ratio < 0.85 else "brighter" if ratio > 1.15 else None
        out["overlay"] = _overlay_png(area, light=share >= 0.35)
        return out

    def save(self, path):
        np.savez(path, w=self.w, base=self.base, thr=self.thr, min_share=self.min_share, hint_share=self.hint_share,
                 backbone=np.array(self.backbone))

    @classmethod
    def load(cls, path):
        z = np.load(path)
        return cls(z["w"], z["base"], float(z["thr"]), str(z["backbone"]), float(z["min_share"]) if "min_share" in z else 0.0,
                   float(z["hint_share"]) if "hint_share" in z else None)


SIDES = ["right", "lower right", "bottom", "lower left", "left", "upper left", "top", "upper right"]


def side_of(u, v):
    """Which edge of the picture a point is toward, in the words a technician sees on screen."""
    ang = np.degrees(np.arctan2(v - 0.5, u - 0.5)) % 360          # 0 = right, 90 = bottom (image y grows downward)
    return SIDES[int(((ang + 22.5) % 360) // 45)]


def eye_and_field(loc, dd):
    """Which eye and which field, from the measured layout.
    Eye: in a right eye the optic disc sits to the right of the macula (94% right on DeepDRiD test photos).
    Field: whichever of the two is nearer the centre; with only one found, that one if it is near the centre."""
    d, m = loc.get("disc"), loc.get("macula")
    eye = ("right" if d[0] > m[0] else "left") if (d and m) else None
    dist = {k: np.hypot(p[0] - 0.5, p[1] - 0.5) for k, p in (("disc", d), ("macula", m)) if p}
    if len(dist) == 2:
        field = min(dist, key=dist.get)
    elif len(dist) == 1 and next(iter(dist.values())) <= 2 * dd:
        field = next(iter(dist))
    else:
        field = None
    return dict(eye=eye, field=field)


def positions_text(loc, dd):
    """Measured positions as plain facts, e.g. 'the optic disc is near the centre; the macula is near the right edge'."""
    parts = []
    for name, label in [("disc", "the optic disc"), ("macula", "the macula")]:
        p = loc.get(name)
        if p is None:
            parts.append(f"{label} is not in the picture")
        elif np.hypot(p[0] - 0.5, p[1] - 0.5) <= 2 * dd:
            parts.append(f"{label} is near the centre")
        else:
            parts.append(f"{label} is near the {side_of(*p)} edge")
    return "; ".join(parts)


def aim_advice(loc, dd):
    """What is off for the view this photo is, measured, as (fact, action); None when the layout fits its view.
    Screening takes two views of each eye, one centred on the macula and one on the optic disc. So only the structure
    the view centres on belongs in the middle (within 2 disc diameters); the other sits about 2.5 disc diameters away
    and must stay well inside the picture (within 3). The view is whichever of the two is nearer the centre."""
    name = {"disc": "optic disc", "macula": "macula"}
    found = {n: loc[n] for n in ("disc", "macula") if loc.get(n)}
    if not found:
        return ("Neither the optic disc nor the macula shows clearly", "Re-aim so the view is centred on the macula or on the optic disc")
    dist = {n: np.hypot(p[0] - 0.5, p[1] - 0.5) for n, p in found.items()}
    view = min(dist, key=dist.get)
    other = "macula" if view == "disc" else "disc"
    if dist[view] > 2 * dd:
        side = side_of(*found[view])
        if other in found:
            return (f"Neither the macula nor the optic disc is centred; the {name[view]} is nearest, toward the {side}",
                    f"Re-aim so the {name[view]} moves to the middle")
        return (f"The {name[view]} sits toward the {side} and the {name[other]} is not in the picture",
                "Re-aim so the view is centred on the macula or on the optic disc")
    if other not in found:
        return (f"The {name[view]} is centred but the {name[other]} is not in the picture",
                f"Keep the {name[view]} in the middle and turn slightly toward the {name[other]} so it comes into view")
    if dist[other] > 3 * dd:      # it normally sits about 2.5 disc diameters from the centre; further out, it is leaving the view
        return (f"The {name[view]} is centred but the {name[other]} is close to the {side_of(*found[other])} edge",
                f"Re-aim slightly so the {name[other]} comes further into the picture, keeping the {name[view]} near the middle")
    return None


def aim_instruction(loc, dd):
    """A direction measured from the photo, never guessed, for the view the photo is (macula view or disc view)."""
    a = aim_advice(loc, dd)
    if a:
        return f"{a[0]}. {a[1]}, then retake."
    found = {n: loc[n] for n in ("disc", "macula") if loc.get(n)}
    view = "optic disc" if min(found, key=lambda n: np.hypot(found[n][0] - 0.5, found[n][1] - 0.5)) == "disc" else "macula"
    # the aim check says off, but the measured layout already fits the view: only a small adjustment is needed
    return f"The {view} view is close. Fine-tune so the {view} sits exactly in the middle of the picture, then retake."


# ---------- wording: Muse Spark, capped, with templates as the fallback ----------

PROMPT = """You coach a technician who takes retinal (fundus) photos for eye screening. An automatic photo-quality
check found that a photo needs a retake because it is:
{found}
{facts}The standard instruction for this photo is:
"{standard}"
Rewrite it as ONE instruction for the technician: plain words, starting with a verb, at most 25 words. Keep every action
and every measured place in it, and add nothing else. It is not something to say to the patient. Never guess a position
or a direction. Do not mention disease or diagnosis.
Reply with JSON only: {{"instruction": "..."}}"""
PHOTO_NOTE = "The photo is attached. Look at it only to say where a shadow, glare or haze is.\n"


OBSTRUCTION_ADVICE = {
    "darker": ("Dark shadow", "Ask the patient to open wide (lift the eyelid if needed) and centre the camera on the pupil, then retake."),
    "brighter": ("Glare or haze", "Wipe the lens and adjust the working distance until it clears, then retake."),
    None: ("Something is in the way", TEMPLATES["obstruction"]),
}


def obstruction_place(ob):
    """'toward the upper right of the photo', 'in the centre of the photo', 'over most of the photo', or None."""
    w = (ob or {}).get("where")
    if not w:
        return None
    return "over most of the photo" if w == "most" else "in the centre of the photo" if w == "centre" else f"toward the {w} of the photo"


def obstruction_amount(ob):
    sh = ob.get("share", 0)
    return "most of it" if sh >= 0.5 else "a large part of it" if sh >= 0.35 else "about a quarter of it" if sh >= 0.15 else "a small part of it"


def obstruction_instruction(ob):
    """A measured place plus advice for that kind of obstruction; the standard advice when nothing was marked."""
    place = obstruction_place(ob)
    if not place:
        return TEMPLATES["obstruction"]
    what, advice = OBSTRUCTION_ADVICE[ob.get("tone")]
    return f"{what} {place}. {advice}"


def obstruction_facts(ob):
    """The measured facts Muse Spark gets instead of the photo."""
    place = obstruction_place(ob)
    if not place:
        return None
    what = {"darker": "an area darker than the rest", "brighter": "an area brighter or hazier than the rest",
            None: "an area that looks covered"}[ob.get("tone")]
    return f"{what} {place}" + ("" if ob["where"] == "most" else f", covering {obstruction_amount(ob)}")


def obstruction_summary(ob):
    """For the page: 'Toward the upper right, a small part of the photo, darker than the rest'."""
    place = obstruction_place(ob)
    if not place:
        return "Not pinned down: no part of the photo stands out"
    tone = {"darker": ", darker than the rest", "brighter": ", brighter than the rest", None: ""}[ob.get("tone")]
    if ob["where"] == "most":
        return "Over most of the photo" + tone
    return place.replace(" of the photo", "").capitalize() + ", " + obstruction_amount(ob).replace(" of it", " of the photo") + tone


def template(problems, aim_text=None, obstruction=None):
    if not problems:
        return GENERIC
    parts = [aim_text if (p == "aim" and aim_text) else obstruction_instruction(obstruction) if (p == "obstruction" and obstruction)
             else TEMPLATES[p] for p in problems]
    return " ".join([t.replace(", then retake.", ".") for t in parts[:-1]] + parts[-1:])   # say "then retake" once


def _data_url(src, px):
    img = _open(src)
    img.thumbnail((px, px))
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=90)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


def _json_field(text, key):
    try:
        return str(json.loads(text[text.index("{"):text.rindex("}") + 1])[key]).strip() or None
    except (ValueError, KeyError, TypeError):
        return None


class Wording:
    PRICE_IN, PRICE_OUT = 1.25e-6, 4.25e-6       # muse-spark-1.3, Standard mode, $ per token

    def __init__(self, client=None, model="muse-spark-1.3", effort="minimal", max_requests=20, max_usd=0.10, px=768):
        self.client, self.model, self.effort, self.px = client, model, effort, px
        self.max_requests, self.max_usd = max_requests, max_usd
        self.n, self.spent, self.worst = 0, 0.0, 0.01  # assume a call may cost 1 cent until a dearer one is seen

    def can_call(self):
        return self.client is not None and self.n < self.max_requests and self.spent + self.worst <= self.max_usd

    def prompt(self, problems, positions=None, obstruction=None, send_photo=False, aim_text=None):
        """Exactly the text Muse Spark gets. Without send_photo, that text is all it gets."""
        found = "\n".join(f"- {DESCRIBE[p]}" for p in problems)
        facts = ""
        if positions and "aim" in problems:         # positions only matter, and only invite comment, for aim
            facts += f"Measured positions in the picture: {positions}.\n"
        if "obstruction" in problems and obstruction_facts(obstruction):
            facts += f"Measured where the obstruction is: {obstruction_facts(obstruction)}.\n"
        if send_photo:
            facts += PHOTO_NOTE
        return PROMPT.format(found=found, facts=facts, standard=template(problems, aim_text, obstruction))

    def __call__(self, src, problems, positions=None, aim_text=None, obstruction=None, send_photo=False):
        """positions: measured facts such as 'the optic disc is near the centre; the macula is near the right edge'.
        obstruction: what ObstructionMap.locate measured. src is read only when send_photo is True."""
        fallback = template(problems, aim_text, obstruction)
        if not problems or not self.can_call():
            return dict(text=fallback, source="template", cost=0.0)
        content = [{"type": "text", "text": self.prompt(problems, positions, obstruction, send_photo, aim_text)}]
        if send_photo:
            content.append({"type": "image_url", "image_url": {"url": _data_url(src, self.px)}})
        self.n += 1
        try:
            resp = self.client.chat.completions.create(
                model=self.model, extra_body={"reasoning_effort": self.effort}, messages=[{"role": "user", "content": content}])
        except Exception as e:  # network, quota, bad key: the technician still gets an instruction
            return dict(text=fallback, source=f"template (Muse Spark error: {type(e).__name__})", cost=0.0)
        cost = resp.usage.prompt_tokens * self.PRICE_IN + resp.usage.completion_tokens * self.PRICE_OUT
        self.spent += cost
        self.worst = max(self.worst, cost)
        text = _json_field(resp.choices[0].message.content or "", "instruction")
        return dict(text=text or fallback, source="muse-spark" if text else "template (unparsed reply)", cost=cost)


# ---------- the coach's note: a session's results in words ----------

VIEW_NAMES = {("right", "macula"): "right eye centred on the macula", ("right", "disc"): "right eye centred on the optic disc",
              ("left", "macula"): "left eye centred on the macula", ("left", "disc"): "left eye centred on the optic disc"}
KIND = {"darker": "dark shadow", "brighter": "glare or haze", None: "something in the way"}
HABITS = {
    "focus": "Refocus on the small vessels before every shot.",
    "aim": "Before pressing the shutter, check that the optic disc and the macula are both near the centre.",
    "darker": "Ask each patient to open wide before the shot, and centre the camera on the pupil.",
    "brighter": "Wipe the lens between patients and hold the working distance steady.",
    None: "Ask each patient to open wide, and wipe the lens between patients.",
}
NOTE_PROMPT = """You coach a technician who takes retinal (fundus) photos for eye screening. An automatic photo-quality
check found this in today's session:
{facts}
Write a short note to the technician: at most 3 sentences and 60 words. Say what went wrong most often and one concrete
habit that would prevent it. If a view has no kept photo, name it. Use only the facts above and add no numbers of your
own. It is not for the patient. Do not mention disease or diagnosis.
Reply with JSON only: {{"note": "..."}}"""


def _clean_result(r):
    """Only the fields the note reads, each from a fixed vocabulary."""
    ob = r.get("obstruction") or {}
    return dict(verdict=r.get("verdict") if r.get("verdict") in ("keep", "retake", "not_fundus", "unreadable") else "unreadable",
                problems=[p for p in (r.get("problems") or []) if p in PROBLEMS],
                where=ob.get("where") if ob.get("where") in SIDES + ["centre", "most"] else None,
                tone=ob.get("tone") if ob.get("tone") in ("darker", "brighter") else None,
                eye=r.get("eye") if r.get("eye") in ("right", "left") else None,
                field=r.get("field") if r.get("field") in ("macula", "disc") else None)


def session_facts(results):
    """A session's results as counts in plain words. This text is all Muse Spark gets for the coach's note."""
    rs = [_clean_result(r) for r in results]
    kept = [r for r in rs if r["verdict"] == "keep"]
    back = [r for r in rs if r["verdict"] == "retake"]
    away = len(rs) - len(kept) - len(back)
    lines = [f"Photos checked: {len(rs)}. Kept: {len(kept)}. Sent back for a retake: {len(back)}."
             + (f" Not regular fundus photos, or unreadable: {away}." if away else "")]
    count = Counter(p for r in back for p in r["problems"])
    tones = Counter(r["tone"] for r in back if "obstruction" in r["problems"])
    if count:
        parts = []
        for p, c in count.most_common():
            part = f"{p} {c}"
            if p == "obstruction":
                places = Counter((r["tone"], r["where"]) for r in back if "obstruction" in r["problems"] and r["where"])
                if places:
                    part += " (" + "; ".join(f"{KIND[t]} {obstruction_place(dict(where=w))} {k}" for (t, w), k in places.most_common()) + ")"
            parts.append(part)
        lines.append("Problems on the photos sent back (a photo can have more than one): " + ", ".join(parts) + ".")
    kept_views = Counter((r["eye"], r["field"]) for r in kept if r["eye"] and r["field"])
    missing = []
    if any(r["eye"] and r["field"] for r in rs):
        lines.append("Kept photos by view: " + ", ".join(f"{name} {kept_views[v]}" for v, name in VIEW_NAMES.items()) + ".")
        missing = [name for v, name in VIEW_NAMES.items() if not kept_views[v]]
    top = count.most_common(1)[0][0] if count else None
    return dict(text="\n".join("- " + l for l in lines), n=len(rs), kept=len(kept), back=len(back), top=top,
                top_count=count[top] if top else 0, top_tone=tones.most_common(1)[0][0] if tones else None, missing=missing)


def note_template(f):
    """The coach's note without Muse Spark: the same facts, in fixed sentences."""
    if not f["n"]:
        return "No photos checked yet."
    if not f["back"]:
        s = f"None of the {f['n']} photos needed a retake."
    else:
        mostly = {"darker": ", mostly dark shadows", "brighter": ", mostly glare or haze", None: ""}[f["top_tone"]]
        what = {"focus": "blur", "aim": "aim off centre", "obstruction": "something in the way"}[f["top"]]
        habit = HABITS[f["top_tone"]] if f["top"] == "obstruction" else HABITS[f["top"]]
        s = f"{f['back']} of {f['n']} photos needed a retake, most often for {what} ({f['top_count']}){mostly}. {habit}"
    if f["missing"]:
        s += " No kept photo yet for: " + "; ".join(f["missing"]) + "."
    return s


def _note(self, facts, fallback):
    """The coach's note from session_facts(...)["text"] only. A number that is not in the facts sends it back to the
    fixed summary, so the note cannot invent counts."""
    if not self.can_call():
        return dict(text=fallback, source="template", cost=0.0)
    self.n += 1
    try:
        resp = self.client.chat.completions.create(
            model=self.model, extra_body={"reasoning_effort": self.effort},
            messages=[{"role": "user", "content": [{"type": "text", "text": NOTE_PROMPT.format(facts=facts)}]}])
    except Exception as e:
        return dict(text=fallback, source=f"template (Muse Spark error: {type(e).__name__})", cost=0.0)
    cost = resp.usage.prompt_tokens * self.PRICE_IN + resp.usage.completion_tokens * self.PRICE_OUT
    self.spent += cost
    self.worst = max(self.worst, cost)
    text = _json_field(resp.choices[0].message.content or "", "note")
    if text and not set(re.findall(r"\d+", text)) <= set(re.findall(r"\d+", facts)):
        return dict(text=fallback, source="template (Muse Spark added a number)", cost=cost)
    return dict(text=text or fallback, source="muse-spark" if text else "template (unparsed reply)", cost=cost)


Wording.note = _note
