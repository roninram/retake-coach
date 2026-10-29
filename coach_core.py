"""Retake coach: is this fundus photo good enough to grade, and if not, what should the photographer fix?

It judges photo quality only and makes no diagnosis.

- Detection: Meta's DINOv2 (facebook/dinov2-base, frozen) runs locally. Four small logistic heads read its
  features plus three simple image checks: not good enough, focus, aim, obstruction. The heads are trained on
  DeepDRiD's photo-quality labels (CC BY-SA 4.0, https://github.com/deepdrdoc/DeepDRiD).
- Guard: an energy on the same features (a Gaussian model of ordinary fundus photos, i.e. Mahalanobis distance)
  flags inputs that do not look like a regular fundus photo, such as a widefield image or a random picture.
- Wording: Muse Spark (Meta Model API) writes the instruction for the problems the heads found. A hard cap on
  requests and dollars applies; past the cap, or on any error, fixed templates are used.
"""
import base64
import io
import json

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
    "aim": "Re-aim so the optic disc and the macula are both near the centre, then retake.",
    "obstruction": "Ask the patient to open wide (lift the eyelid if needed), wipe the lens, then retake.",
}
GENERIC = "Not good enough to grade: check focus, aim and eyelids, then retake."
MINOR = {"focus": "Focus could be sharper", "aim": "The aim is a little off centre", "obstruction": "Part of the view is shaded"}


def keep_text(minor, aim_text=None):
    if not minor:
        return "Good enough to grade. Move on to the next photo."
    where = aim_text.split(". ")[0] if (minor[0] == "aim" and aim_text and aim_text != TEMPLATES["aim"]) else MINOR[minor[0]]
    return f"Good enough to grade. {where}, but this photo is usable."
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
    """Frozen DINOv2: class token + mean patch token (1,536 numbers per photo)."""

    def __init__(self, name="facebook/dinov2-base", device=None):
        import torch
        from transformers import AutoModel
        self.torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = torch.float16 if self.device == "cuda" else torch.float32
        self.model = AutoModel.from_pretrained(name).to(self.device, self.dtype).eval()

    def __call__(self, xs, bs=32, patches=False):
        """patches=True also returns the patch tokens as a (n, 32, 32, 768) float16 grid, used to locate the disc and macula."""
        out, grids = [], []
        with self.torch.no_grad():
            for i in range(0, len(xs), bs):
                b = self.torch.tensor((np.stack(xs[i:i + bs]) - MEAN) / STD).permute(0, 3, 1, 2)
                h = self.model(pixel_values=b.to(self.device, self.dtype)).last_hidden_state
                out.append(self.torch.cat([h[:, 0], h[:, 1:].mean(1)], 1).float().cpu())
                if patches:
                    n = int(round((h.shape[1] - 1) ** 0.5))
                    grids.append(h[:, 1:].reshape(len(b), n, n, -1).half().cpu())
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

def _sigmoid(z):
    return 1 / (1 + np.exp(-np.clip(z, -50, 50)))


def _logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


class Coach:
    HEADS = ("retake",) + PROBLEMS
    RULES = ("overall", "problems", "both")   # what triggers a retake: the overall head, any problem head, or both

    def __init__(self, mu, sd, W, b, thr, g_mean=None, g_white=None, g_thr=None, rule="overall", locator=None):
        assert rule in self.RULES
        self.mu, self.sd, self.W, self.b, self.thr, self.rule = mu, sd, W, b, np.asarray(thr, float), rule
        self.g_mean, self.g_white, self.g_thr = g_mean, g_white, g_thr
        self.locator = locator                   # optional Locator; saved in its own file

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
        from sklearn.covariance import LedoitWolf
        lw = LedoitWolf().fit(np.asarray(X_ref, np.float64))
        lam, V = np.linalg.eigh(lw.covariance_)               # whiten from the covariance itself:
        lam = np.maximum(lam, lam.max() * 1e-6)               # tiny directions are floored, never inverted to inf
        self.g_mean, self.g_white = lw.location_, V / np.sqrt(lam)
        self.g_thr = float(np.quantile(self.energy(X_cal), quantile))
        return self

    def energy(self, X):
        """Mahalanobis energy per feature dimension: about 1 for typical fundus photos, large for unfamiliar inputs."""
        return (((X - self.g_mean) @ self.g_white) ** 2).sum(1) / X.shape[1]

    def verdict(self, X, checks, patches=None, max_problems=2):
        """patches (from Encoder(..., patches=True)) adds measured disc and macula positions when a locator is set."""
        P = self.probs(X, checks)
        E = self.energy(X) if self.g_mean is not None else np.zeros(len(X))
        L = self.locator.locate(patches) if (self.locator is not None and patches is not None) else [None] * len(X)
        out = []
        for p, e, loc in zip(P, E, L):
            fundus = self.g_thr is None or bool(e <= self.g_thr)
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
                if not retake and margin[1] > 0 and both_found and aim_instruction(loc, dd) != TEMPLATES["aim"]:
                    v["minor"] = ["aim"]
                v.update(disc=loc["disc"], macula=loc["macula"], dd=dd, positions=positions_text(loc, dd),
                         aim_text=aim_instruction(loc, dd) if "aim" in v["problems"] + v["minor"] else None)
            out.append(v)
        return out

    def save(self, path):
        g = {} if self.g_mean is None else dict(g_mean=self.g_mean, g_white=self.g_white.astype(np.float32), g_thr=self.g_thr)
        np.savez(path, mu=self.mu, sd=self.sd, W=self.W, b=self.b, thr=self.thr, heads=np.array(self.HEADS),
                 rule=np.array(self.rule), **g)

    @classmethod
    def load(cls, path):
        z = np.load(path)
        assert tuple(z["heads"]) == cls.HEADS
        g = dict(g_mean=z["g_mean"], g_white=z["g_white"].astype(np.float64), g_thr=float(z["g_thr"])) if "g_mean" in z else {}
        return cls(z["mu"], z["sd"], z["W"], z["b"], z["thr"], rule=str(z["rule"]) if "rule" in z else "overall", **g)


# ---------- where are the optic disc and the macula? ----------

class Locator:
    """A linear map from DINOv2 patch tokens to two heatmaps (optic disc, fovea), fitted by ridge regression on
    IDRiD's expert-marked centres. Positions are fractions (0..1) of the square photo; dd is one disc diameter
    in the same units, so distances can be read the way DeepDRiD's field-definition score reads them."""
    NAMES = ("disc", "macula")

    def __init__(self, W, peak_min, dd):
        self.W, self.peak_min, self.dd = np.asarray(W, np.float32), np.asarray(peak_min, float), float(dd)

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
        np.savez(path, W=self.W, peak_min=self.peak_min, dd=self.dd)

    @classmethod
    def load(cls, path):
        z = np.load(path)
        return cls(z["W"], z["peak_min"], float(z["dd"]))


SIDES = ["right", "lower right", "bottom", "lower left", "left", "upper left", "top", "upper right"]


def side_of(u, v):
    """Which edge of the picture a point is toward, in the words a technician sees on screen."""
    ang = np.degrees(np.arctan2(v - 0.5, u - 0.5)) % 360          # 0 = right, 90 = bottom (image y grows downward)
    return SIDES[int(((ang + 22.5) % 360) // 45)]


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


def aim_instruction(loc, dd):
    """A direction measured from the photo, never guessed."""
    off = {n: loc[n] for n in ("disc", "macula") if loc.get(n) and np.hypot(loc[n][0] - 0.5, loc[n][1] - 0.5) > 2 * dd}
    missing = [n for n in ("disc", "macula") if loc.get(n) is None]
    name = {"disc": "optic disc", "macula": "macula"}
    if missing:
        return f"The {name[missing[0]]} is not in the picture. Re-aim so the optic disc and the macula are both near the centre, then retake."
    if len(off) == 1:
        (n, p), = off.items()
        return f"The {name[n]} is near the {side_of(*p)} edge of the picture. Re-aim so it moves toward the centre, then retake."
    if len(off) == 2:
        mid = ((off["disc"][0] + off["macula"][0]) / 2, (off["disc"][1] + off["macula"][1]) / 2)
        return f"The disc and macula both sit toward the {side_of(*mid)} of the picture. Re-aim so they move toward the centre, then retake."
    return TEMPLATES["aim"]


# ---------- wording: Muse Spark, capped, with templates as the fallback ----------

PROMPT = """You coach a technician who takes retinal (fundus) photos for eye screening. An automatic photo-quality
check found that this photo needs a retake because it is:
{found}
{facts}Write ONE instruction telling the technician what to do: plain words, starting with a verb, at most 20 words,
fixing only the problems listed. It is not something to say to the patient. Say nothing about anything else, including framing or aim if aim is not listed. Look at the
photo only to say where a shadow, glare or haze is. Never guess where the optic disc or the macula is or which way to
aim: use only the measured positions given above. If one of them is not in the picture, say to re-aim so it comes into
view, without a direction. Do not mention disease or diagnosis.
Reply with JSON only: {{"instruction": "..."}}"""


def template(problems, aim_text=None):
    if not problems:
        return GENERIC
    parts = [aim_text if (p == "aim" and aim_text) else TEMPLATES[p] for p in problems]
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

    def __call__(self, src, problems, positions=None, aim_text=None):
        """positions: measured facts such as 'the optic disc is near the centre; the macula is near the right edge'."""
        if not problems or not self.can_call():
            return dict(text=template(problems, aim_text), source="template", cost=0.0)
        found = "\n".join(f"- {DESCRIBE[p]}" for p in problems)
        positions = positions if "aim" in problems else None     # positions only matter, and only invite comment, for aim
        facts = f"Measured positions in the picture: {positions}.\n" if positions else ""
        self.n += 1
        try:
            resp = self.client.chat.completions.create(
                model=self.model, extra_body={"reasoning_effort": self.effort},
                messages=[{"role": "user", "content": [
                    {"type": "text", "text": PROMPT.format(found=found, facts=facts)},
                    {"type": "image_url", "image_url": {"url": _data_url(src, self.px)}}]}])
        except Exception as e:  # network, quota, bad key: the technician still gets an instruction
            return dict(text=template(problems, aim_text), source=f"template (Muse Spark error: {type(e).__name__})", cost=0.0)
        cost = resp.usage.prompt_tokens * self.PRICE_IN + resp.usage.completion_tokens * self.PRICE_OUT
        self.spent += cost
        self.worst = max(self.worst, cost)
        text = _json_field(resp.choices[0].message.content or "", "instruction")
        return dict(text=text or template(problems, aim_text), source="muse-spark" if text else "template (unparsed reply)", cost=cost)
