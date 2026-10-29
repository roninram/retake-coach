"""Retake Coach demo server.

    pip install -r requirements.txt
    export MODEL_API_KEY=...            # optional: Muse Spark writes the coach's note; without it, a fixed summary
    python app.py                       # uses models/coach_model.npz and models/locator.npz
    python app.py --watch ~/EyerInbox   # also checks every new photo that lands in that folder
    HF_TOKEN=... python app.py          # DINOv3 (gated on Hugging Face) for the obstruction map in models/obstruction_map.npz

Pages: / checks one photo at a time during a visit; /batch checks many photos at once, or follows a watched folder
(for example the folder a camera exports to over Wi-Fi).

The photo never leaves this computer: DINOv2, the heads, the disc and macula locator and the obstruction map (DINOv3)
run here, and each instruction is a standard sentence built from what they measured. At the end of a session, the
coach's note is written by Muse Spark (Meta Model API) from the session's counts in words, within the budget below.
No photo is sent.

Budget for the coach's note (environment variables, dollars at Muse Spark 1.3 Standard prices):
    MUSE_DAILY_USD      default 0.25    MUSE_DAILY_REQUESTS default 60    MUSE_TOTAL_USD default 1.00
Past any cap the note is a fixed summary of the same counts.
"""
import argparse
import base64
import datetime
import io
import os
import threading
import time
from collections import OrderedDict
from pathlib import Path

import numpy as np
from fastapi import Body, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from PIL import Image, UnidentifiedImageError

import coach_core as cc

HERE = Path(__file__).parent
MAX_BYTES = 25 * 1024 * 1024
IMAGE_TYPES = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".webp"}


class Budget:
    """Muse Spark spending: a fresh daily allowance, never past the total cap for this server's lifetime."""

    def __init__(self, client, daily_usd, daily_requests, total_usd):
        self.client, self.daily_usd, self.daily_requests, self.total_usd = client, daily_usd, daily_requests, total_usd
        self.spent_before_today, self.day, self.words = 0.0, None, None
        self.lock = threading.Lock()

    def today(self):
        with self.lock:
            d = datetime.date.today()
            if d != self.day:
                if self.words is not None:
                    self.spent_before_today += self.words.spent
                left = max(0.0, self.total_usd - self.spent_before_today)
                self.day = d
                self.words = cc.Wording(self.client, max_requests=self.daily_requests, max_usd=min(self.daily_usd, left))
            return self.words

    def status(self):
        w = self.today()
        return dict(muse=self.client is not None, can_call=w.can_call(), spent_today=round(w.spent, 4),
                    cap_today=round(w.max_usd, 4), requests_today=w.n, request_cap=w.max_requests,
                    spent_total=round(self.spent_before_today + w.spent, 4), cap_total=self.total_usd)


def preview(x):
    """The 448 px square the detector saw, as a small JPEG data URL for the page."""
    buf = io.BytesIO()
    Image.fromarray((x * 255).astype(np.uint8)).save(buf, "JPEG", quality=85)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


def read_image(data):
    """Bytes -> RGB PIL image, or None if it is not a picture we can read."""
    try:
        img = Image.open(io.BytesIO(data))
        img.load()
        return img.convert("RGB")
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError, ValueError):
        return None


class Checker:
    """One photo in, one result out. Shared by the single-photo page, the batch page and the watched folder."""

    def __init__(self, coach, encoder, budget, omap=None, map_encoder=None):
        self.coach, self.encoder, self.budget = coach, encoder, budget
        self.omap, self.map_encoder = omap, map_encoder               # where an obstruction is (optional)
        self.lock, self.wlock = threading.Lock(), threading.Lock()   # one photo at a time through the encoders; exact note budget

    def where_obstruction(self, x, grid, exclude=None):
        """Run the obstruction map, reusing the first pass's patch tokens when the map reads the same encoder."""
        if self.omap.backbone != self.coach.backbone or grid is None:
            with self.lock:
                _, grid = self.map_encoder([x], patches=True)
        return self.omap.locate(x, grid[0], exclude)

    def __call__(self, img):
        coach = self.coach
        t0 = time.perf_counter()
        x = cc.load_square(img)
        with self.lock:
            X, grid = self.encoder([x], patches=True) if coach.locator is not None else (self.encoder([x]), None)
        v = coach.verdict(X, np.array([cc.simple_checks(x)]), patches=grid)[0]
        t1 = time.perf_counter()
        ob, hint = None, False
        if self.omap is not None and v["looks_like_fundus"] and v["retake"]:
            named = "obstruction" in v["problems"]
            # on a retake for another reason, leave out the macula, which is naturally darker (1.25 disc diameters)
            around_macula = (*v["macula"], 1.25 * v["dd"]) if (not named and v.get("macula") and v.get("dd")) else None
            try:
                ob = self.where_obstruction(x, grid, around_macula)
            except Exception as e:                          # the extra map must never stop the check itself
                print("obstruction map:", type(e).__name__, e)
            if ob is not None and not named:
                # retake for another reason: show the area only if it is larger than clear photos ever mark (95%)
                hint = bool(ob["where"]) and ob["share"] > self.omap.hint_share
                ob = ob if hint else None
        tm = time.perf_counter()

        if not v["looks_like_fundus"]:
            verdict, words = "not_fundus", dict(text=cc.NOT_FUNDUS, source="template", cost=0.0)
        elif v["retake"]:                                   # a standard sentence built from what was measured
            named = v["problems"] + (["obstruction"] if hint else [])
            verdict, words = "retake", dict(text=cc.template(named, v.get("aim_text"), ob), source="template", cost=0.0)
        else:
            verdict, words = "keep", dict(text=cc.keep_text(v["minor"], v.get("aim_text")), source="template", cost=0.0)
        t2 = time.perf_counter()

        checks = {k: dict(score=round(v["scores"][k], 3), threshold=round(float(t), 3), found=k in v["problems"],
                          minor=k in v["minor"], marked=(k == "obstruction" and hint))
                  for k, t in zip(cc.PROBLEMS, coach.thr[1:])}
        return dict(
            verdict=verdict, instruction=words["text"], source=words["source"], cost=round(words["cost"], 5),
            problems=v["problems"], checks=checks, p_not_good=round(v["p_not_good"], 3),
            retake_threshold=round(float(coach.thr[0]), 3), rule=coach.rule, energy=round(v["energy"], 2),
            energy_limit=round(float(coach.g_thr), 2) if coach.g_thr is not None else None,
            disc=v.get("disc"), macula=v.get("macula"), dd=v.get("dd"), positions=v.get("positions"),
            eye=v.get("eye"), field=v.get("field"),
            obstruction=None if ob is None else dict(where=ob["where"], tone=ob["tone"], share=round(ob["share"], 3), hint=hint,
                                                     summary=cc.obstruction_summary(ob), overlay=ob["overlay"]),
            ms=dict(detect=round(1000 * (t1 - t0)), map=round(1000 * (tm - t1)), wording=round(1000 * (t2 - tm))),
            preview=preview(x), budget=self.budget.status())


class Watcher:
    """Checks every new photo that appears in a folder, e.g. where a camera exports over Wi-Fi (FTP or a shared folder).
    A file is checked once its size has stopped changing, so half-copied photos are not read."""

    def __init__(self, folder, checker, every=2.0):
        self.folder, self.checker, self.every = Path(folder).expanduser(), checker, every
        self.results, self.seen, self.sizes, self.seq = [], set(), {}, 0
        self.lock, self.stopped = threading.Lock(), False

    def scan_once(self):
        for p in sorted(self.folder.rglob("*")):
            if p.suffix.lower() not in IMAGE_TYPES or not p.is_file():
                continue
            key = (str(p), p.stat().st_mtime)
            if key in self.seen:
                continue
            size = p.stat().st_size
            if self.sizes.get(str(p)) != size:              # still being written, or first sight: look again next round
                self.sizes[str(p)] = size
                continue
            self.seen.add(key)
            img = read_image(p.read_bytes())
            r = dict(verdict="unreadable", instruction="Not a photo the check can read.") if img is None else self.checker(img)
            r.update(name=str(p.relative_to(self.folder)), arrived=datetime.datetime.now().strftime("%H:%M:%S"))
            with self.lock:
                r["seq"] = self.seq                         # a running number, so pages can ask for what is new
                self.seq += 1
                self.results.append(r)

    def run(self):
        while not self.stopped:
            try:
                if self.folder.is_dir():
                    self.scan_once()
            except Exception as e:                          # keep watching whatever happens to one file
                print("watch:", type(e).__name__, e)
            time.sleep(self.every)

    def start(self):
        threading.Thread(target=self.run, daemon=True).start()
        return self

    def stop(self):
        self.stopped = True

    def since(self, n):
        with self.lock:
            return [r for r in self.results if r["seq"] >= n]

    def clear(self):
        """Forget the results shown so far; files already checked are not checked again."""
        with self.lock:
            self.results = []


class Store:
    """What the two pages have checked so far, kept here so that switching pages or reloading loses nothing:
    the current visit on the one-photo page, and the list on the many-photos page. Kept per browser tab (the pages send
    a random session id), so people sharing a public demo never see each other's photos. In memory only; a restart
    clears it, and the least recently used sessions are dropped past `sessions`."""

    def __init__(self, limit=1000, sessions=200):
        self.data, self.lock, self.limit, self.sessions = OrderedDict(), threading.Lock(), limit, sessions

    def _lists(self, sid):
        if sid not in self.data:
            self.data[sid] = {"visit": [], "batch": []}
            while len(self.data) > self.sessions:
                self.data.popitem(last=False)
        self.data.move_to_end(sid)
        return self.data[sid]

    def add(self, sid, which, r):
        with self.lock:
            lst = self._lists(sid)[which]
            lst.append(r)
            del lst[:-self.limit]

    def clear(self, sid, which):
        with self.lock:
            self._lists(sid)[which] = []

    def get(self, sid):
        with self.lock:
            return {k: list(v) for k, v in self._lists(sid).items()}


def session_id(sid):
    """A page's session id, or 'local' when it sends none."""
    sid = (sid or "local").strip()
    return sid[:64] if sid.replace("-", "").isalnum() else "local"


def create_app(coach, encoder, budget, examples_dir=None, watcher=None, checker=None, store=None):
    app = FastAPI(title="Retake Coach")
    checker = checker or (watcher.checker if watcher else Checker(coach, encoder, budget))
    store = store or Store()
    examples = Path(examples_dir) if examples_dir else None

    @app.get("/")
    def index():
        return FileResponse(HERE / "static" / "index.html")

    @app.get("/batch")
    def batch():
        return FileResponse(HERE / "static" / "batch.html")

    @app.get("/api/status")
    def status():
        return dict(budget=budget.status(), thresholds=dict(zip(cc.Coach.HEADS, map(float, coach.thr))),
                    energy_limit=coach.g_thr, watching=str(watcher.folder) if watcher else None,
                    demo=bool(os.environ.get("RC_DEMO")),
                    detector=coach.backbone, obstruction_map=checker.omap.backbone if checker.omap else None)

    @app.get("/api/examples")
    def list_examples():
        """Sample photos, with an optional note per file from examples/labels.json (e.g. the dataset's own label)."""
        if not examples or not examples.is_dir():
            return []
        notes = {}
        if (examples / "labels.json").is_file():
            import json
            notes = json.loads((examples / "labels.json").read_text())
        return [dict(name=p.name, note=notes.get(p.name, "")) for p in sorted(examples.iterdir()) if p.suffix.lower() in IMAGE_TYPES]

    @app.get("/examples/{name}")
    def example(name: str):
        p = (examples / name).resolve() if examples else None
        if not p or p.parent != examples.resolve() or not p.is_file():
            raise HTTPException(404, "No such sample photo.")
        return FileResponse(p)

    @app.post("/api/check")
    def check(photo: UploadFile = File(...), keep_in: str = Form(""), name: str = Form(""),
              sample_name: str = Form(""), sample_note: str = Form(""), sid: str = Form("")):
        """keep_in: 'visit' or 'batch' keeps the result for that page (see /api/state)."""
        data = photo.file.read(MAX_BYTES + 1)
        if len(data) > MAX_BYTES:
            raise HTTPException(413, "That file is over 25 MB. Export the photo as a JPEG and try again.")
        img = read_image(data)
        if img is None:
            raise HTTPException(400, "That file is not a photo the check can read. Use a JPEG or PNG.")
        r = checker(img)
        r.update(name=name or photo.filename or "photo", arrived=datetime.datetime.now().strftime("%H:%M:%S"),
                 sample=dict(name=sample_name, text=sample_note) if sample_name else None)
        if keep_in in ("visit", "batch"):
            store.add(session_id(sid), keep_in, r)
        return JSONResponse(r)

    @app.get("/api/state")
    def state(sid: str = ""):
        """Everything this browser tab has checked so far on each page, oldest first."""
        return store.get(session_id(sid))

    @app.post("/api/clear")
    def clear(which: str, sid: str = ""):
        """'visit' (New patient) or 'batch' (Clear list; also forgets photos shown from the watched folder)."""
        if which not in ("visit", "batch"):
            raise HTTPException(400, "Clear 'visit' or 'batch'.")
        store.clear(session_id(sid), which)
        if which == "batch" and watcher:
            watcher.clear()
        return dict(cleared=which)

    @app.post("/api/note")
    def coach_note(payload: dict = Body(...)):
        """The coach's note for a session. The page sends each result's verdict, problems, obstruction place and tone,
        eye and field; only those are read, and only their counts in words go to Muse Spark."""
        results = payload.get("results")
        if not isinstance(results, list) or len(results) > 5000:
            raise HTTPException(400, "Send the session's results as a list.")
        facts = cc.session_facts([r for r in results if isinstance(r, dict)])
        fallback = cc.note_template(facts)
        if payload.get("use_muse", True) and facts["n"]:
            with checker.wlock:
                out = budget.today().note(facts["text"], fallback)
        else:
            out = dict(text=fallback, source="template", cost=0.0)
        return dict(note=out["text"], source=out["source"], cost=round(out["cost"], 5), facts=facts["text"], budget=budget.status())

    @app.get("/api/watch")
    def watch(since: int = 0):
        if not watcher:
            return dict(folder=None, results=[])
        return dict(folder=str(watcher.folder), results=watcher.since(since))

    return app


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="models/coach_model.npz")
    ap.add_argument("--locator", default="models/locator.npz", help="optional: disc and macula positions for aim directions")
    ap.add_argument("--map", default="models/obstruction_map.npz", help="optional: where an obstruction is (DINOv3; needs HF_TOKEN)")
    ap.add_argument("--examples", default="examples")
    ap.add_argument("--watch", default=None, help="folder to check new photos in as they arrive")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    a = ap.parse_args()

    client = None
    if os.environ.get("MODEL_API_KEY"):
        from openai import OpenAI
        client = OpenAI(base_url="https://api.meta.ai/v1", api_key=os.environ["MODEL_API_KEY"], timeout=30)
    budget = Budget(client, float(os.environ.get("MUSE_DAILY_USD", 0.25)), int(os.environ.get("MUSE_DAILY_REQUESTS", 60)),
                    float(os.environ.get("MUSE_TOTAL_USD", 1.00)))
    coach = cc.Coach.load(a.model)
    if os.path.exists(a.locator):
        coach.attach_locator(cc.Locator.load(a.locator))
    print(f"loading {coach.backbone} ...")
    encoder = cc.Encoder(coach.backbone)
    omap = map_encoder = None
    if os.path.exists(a.map):
        omap = cc.ObstructionMap.load(a.map)
        try:
            map_encoder = encoder if omap.backbone == coach.backbone else cc.Encoder(omap.backbone)
        except Exception as e:                               # DINOv3 is gated: without access the page works without the map
            print(f"obstruction map off: could not load {omap.backbone} ({type(e).__name__}). Set HF_TOKEN after accepting its licence.")
            omap = None
    checker = Checker(coach, encoder, budget, omap, map_encoder)
    watcher = Watcher(a.watch, checker).start() if a.watch else None
    app = create_app(coach, encoder, budget, a.examples, watcher, checker)
    import uvicorn
    uvicorn.run(app, host=a.host, port=a.port)


if __name__ == "__main__":
    main()
