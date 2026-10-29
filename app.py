"""Retake Coach demo server: one page, one endpoint.

    pip install -r requirements.txt
    export MODEL_API_KEY=...            # optional: without it, instructions come from fixed templates
    python app.py                       # uses models/coach_model.npz and models/locator.npz

The photo never leaves this computer for detection: DINOv2 and the heads run here. Only when a retake is needed,
and only within the Muse Spark budget below, the photo is sent to the Meta Model API to word the instruction.

Budget (environment variables, dollars at Muse Spark 1.3 Standard prices):
    MUSE_DAILY_USD      default 0.25    MUSE_DAILY_REQUESTS default 60    MUSE_TOTAL_USD default 1.00
Past any cap the page keeps working with template instructions.
"""
import argparse
import base64
import datetime
import io
import os
import threading
import time
from pathlib import Path

import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
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


def create_app(coach, encoder, budget, examples_dir=None):
    app = FastAPI(title="Retake Coach")
    lock, wlock = threading.Lock(), threading.Lock()   # one photo at a time through the encoder; exact budget counts
    examples = Path(examples_dir) if examples_dir else None

    @app.get("/")
    def index():
        return FileResponse(HERE / "static" / "index.html")

    @app.get("/api/status")
    def status():
        return dict(budget=budget.status(), thresholds=dict(zip(cc.Coach.HEADS, map(float, coach.thr))),
                    energy_limit=coach.g_thr)

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
    def check(photo: UploadFile = File(...), use_muse: bool = Form(True)):
        data = photo.file.read(MAX_BYTES + 1)
        if len(data) > MAX_BYTES:
            raise HTTPException(413, "That file is over 25 MB. Export the photo as a JPEG and try again.")
        try:
            img = Image.open(io.BytesIO(data))
            img.load()
            img = img.convert("RGB")
        except (UnidentifiedImageError, OSError, Image.DecompressionBombError):
            raise HTTPException(400, "That file is not a photo the check can read. Use a JPEG or PNG.")

        t0 = time.perf_counter()
        x = cc.load_square(img)
        with lock:
            X, grid = encoder([x], patches=True) if coach.locator is not None else (encoder([x]), None)
        v = coach.verdict(X, np.array([cc.simple_checks(x)]), patches=grid)[0]
        t1 = time.perf_counter()

        if not v["looks_like_fundus"]:
            verdict, words = "not_fundus", dict(text=cc.NOT_FUNDUS, source="template", cost=0.0)
        elif v["retake"]:
            verdict = "retake"
            if use_muse:
                with wlock:
                    words = budget.today()(img, v["problems"], positions=v.get("positions"), aim_text=v.get("aim_text"))
            else:
                words = dict(text=cc.template(v["problems"], v.get("aim_text")), source="template", cost=0.0)
        else:
            verdict, words = "keep", dict(text=cc.keep_text(v["minor"], v.get("aim_text")), source="template", cost=0.0)
        t2 = time.perf_counter()

        checks = {k: dict(score=round(v["scores"][k], 3), threshold=round(float(t), 3), found=k in v["problems"],
                          minor=k in v["minor"])
                  for k, t in zip(cc.PROBLEMS, coach.thr[1:])}
        return JSONResponse(dict(
            verdict=verdict, instruction=words["text"], source=words["source"], cost=round(words["cost"], 5),
            problems=v["problems"], checks=checks, p_not_good=round(v["p_not_good"], 3),
            retake_threshold=round(float(coach.thr[0]), 3), rule=coach.rule, energy=round(v["energy"], 2),
            energy_limit=round(float(coach.g_thr), 2) if coach.g_thr is not None else None,
            disc=v.get("disc"), macula=v.get("macula"), dd=v.get("dd"), positions=v.get("positions"),
            ms=dict(detect=round(1000 * (t1 - t0)), wording=round(1000 * (t2 - t1))),
            preview=preview(x), budget=budget.status()))

    return app


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="models/coach_model.npz")
    ap.add_argument("--locator", default="models/locator.npz", help="optional: disc and macula positions for aim directions")
    ap.add_argument("--examples", default="examples")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    a = ap.parse_args()

    client = None
    if os.environ.get("MODEL_API_KEY"):
        from openai import OpenAI
        client = OpenAI(base_url="https://api.meta.ai/v1", api_key=os.environ["MODEL_API_KEY"], timeout=30)
    budget = Budget(client, float(os.environ.get("MUSE_DAILY_USD", 0.25)), int(os.environ.get("MUSE_DAILY_REQUESTS", 60)),
                    float(os.environ.get("MUSE_TOTAL_USD", 1.00)))
    print("loading DINOv2 ...")
    coach = cc.Coach.load(a.model)
    if os.path.exists(a.locator):
        coach.locator = cc.Locator.load(a.locator)
    app = create_app(coach, cc.Encoder(), budget, a.examples)
    import uvicorn
    uvicorn.run(app, host=a.host, port=a.port)


if __name__ == "__main__":
    main()
