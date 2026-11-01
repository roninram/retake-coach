"""Test setup: a small stand-in for DINOv2, so the tests run on any CPU with no model download and no API key.

The stand-in turns each 14 x 14 patch into its mean colour and mean squared colour. It is enough to exercise every code
path (checks, guard, positions, pages, note, report); it says nothing about accuracy, which the results/ reports cover.
"""
import io
import os
import sys
import types

import numpy as np
import pytest
import torch
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)


class StandInEncoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.p = torch.nn.Parameter(torch.zeros(1))
        self.config = types.SimpleNamespace(patch_size=14)

    def forward(self, pixel_values):
        x = torch.nn.functional.avg_pool2d(pixel_values, 14)
        t = torch.cat([x, x ** 2], 1).flatten(2).transpose(1, 2)
        return types.SimpleNamespace(last_hidden_state=torch.cat([t.mean(1, keepdim=True), t], 1))


sys.modules["transformers"] = types.SimpleNamespace(AutoModel=types.SimpleNamespace(from_pretrained=lambda name, **k: StandInEncoder()))

import app as appmod  # noqa: E402
import coach_core as cc  # noqa: E402


def fundus_jpeg(seed=0, size=600):
    """A synthetic fundus-like picture (orange disc on black, a pale spot, a darker spot) as JPEG bytes."""
    r = np.random.default_rng(seed)
    yy, xx = np.mgrid[:size, :size] / size
    inside = (xx - 0.5) ** 2 + (yy - 0.5) ** 2 < 0.22
    img = np.zeros((size, size, 3))
    img[inside] = [0.75, 0.35, 0.15]
    img[(xx - 0.65) ** 2 + (yy - 0.5) ** 2 < 0.004] = [0.95, 0.85, 0.6]
    img[(xx - 0.4) ** 2 + (yy - 0.5) ** 2 < 0.003] = [0.45, 0.2, 0.08]
    img = np.clip(img + r.normal(0, 0.02, img.shape), 0, 1)
    buf = io.BytesIO()
    Image.fromarray((img * 255).astype(np.uint8)).save(buf, "JPEG")
    return buf.getvalue()


def everyday_png(size=300):
    """A picture that is not a fundus photo: a grey gradient with a square, as PNG bytes."""
    img = np.tile(np.linspace(40, 220, size, dtype=np.uint8)[None, :, None], (size, 1, 3))
    img[100:200, 100:200] = [30, 120, 200]
    buf = io.BytesIO()
    Image.fromarray(img).save(buf, "PNG")
    return buf.getvalue()


class FakeMuse:
    """Stands in for the Meta Model API client. Records every prompt and replies with a fixed note."""

    def __init__(self, reply='{"note": "Aim was the most common problem."}'):
        self.reply, self.prompts = reply, []
        self.chat = types.SimpleNamespace(completions=types.SimpleNamespace(create=self.create))

    def create(self, **kw):
        content = kw["messages"][0]["content"]
        assert all(c["type"] == "text" for c in content), "a photo was sent to Muse Spark"
        self.prompts.append(content[0]["text"])
        return types.SimpleNamespace(choices=[types.SimpleNamespace(message=types.SimpleNamespace(content=self.reply))],
                                     usage=types.SimpleNamespace(prompt_tokens=300, completion_tokens=60))


@pytest.fixture
def coach():
    """A coach with random heads on the stand-in's 12 numbers per photo. Its guard accepts everything."""
    r = np.random.default_rng(0)
    X, C = r.normal(size=(60, 12)), r.normal(size=(60, 3))
    c = cc.Coach.fit(X, C, {h: r.random(60) < 0.5 for h in cc.Coach.HEADS}).set_guard(X, X)
    c.g_thr = 1e9
    return c


@pytest.fixture
def make_client(coach, tmp_path):
    """make_client(muse=None, report=True) -> (TestClient, the VisitLog or None)."""
    from fastapi.testclient import TestClient

    def make(muse=None, report=True):
        budget = appmod.Budget(muse, 0.25, 60, 1.0)
        log = appmod.VisitLog(tmp_path / "reports" / "visits.csv") if report else None
        encoder = cc.Encoder(device="cpu")
        app = appmod.create_app(coach, encoder, budget, None, checker=appmod.Checker(coach, encoder, budget), report=log)
        return TestClient(app), log
    return make
