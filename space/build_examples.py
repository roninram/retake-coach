"""Sample photos for the public demo, fetched when the Space is built (nothing is copied into this repository).

    python space/build_examples.py examples

DeepDRiD photos (CC BY-SA 4.0, https://github.com/deepdrdoc/DeepDRiD): nine test-set photos chosen to show each kind of
result, plus one widefield photo the guard turns away. scikit-image samples: a fundus photo from the web and a cat.
"""
import json
import os
import shutil
import subprocess
import sys
import tempfile

NOTES = {   # each photo's own DeepDRiD label, shown under it on the page
    "354_r2": "DeepDRiD label: good",
    "354_l2": "DeepDRiD label: good",
    "354_l1": "DeepDRiD label: good (aim flagged)",
    "354_r1": "DeepDRiD label: good (aim flagged)",
    "368_r1": "DeepDRiD label: not good enough (focus, aim, obstruction flagged)",
    "384_r2": "DeepDRiD label: not good enough (aim flagged)",
    "347_r1": "DeepDRiD label: not good enough (focus, obstruction flagged)",
    "353_l1": "DeepDRiD label: not good enough (focus, obstruction flagged)",
    "368_l2": "DeepDRiD label: not good enough (focus, obstruction flagged)",
}
TEST = "regular_fundus_images/Online-Challenge1&2-Evaluation/Images"
UWF = "ultra-widefield_images/ultra-widefield-validation/Images/34/34_l2.jpg"


def main(out):
    os.makedirs(out, exist_ok=True)
    notes = {}
    with tempfile.TemporaryDirectory() as tmp:
        def git(*a):
            subprocess.run(["git", *a], cwd=tmp, check=True)
        git("clone", "-q", "--filter=blob:none", "--no-checkout", "--depth", "1", "https://github.com/deepdrdoc/DeepDRiD.git", ".")
        git("config", "core.sparseCheckout", "true")
        paths = [f"{TEST}/{i.split('_')[0]}/{i}.jpg" for i in NOTES] + [UWF]
        with open(os.path.join(tmp, ".git", "info", "sparse-checkout"), "w") as f:
            f.write("".join("/" + p + "\n" for p in paths))
        git("read-tree", "-mu", "HEAD")
        for i, note in NOTES.items():
            shutil.copy(os.path.join(tmp, TEST, i.split("_")[0], f"{i}.jpg"), os.path.join(out, f"{i}.jpg"))
            notes[f"{i}.jpg"] = note
        shutil.copy(os.path.join(tmp, UWF), os.path.join(out, "widefield_34_l2.jpg"))
        notes["widefield_34_l2.jpg"] = "DeepDRiD widefield photo (a different kind of camera)"
    try:
        import numpy as np
        import skimage.data as sd
        from PIL import Image
        for name, fn, txt in [("web_retina.jpg", "retina", "Fundus photo from the web (scikit-image sample)"),
                              ("everyday_cat.jpg", "chelsea", "Everyday picture (scikit-image sample)")]:
            try:
                Image.fromarray(np.asarray(getattr(sd, fn)())[..., :3]).save(os.path.join(out, name), quality=92)
                notes[name] = txt
            except Exception as e:  # the demo works without these two
                print("skipped", name, type(e).__name__, e)
    except ImportError:
        print("scikit-image not installed: no web fundus photo or everyday picture")
    with open(os.path.join(out, "labels.json"), "w") as f:
        json.dump(notes, f, indent=1)
    print(len(notes), "sample photos in", out)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "examples")
