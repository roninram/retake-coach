# Models

- `coach_model.npz`: four logistic heads (overall, focus, aim, obstruction) on DINOv2 features plus three image checks,
  their thresholds, the retake rule, and the guard (mean and whitening matrix of a Gaussian over DINOv2 features of
  ordinary fundus photos, and its energy limit). Built by `scripts/final_model.py`.
- `locator.npz`: a ridge map from DINOv2 patch tokens to optic-disc and fovea heatmaps, built by
  `notebooks/4_aim_locator_build.ipynb`. Copy it here from `quality/locator.npz` in your Drive. Without it, aim
  instructions use standard wording.

Neither file contains images. They are fitted on DeepDRiD labels (CC BY-SA 4.0) and IDRiD marks (CC BY 4.0), so we
release them under CC BY-SA 4.0 with credit to both datasets. DINOv2 itself is not included; it downloads from
Hugging Face (`facebook/dinov2-base`, Apache 2.0) on first run.
