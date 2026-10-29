# Models

- `coach_model.npz`: four logistic heads (overall, focus, aim, obstruction) on DINOv2 features plus three image checks,
  their thresholds, the retake rule, and the guard (mean and whitening matrix of a Gaussian over DINOv2 features of
  ordinary fundus photos, and its energy limit). Built by `scripts/final_model.py`.
- `locator.npz`: a ridge map from DINOv2 patch tokens to optic-disc and fovea heatmaps, built by
  `notebooks/4_aim_locator_build.ipynb`. Copy it here from `quality/locator.npz` in your Drive. Without it, aim
  instructions use standard wording.

- `obstruction_map.npz`: where an obstruction is. It holds the obstruction head's weights on DINOv3's mean patch token
  (from `notebooks/6_dinov3_ab.ipynb`), the typical per-patch map of a clean DeepDRiD validation photo, the marking
  level, and the smallest area worth reporting. Built by `notebooks/10_obstruction_map_build.ipynb`; copy it here from
  `quality/obstruction_map.npz`. Without it, obstruction retakes use standard wording.

None of these files contains images. They are fitted on DeepDRiD labels (CC BY-SA 4.0) and IDRiD marks (CC BY 4.0), so we
release them under CC BY-SA 4.0 with credit to both datasets. DINOv2 and DINOv3 are not included; they download from
Hugging Face on first run (`facebook/dinov2-base`, Apache 2.0; `facebook/dinov3-vitb16-pretrain-lvd1689m`, DINOv3
License, gated). The obstruction map is computed from DINOv3 features, so using it also falls under the DINOv3 License.
