# What we tried first

We started the event on energy-based models for diabetic-retinopathy grading, building on our IntelliSys 2026 paper.
Grading a retina is a diagnosis, and the results below did not support shipping a grader. Checking photo quality
does help a technician today and makes no medical claim, so we moved there. Every notebook here was written and run
during the event. Results we have files for are in `results/`.

## 1. Is an energy-margin grader better calibrated on a new camera?

`retrain_and_export.ipynb`, `export_logits.py`, `energy_eval.py`

We trained five seeds each of a softmax (cross-entropy) and an energy-margin ResNet-18 grader on IDRiD. We tested them
on IDRiD's test split and on all 3,662 labelled APTOS-2019 photos, which come from another camera. The training code
is prior work from the paper and is not in this repository; these files only call it.

- On APTOS, the energy-margin models were better calibrated in 5 of 5 seeds: expected calibration error 0.043 vs 0.104
  (p = 0.032). Their confidence matched their accuracy, where softmax models were 10–13 points overconfident.
- The paper's accuracy gain did not hold on the full APTOS set: +3.5 points, p = 0.28.

Results: `results/energy_eval.csv`.

## 2. Can a grader adapt to a new camera from unlabeled photos?

`tta_new_camera.ipynb`

We compared AdaBN, Tent and an energy-based test-time adaptation in the style of TEA (CVPR 2024). Each adapted on 500
unlabeled APTOS photos and was scored on the other 3,162.

- The energy-based adaptation raised the energy-margin model's quadratic-weighted kappa by 12.6 points
  (5 of 5 seeds, p = 0.002).
- Tent reached higher accuracy. The energy-based adaptation stayed better calibrated (5 of 5 seeds).

Results: `results/tta_results.csv`.

## 3. Can SAM 3.1 find lesions from a text prompt?

`sam_lesions.py`, `sam_pilot.ipynb`, `exudates_vs_grade.ipynb`, `exudate_evidence_test.ipynb`, `compose_eval.py`

We scored SAM 3.1 against IDRiD's pixel masks.

- Medical prompts ("hard exudate", "microaneurysm") on whole photos found nothing (`results/sam_first_pilot_summary.csv`).
- Plain prompts on 1024-pixel tiles worked for hard exudates: "yellow spot" found them at about 97% pixel precision.
- Microaneurysms, haemorrhages and soft exudates were not found.
- Adding exudate evidence to the grader's energy did not improve grading on IDRiD's 57 test photos
  (`results/exudate_features_test.csv`, `results/compose_eval.csv`).

Every SAM run had a hard request cap.

## 4. Is Muse Glimmer's energy a useful confidence signal?

`glimmer_energy_pilot.ipynb`

On an MMLU pilot with Muse Glimmer 30B, the free energy of the answer logits separated right from wrong answers only
weakly (`results/glimmer_mmlu_pilot.csv`).

## 5. Can energies trained on synthetic damage judge real photo quality?

`../notebooks/1_quality_energy_test.ipynb`

We trained contrastive energies on IDRiD photos with synthetic blur, haze, darkening, shadows, glare and off-centre
crops, and a Gaussian energy on good photos only. On DeepDRiD's real quality labels, neither beat a simple sharpness
check. As a coach, the contrastive energies named the right problem 13% (focus), 21% (aim) and 95% (obstruction) of
the time. Supervised heads trained on DeepDRiD's labels did much better, and those became Retake Coach.
The Gaussian energy found its use as the guard that turns away non-fundus inputs.
