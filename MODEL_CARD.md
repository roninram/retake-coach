# Model card: Retake Coach

Photo-quality feedback for fundus photos at the camera: **keep** or **retake**, and what to fix (focus, aim, or something
in the way). It checks photo quality only. It does not look for disease, grade diabetic retinopathy, or give medical
advice. Numbers below come from the reports in [`results/`](results/); the [README](README.md) has the full story.

## Model details

- **Built by** Rami Alroobi, October 2026, during the Meta Global AI Developer Hackathon.
- **Parts.** Everything that looks at the photo runs on the computer next to the camera:
  - **Quality checks:** frozen **DINOv2** (`facebook/dinov2-base`) features plus three simple image measures
    (sharpness, brightness and contrast), read by four logistic heads. One head gives the overall "not good enough to grade" score; the others score focus, aim and
    obstruction.
  - **Guard:** a Gaussian energy (Mahalanobis distance) on DINOv2 features, fitted on ordinary fundus photos without
    quality labels. It turns away inputs that are not regular fundus photos. A new camera is added the same way, from
    its own unlabelled photos.
  - **Locator:** a ridge map from DINOv2 patch tokens to optic-disc and macula heatmaps. It gives aim directions in
    disc diameters, the view (macula or disc) and the eye.
  - **Obstruction map:** the obstruction head's weights applied to frozen **DINOv3**
    (`facebook/dinov3-vitb16-pretrain-lvd1689m`) patch tokens, compared with a typical clean photo. It marks where a
    shadow or glare is.
  - **Wording:** each instruction is a standard sentence filled in with what was measured.
  - **Coach's note** (optional, in the cloud): **Muse Spark 1.3** through the Meta Model API summarises a visit from
    counts in words only, under a daily dollar and request cap.
- **Files.** `models/coach_model.npz`, `models/locator.npz` and `models/obstruction_map.npz`. They hold small fitted
  weights, no images.
- **Licences.** Code: MIT. Fitted models: CC BY-SA 4.0, crediting DeepDRiD and IDRiD. DINOv2: Apache 2.0. DINOv3: the
  DINOv3 License (gated on Hugging Face).

## Intended use

- **Users:** technicians and photographers taking fundus photos for eye screening, and the people who train them.
- **Use:** a second look at each photo before the patient leaves the chair, plus a per-visit report of counts for
  training. The technician decides; the tool advises.
- **Setting:** a regular (not widefield) fundus camera, with Retake Coach running on the computer next to it.

## Out of scope

- Any decision about the patient: diagnosis, grading, or referral.
- Being the only quality gate before grading. About a quarter of not-good-enough photos are kept (see below).
- Widefield, smartphone, anterior-segment or red-free photos. The guard turns most unfamiliar inputs away, but it has
  not been tested on every kind.
- A new camera, until it has been onboarded and checked on its own photos.

## Data

- **DeepDRiD** regular fundus photos (CC BY-SA 4.0). The heads were trained on the training split. Every choice was
  made on the validation split: the regularisation per head, the problem lines, and the retake line. The retake line
  was set to send back at most 25% of good validation photos, a limit declared before the test set was scored. All
  results use the 400 test photos, whose patients are separate from training and validation. The labels are an
  overall quality grade plus clarity, field definition and artifact sub-scores.
- **IDRiD** (CC BY 4.0): expert disc and fovea marks for the locator (413 training photos, 103 test). Its photos also
  serve as a second camera and as guard calibration.
- **Other:** DeepDRiD widefield photos and scikit-image samples, as inputs the guard must turn away.

## Results (DeepDRiD test, 400 photos)

| Check | Photos with the problem | AUROC (95% CI) | Sensitivity | Specificity |
|---|---|---|---|---|
| Not good enough overall | 220 | 0.82 (0.78–0.86) | 0.74 | 0.77 |
| Focus (clarity ≤ 6) | 145 | 0.90 (0.87–0.93) | 0.87 | 0.70 |
| Aim (field definition ≤ 6) | 77 | 0.85 (0.80–0.90) | 0.71 | 0.89 |
| Obstruction (artifact ≥ 6) | 123 | 0.95 (0.92–0.96) | 0.81 | 0.87 |

- **Retake decision.** It catches 162 of 220 not-good-enough photos (74%) and sends back 42 of 180 good ones (23%).
  Stricter lines cost a lot: catching 90% sends back 50% of good photos, and catching 95% sends back 64%. Each line
  is chosen on validation photos and scored on test ([results/operating_points.txt](results/operating_points.txt)).
- **Naming the problem.** On caught photos with a labelled problem (135), the first problem named is a real one 89% of
  the time. By the photo's only problem:

  | Only problem | First named right | Photos |
  |---|---|---|
  | Focus | 100% | 26 |
  | Aim | 89% | 18 |
  | Obstruction | 30% | 10 |

- **Disc and macula positions** (IDRiD test, 103 photos): median error 0.08 disc diameters for both, all within 1 disc
  diameter.
- **Which eye:** named correctly 94% of the time on DeepDRiD test photos where both structures are found (89–90% of
  photos).
- **Guard.** It accepted 100% of fundus photos from DeepDRiD test, held-out IDRiD and the web. It turned away 100% of
  widefield photos and everyday pictures. Energies: fundus 0.3–6.0, widefield 17–54, everyday 75–210 (limit 6.9).
- **Another hospital:** heads trained on one DeepDRiD hospital score AUROC 0.75–0.93 on the other two.
- **Another camera:** it asks for a retake on 9% of IDRiD photos that were picked for grading. IDRiD has no quality
  labels.
- **Obstruction map.**
  - Its patch part alone scores AUROC 0.94 for artifact ≥ 6 on validation photos.
  - It lands on a pasted shadow 85% of the time and a pasted glare 80% (chance: 7%).
  - It is not yet scored against marked real obstructions.
- **Speed:** about 93 ms per photo on a Colab T4 GPU. Not yet measured on a clinic computer's CPU.

## Limits and risks

- **Missed bad photos.** At the current line, 26% of not-good-enough photos are kept. A stricter line catches more,
  at the cost of more retakes of good photos.
- **Noisy labels.** In 34% of photos graded good enough, the sub-scores still show a problem. 36% of not-good-enough
  photos have none.
- **One training dataset** (DeepDRiD, three hospitals). There is no quality-labelled test from another camera yet. On
  synthetic damage, it often named focus for shadows and crops.
- **Obstruction is the weakest for naming.** The blue area can mark a naturally darker region. It is shown only on
  retakes, and on a retake for another reason the area around the macula is left out.
- **No patient identity.** A visit is every photo checked since "New patient". Forgetting to press it merges two
  patients in the checklist, note and report.
- **The test set was scored twice:** with the first heads and with the final ones. Every choice was made on
  validation photos.

## Privacy

- Photos never leave the computer running Retake Coach.
- Muse Spark gets only counts in words from a fixed vocabulary. Names, file names and free text are filtered out
  before the request, and a test checks this ([`tests/test_text.py`](tests/test_text.py)).
- While free API credits are in use, Meta may use requests for product improvement. That is one more reason to send
  counts, never photos.
- The visit report keeps counts and the date only: no photos, file names, patient details or clock times.
- On the public demo, the report is off and a banner asks people not to upload patient photos.

## Recommendations before clinical use

1. Check the tool on the clinic's own camera and photos, with a grader's labels.
2. Choose the retake line for the clinic's balance between missed photos and extra retakes.
3. Run a short pilot. Measure ungradable photos per visit and time per visit, with and without the tool. The visit
   report records the counts.
