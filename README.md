# Retake Coach

Catch an ungradable fundus photo while the patient is still in the chair.

In diabetic eye screening, a technician photographs the retina and sends the photos on to be graded. When a photo is
too blurry, badly aimed or partly covered, nobody can grade it, and the patient has to come back. This happens often.
In a US telehealth program, 12.4% of eyes photographed with a non-mydriatic camera were ungradable
([Silva et al., *Diabetes Care* 2014](https://doi.org/10.2337/dc13-1292)). With a handheld non-mydriatic camera in
India, 23.9% of images were ungradable
([Davila et al., *Ophthalmologica* 2017](https://pmc.ncbi.nlm.nih.gov/articles/PMC5733780)).

Retake Coach checks each photo in about a second, on the computer next to the camera. It says **keep** or **retake**.
For a retake, it says what to fix: focus, aim, or something blocking the view. For aim, it measures where the optic
disc and macula are, tells whether the photo is the macula view or the disc view, and says what is off for that view. For something in the way, it marks where on the photo it is. It
turns away pictures that are not regular fundus photos, such as a widefield image or the wrong file.

No photo leaves that computer. Each instruction is a standard sentence built from what was measured. At the end of a
visit or a batch, Muse Spark writes a short coach's note from the session's counts in words: the most common problem,
one habit that would prevent it, and any view still missing.

It checks photo quality only. It does not look for disease and gives no medical advice.

What's on the two pages:
- **One photo** (`/`): keep or retake, one instruction, the measured disc and macula on the photo, and for an
  obstruction, the area that is in the way. A visit checklist shows which of the four usual views are kept so far:
  each eye, centred on the macula and on the disc. A missing view is clear before the patient leaves, and the coach's
  note sums up the visit.
- **Many photos** (`/batch`): drop a session's photos or a whole folder and get a table with keep or retake, what to
  fix, the eye and the field, plus a CSV download. It can also watch a folder, such as the one a camera exports to over
  Wi-Fi, and check each photo as it arrives. The coach's note sums up the batch.

Switching between the pages, or reloading one, keeps what has been checked. The server holds it in memory until it
restarts; "New patient" and "Clear list" start over.

A new camera can be added from a few of its own ordinary photos, with no labels: the guard learns what that camera's
photos look like (`notebooks/8_eyer_check.ipynb`, written for the Phelcom Eyer2 handheld camera).

![Retake Coach asking for a retake of DeepDRiD photo 347_r1: it is blurry, and a dark shadow covers about a quarter of the photo at the upper left. The blue area is where the obstruction check found it, measured on the computer with DINOv3. The ring and dot are the measured optic disc and macula.](docs/screenshot.png)

## How it uses Meta models

| Job | Meta model | Runs | Cost |
|---|---|---|---|
| Detect focus, aim and obstruction problems, and overall gradability | **DINOv2** (`facebook/dinov2-base`, frozen) plus four small logistic heads | on the local computer | $0 |
| Turn away inputs that are not regular fundus photos | **DINOv2** features plus a Gaussian energy (Mahalanobis distance) fitted on ordinary fundus photos only; see [Energy](#energy-in-retake-coach) | on the local computer | $0 |
| Measure where the optic disc and macula are, so aim directions are never guessed, and tell which eye | **DINOv2** patch tokens plus a ridge map to two heatmaps | on the local computer | $0 |
| Say where a shadow or glare is | **DINOv3** (`facebook/dinov3-vitb16-pretrain-lvd1689m`, frozen) patch tokens, read through the obstruction head's own weights | on the local computer | $0 |
| Accept photos from a newly added camera | **DINOv2** features plus a second Gaussian energy fitted on that camera's own photos (no labels) | on the local computer | $0 |
| Write the coach's note at the end of a visit or batch, from the session's counts in words | **Muse Spark 1.3** through the Meta Model API, with `reasoning_effort="minimal"` | cloud, under a daily dollar and request cap | about $0.003 per note (estimate) |

Everything that looks at the photo runs on the camera's computer, and the photo stays there. Muse Spark gets only the
session's counts in words, such as "Problems on the photos sent back: obstruction 3 (dark shadow toward the top of the
photo 2; glare or haze in the centre of the photo 1), focus 1". The page shows exactly what it was given. If the note
contains a number that is not in those counts, a fixed summary of the same counts is shown instead; the same happens
past the cap or on any error.

Muse Spark neither detects problems nor writes each instruction, by design. We tested both:
- **As the judge:** in a 20-photo pilot it asked for a retake on 4 of 5 good photos and named almost every problem on
  every photo ([results/muse_coach_pilot.csv](results/muse_coach_pilot.csv)).
- **Wording one retake, with the photo:** it put the macula on the wrong side when asked for an aim direction
  ([results/muse_wording_pilot.csv](results/muse_wording_pilot.csv), photo 354_l1).
- **Wording one retake, from the measured facts in words:** its rewrites read worse than the standard sentence, and
  one was wrong. For 267_r1 it said to re-aim so "the macula not in the picture" is "near the centre"
  ([results/muse_words_only.csv](results/muse_words_only.csv)). It also cost about $0.003 a retake, no less than
  sending the photo.

So the tool splits the jobs:
- **The heads** decide whether there is a problem.
- **The locator** says where the disc and macula are.
- **The obstruction map** says where a shadow or glare is.
- **Standard sentences** turn those facts into the instruction.
- **Muse Spark** writes the coach's note across a session: summing up and suggesting one habit is a language job.

## Results on photos the model never trained on

**DeepDRiD test set** (400 photos, patients separate from training and validation). Every threshold and setting was
chosen on validation photos. The retake threshold sends back at most 25% of good *validation* photos, a limit set
before the test set was scored. ([results/final_model_report.txt](results/final_model_report.txt))

| Check | AUROC (95% CI) | Sharpness alone |
|---|---|---|
| Focus (clarity ≤ 6) | 0.90 (0.87–0.93) | 0.88 |
| Aim (field definition ≤ 6) | 0.85 (0.80–0.90) | 0.53 |
| Obstruction (artifact ≥ 6) | 0.95 (0.92–0.96) | 0.85 |
| Overall "not good enough for diagnosis" | 0.82 (0.78–0.86) | 0.74 |

- **Retake decision:** it catches 74% of not-good-enough photos (162 of 220) and sends back 23% of good ones (42 of 180).
- **Naming:** on caught photos with a labelled problem, the first problem it names is a real one 89% of the time.
  By problem: focus-only 100% (26), aim-only 89% (18), obstruction-only 30% (10). Obstruction is the weak spot.
- **Another hospital:** heads trained on one DeepDRiD hospital score AUROC 0.75–0.93 on the other two
  ([results/first_build_report.txt](results/first_build_report.txt), section 2).
- **Another camera:** on IDRiD photos (India) that were picked for grading, it asks for a retake on 9%.
- **Guard:** it accepted 100% of fundus photos from DeepDRiD, IDRiD and the web. It turned away 100% of DeepDRiD
  widefield photos and of everyday pictures plus a black frame. Held-out fundus photos scored an energy of 0.3–6.0,
  widefield photos 17–54, and everyday pictures and a black frame 75–210
  ([results/energy_report.txt](results/energy_report.txt)).
- **Disc and macula positions** on IDRiD's 103 held-out photos: median error 0.08 disc diameters for both, all within
  1 disc diameter of the experts' marks. A brightest/darkest-spot baseline gets 0.21 for the disc and 0.82 for the
  fovea ([results/locator_report.txt](results/locator_report.txt), [figure](results/locator_examples.png)).
- **Which eye:** in a right eye the optic disc sits to the right of the macula. On DeepDRiD test photos where both
  are found (89–90% of photos), that measured layout names the eye correctly 94% of the time
  ([results/dinov3_report.txt](results/dinov3_report.txt)).
- **DINOv3 vs DINOv2:** we ran both on the same photos with the same rules, and set the switching rule in advance:
  DINOv3's mean validation AUROC had to be at least 0.01 higher. It was 0.898 vs 0.890, so we kept DINOv2. Test
  results were mixed (focus 0.92 vs 0.90; overall 0.81 vs 0.83), and the locator tied. DINOv2 is also Apache 2.0 and
  needs no sign-up ([results/dinov3_wider_grid.txt](results/dinov3_wider_grid.txt),
  [notebooks/6_dinov3_ab.ipynb](notebooks/6_dinov3_ab.ipynb)).
- **Where the obstruction is:** part of the obstruction head's score is an average over the photo's patches, so it
  splits exactly into a per-patch map. Each patch is compared with the same place on a typical clean photo. On the 400
  DeepDRiD validation photos, that part of the score alone scores AUROC 0.94 for artifact ≥ 6 (the whole head: 0.95),
  and the marked area grows with the artifact score (Spearman 0.81), which is defined by how much of the photo is
  covered. SAM 3.1 marked nothing on the same kind of photos. We also pasted a shadow and a glare at random places on 40
  clean photos. The map's strongest spot landed on them as follows (chance: 7%):

  | | DINOv3 | DINOv2 |
  |---|---|---|
  | Pasted shadow | 85% | 92% |
  | Pasted glare | 80% | 52% |

  The map uses DINOv3 because it found glare far more often, while the checks stay on DINOv2
  ([results/obstruction_map_report.txt](results/obstruction_map_report.txt),
  [real photos](results/obstruction_map_examples.png), [pasted](results/obstruction_map_synthetic.png)).
- **What the instruction says about the obstruction:** with the DINOv3 map, the side of the photo it names for the
  pasted shadows and glare was:

  | | Shadow (named for 35 of 40) | Glare (named for 28 of 40) |
  |---|---|---|
  | The right side | 24 | 17 |
  | The next side over | 6 | 6 |
  | Farther off | 0 | 0 |
  | Called darker / brighter correctly | 33 | 25 |

  The rest mixed up "centre" with a side. On the 125 validation photos the app sends back for an obstruction, it names
  a place for 123. Mapping takes about 0.1 s a photo on a T4 GPU
  ([results/obstruction_map_build_report.txt](results/obstruction_map_build_report.txt)).

**Limits, plainly:**
- **Noisy labels.** DeepDRiD's own labels disagree with each other. 34% of photos graded good enough have a problem
  in their sub-scores, and 36% of not-good-enough photos have none. That limits any detector scored against the
  overall label.
- **The test set was scored twice:** once with the first heads ([results/first_build_report.txt](results/first_build_report.txt))
  and once with the final ones. All choices were made on validation photos.
- **Field definition scores follow measured positions only loosely** (AUROC 0.66). So the aim head decides *whether*
  aim is off, and the locator says *what* is off. Screening takes two views of each eye, one centred on the macula
  and one on the optic disc, and the two structures sit about 2.5 disc diameters apart. So the locator first decides
  which view a photo is (whichever structure is nearer the centre), then judges only that view: is its structure
  within 2 disc diameters of the centre, and is the other one well inside the picture (within 3; it normally sits
  about 2.5 away)? When the layout already fits the view, the instruction says the view is close and asks to
  fine-tune it. An earlier version asked disc-view photos to move the
  macula to the centre; the visit checklist caught the contradiction.
- **The obstruction map can mark the wrong place.** On clean photos it sometimes marks the darker side near the
  macula or the dark rim, and it mostly misses a photo that is dark all over. So it is shown only on retakes: always
  when the obstruction check asks for one, and on a retake for another reason only when it marks more of the retina
  than 95% of clear validation photos do (5.4%). Then the instruction mentions it too. About 1 in 20 clear photos
  marks that much, so on a retake for another reason the extra area is sometimes not a real obstruction. In that case
  the area within 1.25 disc diameters of the measured macula is left out, because the macula is naturally darker: the
  first version marked the macula of 354_r1 as a shadow. The pasted shadows and glare are a stand-in for real ones, and the marking
  level was set on the same validation photos (1% of patches on clean photos).
- **Training data comes from one dataset** (three hospitals). On synthetically damaged IDRiD photos, it often named
  focus for shadows and crops. Synthetic damage is a weak stand-in for real problems, but it is a reason to check a
  new camera before relying on the tool.

## Energy in Retake Coach

Energy runs through the tool, and each use is measured. `scripts/energy_analysis.py` reruns the numbers below from cached
features at $0 ([results/energy_report.txt](results/energy_report.txt)).

- **The guard is an energy model.** A photo's energy is how far its DINOv2 features sit from a Gaussian fitted on
  ordinary fundus photos (Mahalanobis distance per dimension). Above the limit, 6.9 (the 99th percentile on held-out
  fundus photos), the photo is turned away.
- **The same energy ranks bad photos higher without seeing a label.** The Gaussian was fitted on good and bad fundus
  photos alike, with no quality labels. On the DeepDRiD test set it still gives not-good-enough photos higher energy:

  | Problem (DeepDRiD test, 400 photos) | AUROC, energy alone (no labels) | AUROC, trained check |
  |---|---|---|
  | Not good enough overall | 0.72 | 0.82 |
  | Obstruction | 0.80 | 0.95 |
  | Focus | 0.77 | 0.90 |
  | Aim | 0.54 | 0.85 |

  Most of the quality signal is there before any label; the labels add aim and sharpen the rest.
- **A new camera is added by label-free energy adaptation.** A second energy is fitted on the new camera's own photos,
  and a photo passes if either energy finds it familiar. Each camera photo is judged by a model fitted without it
  (`notebooks/8_eyer_check.ipynb`).
- **The obstruction map splits an energy gap.** Read as an energy model, the obstruction check's score is the energy gap
  between "clear" and "obstructed". The patch part of that gap splits exactly into one term per patch, and that is the
  map.

What did not work, plainly:
- The energy does not tell when the checks are wrong (AUROC 0.45; 0.50 is chance). The checks' own distance from their
  threshold does better (0.69).
- Contrastive energies trained on synthetic damage did not beat a sharpness check on real photos
  (`notebooks/1_quality_energy_test.ipynb`). The free energy of Muse Glimmer's answers separated right from wrong only
  weakly on an MMLU pilot ([experiments/README.md](experiments/README.md)).

## Run it

**In Colab** (what the demo video uses). Put the notebooks' data in a Drive folder `MyDrive/ebl_fundus/`, including
the IDRiD zips (*A. Segmentation*, *B. Disease Grading*, *C. Localization*). Then run in order:

1. `notebooks/1_quality_energy_test.ipynb`: caches DINOv2 features for the DeepDRiD training and validation photos.
   It also holds the zero-shot energy experiment, which did not beat a sharpness check.
2. `notebooks/2_muse_coach_pilot.ipynb`: Muse Spark as a judge. Capped at 20 requests and $0.25. Optional.
3. `notebooks/3_retake_coach_build.ipynb`: test, IDRiD, widefield and everyday-picture features, and the first report.
4. `python scripts/final_model.py --features quality/features.npz --build quality/build_features.npz --labels Challenge2_labels.xlsx`:
   the final heads, thresholds and guard. This needs no GPU or API.
5. `notebooks/4_aim_locator_build.ipynb`: the disc and macula locator.
6. `notebooks/6_dinov3_ab.ipynb`: DINOv2 against DINOv3 on identical photos and rules ($0). It also saves the DINOv3
   heads the obstruction map reads. DINOv3 is gated: accept its licence on Hugging Face and put `HF_TOKEN` in Colab
   Secrets.
7. `notebooks/10_obstruction_map_build.ipynb`: fits the obstruction map and saves `quality/obstruction_map.npz`. It
   also checks the sentences against the pasted shadows and glare, and tests Muse Spark rewriting the instruction from
   words only (capped at $0.04); the tool does not use those rewrites.
8. `notebooks/5_run_demo.ipynb`: starts both pages and prints their links. Muse Spark (the coach's note) is capped at
   $0.25 a day and $1 in total. Photos dropped into `eyer_inbox/` in the Drive folder appear on the many-photos page as
   they arrive.
   Without the map file, obstruction retakes use standard wording.

Further checks, each optional:
- `python scripts/energy_analysis.py --build quality/build_features.npz --labels Challenge2_labels.xlsx`: the energy
  numbers above ($0, no GPU).
- `notebooks/7_sam_coverage_pilot.ipynb`: does SAM 3.1's outline of shadows and glare follow DeepDRiD's artifact
  score, which is defined by coverage? Capped at 80 requests and $0.20. **It did not.** Asked for "dark shadow" and
  "bright glare", SAM 3.1 outlined nothing on any of 39 validation photos, including ones with an obvious shadow or
  glare. On the same photos the obstruction head scored AUROC 0.93 for artifact ≥ 6
  ([results/sam_coverage_report.txt](results/sam_coverage_report.txt)). The same code did get outlines back in our
  earlier lesion tests ("yellow spot" on hard exudates, see [experiments/README.md](experiments/README.md)), so this
  is how SAM 3.1 answered these prompts, not a parsing failure. Other wordings, cropped tiles and point prompts were
  not tried. For now the tool does not use SAM 3.1 for shadows or glare.
- `notebooks/8_eyer_check.ipynb`: scores your own Eyer2 photos, named by what you meant to take, and onboards the
  camera ($0). An optional last cell, off by default, asks SAM 3.1 to outline the lid and lashes on photos named
  `lid` (at most 6 requests, $0.015).
- `notebooks/9_obstruction_map_test.ipynb`: the obstruction map test above, for both DINOv2 and DINOv3 ($0).

**Locally:**

```bash
pip install -r requirements.txt
export MODEL_API_KEY=...        # optional; Muse Spark writes the coach's note, otherwise a standard summary
export HF_TOKEN=...             # optional; DINOv3 for the obstruction map (accept its licence on Hugging Face first)
python app.py                   # http://127.0.0.1:8000, uses models/coach_model.npz, locator.npz and obstruction_map.npz
python app.py --watch ~/EyerInbox   # also checks each new photo that lands in that folder (see /batch)
```

`MUSE_DAILY_USD`, `MUSE_DAILY_REQUESTS` and `MUSE_TOTAL_USD` set the coach's-note caps (defaults $0.25, 60 and $1.00).
Put sample photos in `examples/`, with an optional `labels.json` of notes, to get a row of samples on the page.

## Built during the event

Everything in this repository was built during the hackathon (Oct 26 – Nov 4, 2026). One exception: the
experiments in `experiments/` that retrain a diabetic-retinopathy grader use training code from our earlier
IntelliSys 2026 paper. That code is not included or needed for Retake Coach. See
[experiments/README.md](experiments/README.md) for the directions we tried first, with their results, including what
did not work.

Total Meta Model API spend for the whole event, experiments included, was under $3.

## Data and licences

- **DeepDRiD** ([github.com/deepdrdoc/DeepDRiD](https://github.com/deepdrdoc/DeepDRiD), CC BY-SA 4.0): photo-quality
  labels for training and testing, plus widefield photos for the guard.
- **IDRiD** ([idrid.grand-challenge.org](https://idrid.grand-challenge.org/Data/), CC BY 4.0;
  [Porwal et al. 2018](https://dx.doi.org/10.21227/H25W98)): expert-marked disc and fovea centres, and a second camera.
- **scikit-image** sample pictures for the guard's everyday-picture check.
- **DINOv2** (Apache 2.0) and **DINOv3** (DINOv3 License, gated on Hugging Face) download on first run; neither is
  included here.

No images are stored in this repository except the result figures and the screenshot, which show DeepDRiD and IDRiD
photos under their licences. `models/` holds weights fitted on these labels; see [models/README.md](models/README.md).
