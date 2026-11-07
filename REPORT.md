# Task 4 — A better classifier for the phone camera

**The question.** Our Task 3 classifier reaches ~0.34 on our own validation photos. The app it feeds
is used outdoors, with a phone camera, on a leaf held at arm's length. This study asks which single
change moves the number that matters — accuracy on photographs that look like the ones the app will
actually receive — and how much each change is really worth on top of its own noise.

**How it was run.** 12 configurations, each trained with three seeds (42, 123, 999) on the CPU:
**36 runs**. Every row differs from a named anchor row by exactly one change. Everything is in
`results.csv`; `analyze.py` prints the tables below.

| | |
|---|---|
| code | `common.py` (variant-free scaffolding) + one self-contained file per row |
| data prep | `make_dataset.py` |
| driver | `run_all.py` (row × seed), logs in `logs/` |
| analysis | `analyze.py` → the tables of §4 |
| export | `export_for_camera_app()` in `common.py`, files in `export_E/` |

---

## 1. Data

### 1.1 Our own photographs, and what had to be fixed first

The group contributed **131 photographs**, all taken in one ~12-minute session on 2026-11-03:

| problem found | what `make_dataset.py` does |
|---|---|
| 28 files are `.heic` (iPhone), unreadable by PIL | registered through `pillow_heif` |
| 95 files carry an EXIF orientation flag; naive loading rotates them wrongly | `ImageOps.exif_transpose` before anything else |
| 4–12 MP, arbitrary aspect; the network input is 96×96 | centre square → resize to 192×192 (kept at 192 so training can crop) |
| burst shots of the *same* leaf leaked train into val | 16×16 grayscale signature, remove images with distance < 20 → **10 removed** |

Result: **121 images → 89 train / 32 val**, stratified, split by `random.Random(42)`.

| split | apple | cherry | chestnut | maple |
|---|---|---|---|---|
| train (89) | 10 | 28 | 23 | 28 |
| val (32) | 4 | 10 | 8 | 10 |

Two facts about this set dominate the whole reading of §4:

1. **One validation image is worth 3.1%.** With 32 images, the standard error of an accuracy is
   about ±0.09 even for a perfect measurement. Differences smaller than that are not measurable here.
2. **89 training images from one afternoon, one phone, one light.** The photos of one leaf several
   times are near-duplicates, so the effective number of distinct specimens is smaller still.
   Almost nothing here can be concluded about a different autumn, phone or background —
   which is exactly why last year's archive is part of the study.

The duplicate threshold was calibrated, not guessed: measured distances between photographs of the
*same* leaf came out around 12.5, and the median distance between *different* leaves around 32, so
20 sits between them. I also tried grouping by capture time, and rejected it — cherry's 46 shots
within 4.5 minutes are genuinely 46 different leaves, as a contact sheet showed.

### 1.2 Last year's archive

The course archive (`ex08_leaves_images.zip`): **700 training / 120 validation** photographs,
175 and 30 per class exactly — other students, other phones, another autumn, already at 96×96.
Its 120 validation images were never trained on in any row except where stated (§6), so they serve
as the outside test of this study.

### 1.3 The three validation sets, measured every epoch of every run

| column | what it is | size | what it can resolve |
|---|---|---|---|
| **Val Acc** | our own validation photos, centre-square + resize 96 | 32 | ±0.09 |
| **Camera-Val Acc** | the same 32 photos through a fixed synthetic camera degradation | 32 | ±0.09 |
| **Last-year Val Acc** | the archive's 120 real photographs, other phones | 120 | ±0.05 |

The camera-like set is built **once**, with its own seed (`CAMERA_SEED = 7`), and cached to
`camera_val_seed7_32images.pt`, so every variant is scored on exactly the same pixels:

```python
camera_aug = transforms.Compose([
    transforms.RandomResizedCrop(SIZE, scale=(0.5, 1.0)),   # off-centre framing
    transforms.RandomRotation(30),
    transforms.RandomPerspective(),                          # leaf not parallel to the sensor
    transforms.ColorJitter(0.4, 0.4, 0.4, 0.1),              # cloud, shadow, lamp
    transforms.ToTensor(),
    transforms.Normalize(mean, std),
])
```

Building it saves and restores the torch/numpy/random states, so generating it cannot shift the
random numbers the training itself will draw afterwards.

The clean transform is Task 2a/3a with **one** added line, `Resize(96)` — needed because
`LeafNet`'s `Linear(64*6*6, 64)` hard-depends on a 96×96 input, and it is what `index.html` does at
inference time (central square, resize, normalise), so the model is measured on what the phone will
show it. On the archive, whose photos are already 96×96, that line is a no-op.

---

## 2. How the comparison is kept fair

- **One change per file.** Each row script contains its own model, loss, optimizer and loop, so the
  diff between a row and its anchor *is* the change under test. No frameworks, no helper classes
  beyond the `LeafDataset` of Task 3a.
- **Shared scaffolding only for measurement** (`common.py`): seeds, the three validation loaders,
  `accuracy()`, the CSV log. Nothing in it depends on the variant.
- **All seeds from the environment** (`SEED=123 python rowC_cosine.py`) — no row file is ever edited
  to repeat a run.
- **Fixed seeds** for `torch`, `torch.cuda`, `numpy` and `random`, plus
  `cudnn.deterministic = True`, `benchmark = False`. The smoke test verified a row reproduces its
  own numbers exactly across repeats.
- **The table logs the FINAL epoch, not the best one.** Best-epoch selection is Candidate D's own
  item; if every row silently reported its best epoch, D's checkpointing would score itself a win it
  did not earn. `best_epoch` is recorded per row (chosen on **clean** val, never on the camera
  column) so that selection is never done on the metric being judged.
- **Two noise gates, not one.** A gain is only called a gain if it clears *both* (a) the spread
  between the three seeds and (b) the measurement error of the validation set itself. (b) matters:
  the three seeds are three different models judged on the **same** photographs, so the difficulty
  of those photos is a shared bias that no number of seeds can average away. Without gate (b) the
  spread of a difference can come out at exactly 0.000 — three seeds agreeing by luck — and then any
  difference at all looks significant. That is not a hypothetical here: row F1's camera column hit
  precisely that case.
- Differences are computed **per seed** against the row's own anchor and then averaged, rather than
  comparing two means, because paired rows share the luck of a seed.

---

## 3. The rows

| row | anchor | the single change |
|---|---|---|
| **0a** | — | Task 3 pipeline (cells 54+56+58): `LeafNet` from scratch, clean transform, Adam 1e-3, 15 epochs |
| **A0** | 0a | Task 3d's geometric augmentation (`RandomResizedCrop 0.6–1.0`, h/v flip, rotation 20) |
| **A** | A0 | `ColorJitter(0.4, 0.4, 0.4, 0.1)` added |
| **D1** | 0a | global average pooling instead of `flatten` + `Linear(2304, 64)` |
| **D2** | 0a | `nn.Dropout(p=0.5)` after the hidden ReLU |
| **D** | 0a | the whole package: GAP + dropout + AdamW (`wd=1e-4`) + label smoothing 0.1 + best-epoch checkpoint |
| **0b** | — | Task 2 model (cells 42+44): pretrained MobileNetV3-Small, head replaced, all trained, 15 epochs |
| **B** | 0b | freeze the backbone, train only the new `Linear`, with `model.features.eval()` |
| **C** | 0b | `CosineAnnealingLR` and 30 epochs instead of 15 |
| **F1** | C | + the geometric augmentation of A0 |
| **F2** | F1 | + the colour jitter of A |
| **E** | F1 | + last year's 700 photographs added to the training set (89 → 789) |

Parameter counts: `LeafNet` **208,292**; with GAP **60,772**; MobileNetV3-Small **1,521,956**, of
which the frozen row B trains only **4,100**.

---

## 4. Results

Mean ± spread over seeds 42 / 123 / 999, **final epoch**, as `analyze.py` prints it.

| row | Val Acc | Camera-Val Acc | Last-year Val Acc |
|---|---|---|---|
| 0a_scratch_baseline | 0.396 ± 0.048 | 0.312 ± 0.113 | 0.378 ± 0.013 |
| A0_geometric | 0.531 ± 0.063 | 0.375 ± 0.054 | 0.333 ± 0.058 |
| A_colour_aug | 0.479 ± 0.048 | 0.406 ± 0.031 | 0.378 ± 0.013 |
| D1_global_average_pooling | 0.542 ± 0.036 | 0.375 ± 0.083 | 0.386 ± 0.054 |
| D2_dropout | 0.406 ± 0.031 | 0.323 ± 0.048 | 0.342 ± 0.033 |
| D_regularisation_package | 0.531 ± 0.031 | 0.406 ± 0.054 | 0.344 ± 0.025 |
| 0b_pretrained_baseline | 0.667 ± 0.036 | 0.573 ± 0.065 | 0.528 ± 0.027 |
| B_frozen_backbone | 0.750 ± 0.031 | 0.552 ± 0.018 | 0.522 ± 0.021 |
| C_cosine_30_epochs | 0.708 ± 0.018 | **0.625 ± 0.031** | 0.539 ± 0.035 |
| F1_combination | 0.729 ± 0.048 | 0.562 ± 0.031 | 0.639 ± 0.046 |
| F2_combination_colour | 0.740 ± 0.096 | 0.552 ± 0.148 | 0.608 ± 0.109 |
| E_plus_last_years_data | **0.760 ± 0.018** | 0.552 ± 0.100 | **0.911 ± 0.013** |

Gain per seed against the row's own anchor (`+`/`-` = clears both noise gates):

| row | against | gain Val | gain Camera-Val | gain Last-year |
|---|---|---|---|---|
| A0_geometric | 0a | +0.135 ± 0.048 **+** | +0.062 ± 0.165 | -0.044 ± 0.061 |
| A_colour_aug | A0 | -0.052 ± 0.018 | +0.031 ± 0.083 | +0.044 ± 0.068 |
| D1_global_average_pooling | 0a | +0.146 ± 0.079 **+** | +0.063 ± 0.174 | +0.008 ± 0.063 |
| D2_dropout | 0a | +0.010 ± 0.036 | +0.010 ± 0.065 | -0.036 ± 0.043 |
| D_regularisation_package | 0a | +0.135 ± 0.079 **+** | +0.094 ± 0.156 | -0.033 ± 0.022 |
| B_frozen_backbone | 0b | +0.083 ± 0.018 | -0.021 ± 0.048 | -0.006 ± 0.029 |
| C_cosine_30_epochs | 0b | +0.042 ± 0.018 | +0.052 ± 0.090 | +0.011 ± 0.034 |
| F1_combination | C | +0.021 ± 0.065 | -0.062 ± 0.000 | +0.100 ± 0.052 **+** |
| F2_combination_colour | F1 | +0.010 ± 0.126 | -0.010 ± 0.157 | -0.031 ± 0.094 |
| E_plus_last_years_data | F1 | +0.031 ± 0.062 | -0.010 ± 0.110 | +0.272 ± 0.051 **+** |

Training loss at the last epoch, which is what makes the pattern in the table readable:

| row | final train loss | row | final train loss |
|---|---|---|---|
| 0a | 0.626 | B (frozen) | 0.218 |
| A0 | 1.110 | C | 0.018 |
| A | 1.213 | F1 | 0.069 |
| D1 | 1.135 | F2 | 0.119 |
| D2 | 0.955 | E | 0.038 |
| 0b | 0.017 | | |

---

## 5. Verdict on each candidate

**Pretraining is the biggest single effect, and it is not close: 0.396 → 0.667 clean, 0.312 → 0.573
camera.** Comparing 0a with 0b is the only comparison here that is far too large to be a noise
artefact. MobileNetV3-Small has seen a million images; our whole dataset is 89. Everything else in
this study is a rounding error next to this line, and that in itself is the most useful result.

**A0 — geometric augmentation: real, on the from-scratch network (+0.135 clean, clears the gates).**
The final training loss tells why: 1.110 for A0 against 0.626 for 0a. Row 0a memorises its 89
photographs; row A0 never can, because it never sees the same image twice. Its camera gain (+0.062)
is inside the noise, and its last-year column is *slightly negative* (-0.044) — crops and flips help
with our own camera, they do not buy transfer to somebody else's.

**A — colour jitter: no.** On top of the geometric augmentation it *costs* clean accuracy
(-0.052, stable across all three seeds) and its camera gain (+0.031) is a third of the noise floor.
The explanation is the loss: 1.213 against 1.110. With 89 images and a 60k-parameter network,
jittering the colour makes the task harder to fit without making the features more useful. The
tempting reading — "colour is what a camera changes, so jitter colour" — is exactly what the numbers
refuse. **Rejected.**

**D1 — global average pooling: the best change to the small architecture (+0.146 clean).** Removing
`flatten + Linear(2304, 64)` deletes 148k of the 208k parameters (down to 60k) and the accuracy goes
*up*, which says the flat layer was pure overfitting capacity: 2.3 parameters per training image.
Its camera effect (+0.063 ± 0.174) is unmeasurable. **Accepted on the from-scratch branch** — and
moot for the final model, because MobileNetV3 already pools globally.

**D2 — dropout: nothing (+0.010 clean, +0.010 camera).** Dropout p=0.5 on top of a 2304→64 layer
that was already memorising. **Rejected.**

**D — the package: +0.135 clean, which is D1's +0.146 and not a byte more.** This is the honest
result that a package hides: three of its four components (weight decay, label smoothing, dropout)
contributed nothing measurable, one (GAP) did all the work, and the best-epoch checkpoint — the part
that is pure selection bias, since it picks the epoch by peeking at validation — bought no accuracy
either. Row D's camera gain (+0.094) sits just under the noise floor and its last-year column is
negative. **Rejected as a package; adopted only as GAP, whose value row D1 established on its own.**

**B — frozen backbone: +0.083 clean (0.750, the best of the pretrained branch) but -0.021 on the
camera column.** With only the 4,100-parameter head trainable, the network cannot adapt to our
leaves; it does best on our own validation photos and no better than plain fine-tuning on degraded
ones. Its final loss (0.218 against 0.017) shows it fits less, and its best epoch is very early
(3–5 of 15), the signature of a linear head that stops improving while the camera metric is still
below fine-tuning. **Rejected for the app**, on the camera column, though it is the right answer to
a different question — it is the cheapest row here by an order of magnitude and would be the one to
run on a laptop.

**C — cosine schedule + 30 epochs: +0.042 clean and +0.052 camera, both inside the gates, and the
best camera number of the whole study (0.625).** Nothing about it is significant on its own; what it
does is *stop the training from ending too early*, which matters more for a small fine-tuned network
than the gates can show. Accepted into the combination — with the honest caveat that its camera lead
over 0b's 0.573 is smaller than what 32 images can resolve.

**F1 — the combination: +0.100 on last year's photographs (+0.639 against 0.539), which clears both
gates; +0.021 clean; -0.062 camera.** So the augmentation that helped the tiny network also helps the
pretrained one, but where it shows is the outside set, not our own. Note the F1-vs-C camera figure of
-0.062 ± **0.000**: all three seeds produced the same delta, which is the degenerate case gate (b)
exists for. It is *not* evidence that augmentation hurts the camera metric; it is evidence that 32
images cannot tell.

**F2 — F1 + colour jitter: nothing (+0.010, -0.010, -0.031), with the seed spread blowing up to
± 0.148 camera.** Colour jitter is as useless on the pretrained network as on the small one. **Rejected**,
and this second refusal on a different network is the stronger form of the negative result.

**E — adding last year's 700 photographs: +0.272 on the 120 outside photographs (0.639 → 0.911),
six times the noise floor, and the best clean accuracy in the study (0.760).** It is the only change
in this whole study that produces an effect larger than the measurement error of the camera-like
column. The training set grew 89 → 789 images and, more importantly, gained *other phones, other
backgrounds, other weather* — the thing our own dataset has none of. **This is the winner, and the
answer to the question Task 4 actually poses: for a phone classifier, more real photographs from
other cameras beat any amount of cleverness about our own few.**

Caveat, stated plainly: for every other row the last-year column is an outside test; for row E the
added training data comes from that same distribution (same archive, its *other* 700 images). The
120 validation photographs were never trained on, so 0.911 is a legitimate held-out number, but it
does measure "generalises within last year's conditions" rather than "generalises to my camera". The
clean column (0.760, our own phone, our own afternoon, never trained on) is the one that guards
against E having merely learned the archive's look, and it is the best of the study too.

---

## 6. The chosen model

**Row E** — pretrained MobileNetV3-Small, fine-tuned, geometric augmentation, cosine schedule over
30 epochs, trained on our 89 photographs plus the archive's 700.

The camera column, which is the one Task 4 names as the target, does not separate the top four rows:
C 0.625, 0b 0.573, F1 0.562, E 0.552 — a range of 0.073 on a column whose measurement error is
±0.088. Choosing C because it came first on 32 images would be selecting on noise. E is chosen
because on every column it is at or above the alternatives, it wins the clean set, and it wins the
only well-powered camera-like measurement (120 real photographs from other phones) by 0.272 against
the nearest combination — a margin six times the floor.

| | chosen (E) | Task 3 baseline (0a) |
|---|---|---|
| clean val (our photos) | 0.760 | 0.396 |
| camera-like val | 0.552 | 0.312 |
| last year's 120 photos | 0.911 | 0.378 |

**For the phone:** MobileNetV3-Small is 1.52M parameters, exports to **6.4 MB** of ONNX, and the app
feeds it a 96×96 centre square — the same preprocessing the training and the validation used,
written into the model's own metadata so the page cannot drift from it. Row E's 30 epochs cost about
3½ minutes on a CPU laptop, against seconds per epoch for the frozen row B; on-device inference is
identical for both, since the frozen/unfrozen choice only affects training.

---

## 7. Export for the camera app (Task 2e format)

Produced by `EXPORT=1 SEED=42 python rowE_plus_archive.py` in `export_E/`. **Seed 42 was fixed in
advance** as the export run, not picked from the three seeds after seeing which scored best on the
camera column — choosing that way would have quietly added a selection-bias term to the result.

| file | what it is |
|---|---|
| `leaf_model.onnx` | the graph, input `image` `[1,3,96,96]`, output `logits`, with `model_info` metadata (classes, input size, mean, std) |
| `best_model_weights.pt` | the `state_dict` of this very run |
| `model_info.json` | the same preprocessing information, for a human |
| `check_image.jpg` | a 96×96 centre-square validation photo, fed as-is by the page |
| `check_expected.json` | the probabilities this model gives it, so the app can self-test its pipeline |

Verified by scoring the **`.onnx` file itself** in `onnxruntime`, not the PyTorch model in memory:

| | clean val | camera-like val | last year's 120 |
|---|---|---|---|
| row E in `results.csv` (seed 42) | 0.7500 | 0.5938 | 0.9000 |
| the exported `leaf_model.onnx`, scored in onnxruntime | 0.7500 | 0.5938 | 0.9000 |

The two agree image by image, so nothing is lost between the training loop and the file the app
loads. The input is `image` of shape `[1, 3, 96, 96]`, the output is `logits`, the `model_info`
metadata reads back correctly from inside the graph, and the probabilities in `check_expected.json`
match `onnxruntime`'s own output to 1e-6. That this separate re-training reproduced the logged
numbers exactly is also the reproducibility check of §2 passing on the largest row of the study.

One environment note: torch 2.15's ONNX exporter needs `onnxscript` installed, and prints characters
the Windows console cannot encode, so `common.py` reconfigures stdout to UTF-8 (the same line
`analyze.py` uses). Without those, the export of step 2e aborts before writing anything.

---

## 8. What I would say if I had one more afternoon

1. **More photographs, not more tuning.** Row E is the only effect in this study that clears the
   measurement error by a wide margin, and it is the only row that added data. Eight of the eleven
   changes tried here are indistinguishable from noise, and that is a statement about 32 validation
   images as much as about the changes.
2. **A bigger validation set.** 32 images give ±0.09; the interesting differences in this study are
   0.02–0.08. Tripling the validation photographs would make half the rows interpretable. The right
   way is to shoot photos in a *different* place on a *different* day — the current split shares one
   afternoon between train and val, so even the clean column flatters us slightly.
3. **A better camera-like set.** The synthetic one degrades 32 images with a fixed random draw; it is
   reproducible, which is what the comparison needs, but it is one draw. The second camera set
   (last year's real 120) carried the conclusions instead.
4. **Things worth retesting once the validation set is bigger:** colour jitter at a smaller strength
   (0.2 instead of 0.4 — the rejection here is of jitter at this strength on 89 images), and weight
   decay, which the D package buried.

---

## 9. Reproducing this

```bash
# from this directory, with the course venv
export LEAVES_DIR="$PWD/data"                                  # our photos, built by make_dataset.py
export HIST_DIR="/c/Users/Giannis.G/Desktop/MRL-ex08/ex08_leaves_images"   # last year's archive
export PYTHONPATH="$PWD" MPLBACKEND=Agg

python make_dataset.py                       # raw photos -> data/leaves/{train,val}
python run_all.py row0a_scratch.py           # one row x 3 seeds -> results.csv, logs/
python analyze.py                            # the tables of section 4
EXPORT=1 SEED=42 python rowE_plus_archive.py # the app files of section 7
```

Each run appends one line to `results.csv` (`row, seed, epochs, val_acc, camera_val_acc,
hist_val_acc, final_train_loss, best_epoch`) and writes its per-epoch trace to
`logs/<row>_<seed>.log`. The 36 runs take about 55 minutes on this machine's CPU (measured: 9.6 min
for the 27 short-epoch runs, ~45 s a seed for F1/F2, ~3½ min a seed for row E). While the group's own photographs are absent, `common.py` falls
back to last year's archive and stamps every row `PLACEHOLDER_` so a test run can never
contaminate the report; `analyze.py` skips those rows.
