# Turns results.csv into the table of the report.
#
# Every configuration was run with the same three seeds, so a difference is computed PER SEED
# against its own anchor row and then averaged. That matters: an unpaired mean - mean hides the
# fact that both rows were unlucky or lucky on the same seed, and on a 32 image validation set
# the seed noise is of the same size as most of the effects we are looking for. The spread of
# those per seed differences is the noise gate - a change whose mean gain does not clear it is
# reported as no change, however promising its average looks.
#
# Run:  /c/Users/Giannis.G/Desktop/MRL-ex08/_venv/Scripts/python.exe analyze.py

import csv
import os
import sys
from collections import defaultdict

import numpy as np

sys.stdout.reconfigure(encoding="utf-8", errors="replace")   # the Windows console is not UTF-8

HERE = os.path.dirname(os.path.abspath(__file__))
SEEDS = [42, 123, 999]
METRICS = ["val", "camera_val", "hist_val"]

# Which row each configuration is compared against, seed by seed.
ANCHOR = {
    "A0_geometric": "0a_scratch_baseline",
    "A_colour_aug": "A0_geometric",
    "D1_global_average_pooling": "0a_scratch_baseline",
    "D2_dropout": "0a_scratch_baseline",
    "D_regularisation_package": "0a_scratch_baseline",
    "B_frozen_backbone": "0b_pretrained_baseline",
    "C_cosine_30_epochs": "0b_pretrained_baseline",
    # F1 is row C plus the augmentation, so against C it says what the augmentation is worth on
    # the pretrained network; F2 against F1 says the same for the colour part; E against F1 says
    # it for the seven hundred photographs of last year.
    "F1_combination": "C_cosine_30_epochs",
    "F2_combination_colour": "F1_combination",
    "E_plus_last_years_data": "F1_combination",
}
# The second noise gate, and the reason there are two. The spread between the three seeds measures
# how much the TRAINING wanders from run to run; it says nothing about the fact that all three runs
# were judged on the same photographs, whose difficulty is a shared bias that no number of seeds can
# average away. On 32 images the standard error of an accuracy is about 0.09 at best, and on last
# year's 120 about 0.05, so a gain smaller than that is not measurable here however stable its seeds
# are. Without this floor the spread of a difference can come out at exactly zero - three seeds that
# happen to agree perfectly - and then any difference at all looks significant.
NOISE_FLOOR = {"val": 0.088, "camera_val": 0.088, "hist_val": 0.045}

# The order the table shows, so the anchors come first and each branch reads top to bottom.
ORDER = ["0a_scratch_baseline", "A0_geometric", "A_colour_aug", "D1_global_average_pooling", "D2_dropout",
         "D_regularisation_package", "0b_pretrained_baseline", "B_frozen_backbone", "C_cosine_30_epochs",
         "E_plus_last_years_data", "F1_combination", "F2_combination_colour"]

runs = defaultdict(dict)
with open(os.path.join(HERE, "results.csv")) as f:
    for run in csv.DictReader(f):
        if run["row"].startswith("PLACEHOLDER_"):
            continue
        runs[run["row"]][int(run["seed"])] = {m: float(run[m + "_acc"]) for m in METRICS}

missing = {row: [s for s in SEEDS if s not in runs.get(row, {})] for row in runs}
for row, seeds in missing.items():
    if seeds:
        print(f"! {row} has no run for seed(s) {seeds}")


def mean_std(values):
    spread = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
    return float(np.mean(values)), spread


print(f"\n| row | seeds | Val Acc | Camera-Val Acc | Last-year Val Acc |")
print(f"|---|---|---|---|---|")
for row in ORDER + [r for r in runs if r not in ORDER]:
    if row not in runs:
        continue
    cells = []
    for metric in METRICS:
        values = [runs[row][seed][metric] for seed in SEEDS if seed in runs[row]]
        mean, std = mean_std(values)
        cells.append(f"{mean:.3f} ± {std:.3f}")
    n = len([s for s in SEEDS if s in runs[row]])
    print(f"| {row} | {n} | {cells[0]} | {cells[1]} | {cells[2]} |")

print(f"\nAgainst its own anchor, per seed (a gain that does not clear both noise sources is no gain):")
print(f"| row | against | gain Val | gain Camera-Val | gain Last-year |")
print(f"|---|---|---|---|---|")
for row, anchor in ANCHOR.items():
    if row not in runs or anchor not in runs:
        continue
    cells = []
    for metric in METRICS:
        deltas = [runs[row][s][metric] - runs[anchor][s][metric] for s in SEEDS
                  if s in runs[row] and s in runs[anchor]]
        if not deltas:
            cells.append("n/a")
            continue
        mean, std = mean_std(deltas)
        gate = max(std if len(deltas) > 1 else 0.0, NOISE_FLOOR[metric])
        mark = "" if abs(mean) <= gate else ("  +" if mean > 0 else "  -")
        cells.append(f"{mean:+.3f} ± {std:.3f}{mark}")
    print(f"| {row} | {anchor} | {cells[0]} | {cells[1]} | {cells[2]} |")

print(f"\n+ / - mean the mean gain is larger than BOTH noise sources for that metric: the spread between")
print(f"the three seeds, and the measurement error of the validation set itself "
      f"(±{NOISE_FLOOR['val']:.3f} on the 32 clean photos, "
      f"±{NOISE_FLOOR['camera_val']:.3f} on the 32 camera ones, ±{NOISE_FLOOR['hist_val']:.3f} on the 120 of last year).")
print("Note: three seeds give a std with two degrees of freedom. It is a noise scale, not a confidence interval.")
