# NOT part of the experiment. A harness that proves the row scripts run before we start the
# real runs on the GPU.
#
# It makes a small fake group data set (four classes, a few images each: the same four files
# you would get from the teacher's ZIP, only invented), copies a row script with its
# "EPOCHS = 15" replaced by "EPOCHS = 2", and runs it in a throw-away folder, so the real
# results.csv and the real camera cache of task4/ stay untouched.
#
# What it must show, besides "it runs":
#   1. the same script, the same seed, twice -> exactly the same numbers   (the seeds work)
#   2. run 1 builds the camera set, run 2 loads it from the cache -> the same numbers again
#      (the cache is faithful and the camera set is identical for every variant)
#   3. the evaluation really reaches last year's 120 photographs.
#
# Run:  /c/Users/Giannis.G/Desktop/MRL-ex08/_venv/Scripts/python.exe smoke_test.py

import os
import shutil
import subprocess
import sys
import tempfile

import numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
VENV_PYTHON = r"C:\Users\Giannis.G\Desktop\MRL-ex08\_venv\Scripts\python.exe"
REAL_LAST_YEAR = r"C:\Users\Giannis.G\Desktop\MRL-ex08\ex08_leaves_images"   # HIST_DIR: holds leaves/train, leaves/val
CLASSES = ["apple", "cherry", "chestnut", "maple"]
COLORS = {"apple": (200, 60, 50), "cherry": (60, 170, 70), "chestnut": (70, 80, 200), "maple": (225, 200, 60)}


def make_fake_group(folder, seed=0):
    """One folder that looks like the teacher's ZIP: leaves/{train,val}/<class>/*.jpg."""
    rng = np.random.default_rng(seed)
    for split, n in [("train", 6), ("val", 3)]:
        for cls in CLASSES:
            class_folder = os.path.join(folder, "leaves", split, cls)
            os.makedirs(class_folder, exist_ok=True)
            for k in range(n):
                # 96 x 96, like every photograph of the course data set (checked: all 700 of
                # last year's images are exactly 96 x 96). LeafNet's flatten needs that.
                pixels = np.array(COLORS[cls]).reshape(1, 1, 3) + rng.normal(0, 22, (96, 96, 3))
                Image.fromarray(np.clip(pixels, 0, 255).astype("uint8")).save(
                    os.path.join(class_folder, f"leaf_{k}.jpg"))


def run(script, workdir, epochs=2):
    """A row script, with its epoch count replaced, executed in workdir."""
    os.makedirs(workdir, exist_ok=True)
    with open(os.path.join(HERE, script)) as f:
        source = f.read()
    assert "EPOCHS = 15" in source, f"{script}: the EPOCHS = 15 line changed, update the harness"
    name = script.replace(".py", "")
    patched = os.path.join(workdir, f"{name}.py")
    with open(patched, "w") as f:
        f.write(source.replace("EPOCHS = 15", f"EPOCHS = {epochs}"))

    env = dict(os.environ, LEAVES_DIR=fake_group, HIST_DIR=REAL_LAST_YEAR, SEED="42",
               MPLBACKEND="Agg", PYTHONPATH=HERE)
    done = subprocess.run([VENV_PYTHON, patched], cwd=workdir, env=env, capture_output=True, text=True)
    return done


def show(text, lines=14):
    """The tail of a run's output, without the download progress bar in the way."""
    keep = [line for line in text.split("\n")
            if line.strip() and not set(line.strip()) <= set("0123456789.%")]
    print("\n".join(keep[-lines:]))


def last_row(workdir):
    """The final numbers of the last run written to results.csv."""
    with open(os.path.join(workdir, "results.csv")) as f:
        return f.read().strip().split("\n")[-1].split(",")[3:7]



fake_group = os.path.join(tempfile.gettempdir(), "task4_fake_group")
root = tempfile.mkdtemp(prefix="task4_smoke_")
shutil.rmtree(fake_group, ignore_errors=True)
make_fake_group(fake_group)
print(f"fake group data set: {fake_group}\nlast year's photographs: {REAL_LAST_YEAR}\nwork space: {root}\n")

failures = []
for script in ["row0a_scratch.py", "row0b_pretrained.py"]:
    name = script.replace(".py", "")
    print("=" * 78, f"\n{name}\n" + "=" * 78)

    dir_a = os.path.join(root, name)
    first = run(script, dir_a)
    show(first.stdout)
    if first.returncode != 0:
        show(first.stderr, 25)
        failures.append(f"{name}: the script did not run")
        continue
    if "last year's val 120" not in first.stdout:
        failures.append(f"{name}: last year's validation set is not the 120 real photographs")

    # The same script, the same seed, again in the same folder: the numbers must repeat
    # exactly, with the camera set now taken from the cache instead of built.
    first_final = last_row(dir_a)
    second = run(script, dir_a)
    cached_final = last_row(dir_a)
    if "camera validation set built once" in second.stdout:
        failures.append(f"{name}: the camera set was built twice instead of cached")
    if cached_final != first_final:
        failures.append(f"{name}: the cached camera set gives other numbers than the built one "
                        f"-> {first_final} vs {cached_final}")

    # And once in a clean folder, where the camera set is built again from scratch:
    # the same seed has to give the same numbers as the first run.
    dir_b = os.path.join(root, name + "_clean")
    run(script, dir_b)
    clean_final = last_row(dir_b)
    if clean_final != first_final:
        failures.append(f"{name}: not reproducible -> {first_final} vs {clean_final}")
    print(f"run 1 (camera set built)   {first_final}\n"
          f"run 2 (camera set cached)  {cached_final}\n"
          f"run 3 (clean folder)       {clean_final}\n"
          f"columns: val_acc, camera_val_acc, hist_val_acc, final_train_loss")

print("=" * 78)
print("FAILURES:", failures if failures else "none - both baselines run, repeat exactly and see all three sets")
sys.exit(1 if failures else 0)
