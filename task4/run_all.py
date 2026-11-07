# Runs the study on this machine. It trains nothing itself: it starts each row script once per
# seed, one after another, keeps the whole output in logs/ and shows the table row each run
# writes. The seeds are the three the exercise asks for; SEED reaches the script through the
# environment, so no row file is ever edited to repeat it.
#
#   python run_all.py                       the default list below
#   python run_all.py rowE_history.py       only the files named on the command line
#
# Run:  /c/Users/Giannis.G/Desktop/MRL-ex08/_venv/Scripts/python.exe run_all.py

import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
VENV_PYTHON = r"C:\Users\Giannis.G\Desktop\MRL-ex08\_venv\Scripts\python.exe"
LAST_YEAR = r"C:\Users\Giannis.G\Desktop\MRL-ex08\ex08_leaves_images"   # already on this disk, no download
SEEDS = [42, 123, 999]

DEFAULT = ["row0a_scratch.py", "rowA0_geometric.py", "rowA_colour.py",
           "rowD1_gap.py", "rowD2_dropout.py", "rowD_full.py",
           "row0b_pretrained.py", "rowB_frozen.py", "rowC_cosine.py"]

scripts = sys.argv[1:] or DEFAULT
os.makedirs(os.path.join(HERE, "logs"), exist_ok=True)

print(f"{len(scripts)} rows x {len(SEEDS)} seeds = {len(scripts) * len(SEEDS)} runs, one after another\n", flush=True)

started, clock = 0, time.time()
for script in scripts:
    for seed in SEEDS:
        started += 1
        env = dict(os.environ, SEED=str(seed), LEAVES_DIR=os.path.join(HERE, "data"),
                   HIST_DIR=LAST_YEAR, MPLBACKEND="Agg", PYTHONPATH=HERE)
        run_started = time.time()
        done = subprocess.run([VENV_PYTHON, os.path.join(HERE, script)], cwd=HERE, env=env,
                              capture_output=True, text=True)
        seconds = time.time() - run_started
        stamp = f"[{started}/{len(scripts) * len(SEEDS)}] {script.replace('.py', '')} seed {seed}"

        if done.returncode != 0:
            print(f"{stamp}: FAILED after {seconds:.0f}s", flush=True)
            print(done.stderr[-1500:], flush=True)
            with open(os.path.join(HERE, "logs", f"FAILED_{script[:-3]}_{seed}.log"), "w") as f:
                f.write(done.stdout + "\n" + done.stderr)
            continue

        with open(os.path.join(HERE, "logs", f"{script[:-3]}_{seed}.log"), "w") as f:
            f.write(done.stdout)
        table_row = [line for line in done.stdout.split("\n") if line.startswith("| ")]
        last_epoch = [line for line in done.stdout.split("\n") if line.startswith("epoch ")][-1]
        print(f"{stamp}: {seconds:5.0f}s  {table_row[0].strip('| ').strip() if table_row else 'no row logged'}",
              flush=True)
        print(f"    {last_epoch}   (elapsed {time.time() - clock:.0f}s)", flush=True)

print(f"\nall done in {(time.time() - clock) / 60:.1f} minutes -> {os.path.join(HERE, 'results.csv')}")
