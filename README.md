# carla-ranging-validation

**When does monocular range estimation become unsafe in bad weather?**

---

## 1. Title and one-line pitch

> A reliability benchmark for monocular range estimation in CARLA, reporting
> degradation not as detection accuracy but as **brake latency in seconds and
> metres**.

For your CV:

> Built a scenario-based validation harness in CARLA measuring monocular
> ranging degradation across 8 weather/lighting conditions on a bit-identical
> ego trajectory, reporting failure as time-to-collision brake latency rather
> than detection accuracy.

---

## 2. Scope

### What this does

A camera-only system estimates how far away the car in front is. That estimate
gets worse in rain, at night, and into low sun. **This measures how much worse,
and converts that into how late the brake command arrives.**

Two independent estimators run against exact CARLA depth ground truth across
eight conditions. The ego drives a byte-identical path in every condition, so
weather is the only variable.

### What "done" looks like

One sentence, quantified:

> Under `HardRainNight`, the ground-plane estimator crosses the 2.0 s brake
> threshold **0.31 s late** — 4.3 m of extra travel at 50 km/h — and misses the
> trigger entirely on 12% of closing events.

**If every estimator performs well in every condition, the project has failed.**
A benchmark that finds no failure boundary has measured nothing. `src/report.py`
checks for this and tells you when the condition matrix needs to be harsher.

### Explicitly in scope

- Deterministic, reproducible data capture (byte-identical across runs)
- Occlusion-filtered ground truth from depth + semantic buffers
- Two ranging estimators with uncertainty models, plus fusion
- Range error → TTC error → brake latency
- Honest documentation of what the benchmark cannot claim

### Explicitly out of scope

- **No ROS 2.** Nothing imports `rclpy`. Not needed at any phase.
- **No Docker.** Native `CarlaUE4.exe` on Windows.
- **No closed-loop control.** Ego speed and true range are both logged, so brake
  latency is computed offline. Same result, none of the PID-tuning risk.
- **No detector training** (that's optional Phase 5, after everything else works)
- **No sim-to-real validation.** Condition *ordering* is trustworthy; absolute
  numbers are not.

### Time

Phases 0–4: **3–4 weeks** at student pace. Most of it in Phase 0 (determinism)
and Phase 2 (label verification). Phase 5 only if those are finished.

---

## 3. Prerequisites

| | Need | Yours |
|---|---|---|
| GPU | NVIDIA ≥6 GB VRAM | RTX 4060 8GB — above spec at `-quality-level=Low` |
| CPU | 6+ cores (the real bottleneck) | Ryzen 7 7435HS, 8C/16T — fine |
| RAM | 16 GB min | |
| Disk | ~30 GB | |
| OS | Windows 10/11 | |

**Install exactly two things:**

1. **CARLA 0.9.15** — download `CARLA_0.9.15.zip` from
   [the 0.9.15 release](https://github.com/carla-simulator/carla/releases/tag/0.9.15).
   Extract to a short path: `D:\CARLA_0.9.15`. Then:
   ```powershell
   [Environment]::SetEnvironmentVariable("CARLA_ROOT","D:\CARLA_0.9.15","User")
   ```
   Reopen PowerShell afterwards.

2. **Miniforge** (or Miniconda) — then `conda init powershell`, reopen the terminal.

Nothing else. **Not Docker, not Ubuntu, not WSL2, not ROS 2.**

> Docker only ever existed in this project as a workaround for running
> Ubuntu-22.04-targeted CARLA binaries on Ubuntu 24.04. That's a Linux-only
> problem. Windows has a native build.

### Knowledge assumed

- Pinhole camera model, intrinsics, perspective projection
- Homogeneous transforms — CARLA/Unreal is **left-handed** (x forward, y right, z up)
- `TTC = range / closing_speed`

---

## 4. Every file, and what it does

Two environments that never import each other. `carla38` runs Python 3.8 because
the CARLA API requires it; `percep` runs 3.11 because modern NumPy/pandas/PyTorch
do. **They communicate only through files in `dataset\`.** `run_all.ps1` switches
between them via `conda run` — never activate one by hand.

```
carla-ranging-validation\
│
├── run_all.ps1                  ← THE ONLY THING YOU RUN. Orchestrates all phases.
│
├── scripts\                     ══ CARLA side — env: carla38 (Python 3.8) ══
│   ├── preflight.ps1                Checks GPU, driver, CARLA install, envs, disk,
│   │                                power plan, and the semantic tag constant.
│   │                                RUN THIS FIRST. Fails loudly and early.
│   ├── start_server.ps1             Launches CarlaUE4.exe, polls until the RPC
│   │                                port actually answers (not just "process alive").
│   ├── carla_capture.py             THE CORE CAPTURE. Synchronous mode, seeded
│   │                                traffic, frame-matched sensors, uint16 depth,
│   │                                per-frame actor ground truth.
│   └── run_matrix.ps1               Sweeps all 8 conditions, then verifies the ego
│                                    trajectory is identical across every one.
│
├── src\                         ══ Analysis side — env: percep (Python 3.11) ══
│   ├── config.py                    Loads conditions.yaml. Single source of truth —
│   │                                both Python and PowerShell read conditions here.
│   ├── labels.py                    Projects 3D actor boxes into the image, filters
│   │                                occluded vehicles using depth + semantics,
│   │                                emits true range. HIGHEST-RISK FILE.
│   ├── ranging.py                   Two estimators + inverse-variance fusion, each
│   │                                with an uncertainty model.
│   ├── inspect_labels.py            Builds a contact sheet so you can eyeball label
│   │                                quality in ten seconds instead of eight files.
│   ├── evaluate.py                  Range error → TTC error → brake latency.
│   └── report.py                    Figures + RESULTS_AUTO.md. Warns you if no
│                                    failure boundary was found.
│
├── configs\conditions.yaml      The independent variable: 8 conditions with
│                                severity ranks. EDIT HERE, nowhere else.
│
├── tests\test_geometry.py       12 synthetic geometry tests. No server needed,
│                                runs in 1 second. Catches projection bugs before
│                                they masquerade as perception results.
│
├── environment\
│   ├── carla38.yml              Python 3.8 + carla 0.9.15 + opencv
│   └── percep.yml               Python 3.11 + numpy/pandas/matplotlib/pytest
│
├── results\RESULTS.md           YOU WRITE THIS BY HAND. Template with empty tables.
└── dataset\                     Generated. Gitignored. ~1.5 GB.
```

### The three files that actually matter

| File | Why |
|---|---|
| `scripts\carla_capture.py` | If capture isn't deterministic, nothing downstream means anything. |
| `src\labels.py` | If ground truth is wrong, every result is confidently wrong. Errors here are silent. |
| `src\evaluate.py` | Where range error becomes a safety number instead of a statistic. |

---

## 5. How to run it

### One-time setup

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass

conda env create -f environment\carla38.yml
conda env create -f environment\percep.yml

Add-MpPreference -ExclusionPath (Join-Path (Get-Location) "dataset")
```

The Defender exclusion is not optional advice — a 500-frame matrix writes ~16,000
small files, and real-time scanning on every write can roughly double capture time.

### Every session

```powershell
.\scripts\preflight.ps1        # fix every FAIL before continuing
.\scripts\start_server.ps1     # ~60s to load
.\scripts\preflight.ps1        # again — only now can it verify the semantic tag
```

### The pipeline

```powershell
.\run_all.ps1 -Frames 100      # SHORT FIRST PASS — do this before the real run
```

Once that completes cleanly end to end:

```powershell
Remove-Item -Recurse -Force .\dataset, .\results
.\run_all.ps1                  # full 500-frame run
```

Resume after a gate: `.\run_all.ps1 -From 2`. Single phase: `-Only 3`.

**Run the 100-frame pass first.** Phase 0 tells you within two minutes whether
determinism holds. Discovering it doesn't after a 15-minute matrix is wasted time.

---

## 6. The five phases

| Phase | Does | Acceptance test | Env |
|---|---|---|---|
| **0** | Captures the same run twice | File hashes byte-identical | carla38 |
| **1** | Sweeps 8 weather conditions | Ego trajectory identical across all | carla38 |
| **2** | Generates ground-truth labels | **You open the images and look** | percep |
| **3** | Computes error + brake latency | Degradation trend visible; something fails | percep |
| **4** | Figures + report | You write the argument | percep |
| **5** | *(optional)* Real detector | Only after 0–4 are done | percep |

`run_all.ps1` pauses at each gate and asks. Answer honestly — `-Yes` exists for
re-runs after you've already verified, not for the first pass.

### The two gates that need your eyes, not the exit code

**Phase 1:** the trajectory check must report identical ego paths. If it doesn't,
weather is confounded with route and every later number is meaningless.

**Phase 2:** open `dataset\ClearNoon\debug_000042.png` and
`dataset\contact_sheet.png`. Check that boxes sit **on** the vehicles, that no box
is drawn on a car hidden behind a building, and that ranges look plausible.

This is the highest-risk step in the project. A wrong semantic tag, a dropped
bbox offset, or a mis-remapped axis all produce plausible-looking numbers and a
completely invalid benchmark. `preflight.ps1` verifies the semantic tag against
your live build for exactly this reason.

---

## 7. Bugs already found by testing against known answers

Two bugs surfaced during development, both of which produced *plausible numbers
rather than crashes*:

1. **Range definition.** The projected 2D box is sized by the near face of the 3D
   box, but true range was defined as the centroid — a constant −2.4 m bias,
   larger than the weather effect being measured. It would have looked like a bad
   height prior.
2. **Degradation check.** The "no failure boundary" warning pooled all rows, so
   variation *between estimators* masked zero variation *between conditions*. The
   check designed to catch a null result would itself have silently failed.

Both caught by testing against data with a known answer. Keep that habit — it's
the thing that makes this repo worth showing.

---

## 8. Limitations to state in your writeup

Listing these is what separates a benchmark from a demo. A reviewer will find
them whether or not you do.

1. **Closing speed comes from ground truth.** Only range error propagates into
   TTC, so reported latencies are a **lower bound**.
2. **Sim-to-real gap unquantified.** CARLA's rain and night rendering aren't
   photometrically validated. Condition *ordering* is more trustworthy than
   absolute values.
3. **`-quality-level=Low`** changes lighting and shadows. Recorded in
   `run_config.json`; every result is conditioned on it.
4. **Fusion assumes independent errors.** They aren't, once the bounding box
   itself degrades — which is exactly the regime of interest.
5. **Class priors are coarse** — a five-class heuristic from blueprint IDs.
6. **Single map, single route.** Not shown to generalise across road geometry.
