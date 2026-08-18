# Results

> Fill this in as you go. It was written **before** any data was collected —
> deliberately. If you cannot name the columns of your results table before
> running the experiment, the experiment isn't defined yet, and you will end up
> fitting the question to whatever numbers happen to come out.

## Setup

| | |
|---|---|
| CARLA version | 0.9.15 |
| Map | Town10HD_Opt |
| Quality | Low, `-RenderOffScreen` |
| Resolution | 800 × 600, 90° FOV |
| Camera height | 1.6 m |
| Timestep | 0.05 s (20 Hz) |
| Frames per condition | 500 |
| Seed | 42, identical across all conditions |
| Trajectory check | ☐ passed (max ego drift: ___ m) |

## 1. Range error

`absrel` = mean absolute relative error. `invalid` = fraction of detections the
estimator could not range at all — report it next to accuracy, because an
estimator that refuses half its inputs can post an excellent absrel and still be
useless.

| Condition | Estimator | n | AbsRel | RMSE (m) | Bias (m) | Invalid |
|---|---|---|---|---|---|---|
| ClearNoon | height_prior | | | | | |
| ClearNoon | ground_plane | | | | | |
| ClearNoon | fused | | | | | |
| … | | | | | | |

## 2. Range error by distance

Expectation to test: `ground_plane` wins close in, `height_prior` wins far out,
because ground-plane uncertainty grows as Z² while height-prior grows as Z.

| Condition | Estimator | 0–20 m | 20–40 m | 40–60 m | 60–100 m |
|---|---|---|---|---|---|
| | | | | | |

## 3. Brake latency — the headline

Positive latency = brake command arrives **late**.

| Condition | Estimator | Events | Miss rate | Mean latency (s) | p95 (s) | Extra distance (m) |
|---|---|---|---|---|---|---|
| | | | | | | |

## 4. Failure boundary

**Degradation onset:** ________________

**Worst case:** ________________

One sentence, quantified, safety-framed. This is the line that goes in the
README and in your CV bullet:

> ________________________________________________

## 5. What surprised me

The most valuable section. Record anything that contradicted your expectation
before you ran it — a metric that lied, an estimator that won where you
predicted it would lose, a failure mode you didn't anticipate.

- 
- 

## 6. Errors caught during development

Bugs found and fixed, with how they were detected. Silent-but-wrong bugs are
worth more here than crashes: anyone can fix a traceback.

| # | Error | How it surfaced | Fix |
|---|---|---|---|
| 1 | | | |
