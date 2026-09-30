# Fake Data, Real Questions

Workshop materials for **DISC x Meta** at Arizona State University (Oct 1, 2026):
"Fake data, real questions" — an interactive night of mock interviews and a
hand-tracking case study, built around synthetic Reality Labs-style datasets.

- **Slides:** https://docs.google.com/presentation/d/128R66pbk4UGT8BUdtW0fFTGqR_FQrsfbmeBINu3mITs/edit?usp=drivesdk
  (a frozen PowerPoint copy is included (`DISC-Talk-Fake-Data-Real-Questions.pptx`))
- **Everything runs in pandas + numpy + matplotlib.** No Spark needed.

## The data is fake; the questions are real

All three datasets are **synthetic**, generated with `gen_rl_datasets.py`
(seed `20261001`). They mimic the *shape* of hand-tracking telemetry so you can
run real analyses. **Nothing here is real Meta data.**

The joint/column vocabulary follows the public OpenXR `XR_EXT_hand_tracking`
extension: 26 joints per hand (wrist, palm, thumb x4, other fingers x5), each
with pose + radius and per-joint validity flags; `isActive` gates whether the
hand is tracked at all. Accuracy context comes from published Quest studies
(Abdlkarim et al.: ~1.1 cm fingertip error, ~9.6 deg joint-angle error,
~38 ms delay).

## The datasets

### 1. `ht_ab_holdout.csv` (20,000 users) — LIVE CASE
An A/B test of a new hand-tracking model (HT-v2), run on a **1% traffic
allocation** with 50/50 randomization (10,000 users per arm).
Columns: user_id, group (control | treatment), device (quest3 | quest3s),
gesture_attempts, gesture_successes, pinch_attempts, pinch_false_positives,
session_min, crashed (0/1).

**Question:** did HT-v2 actually improve gestures, and is it safe to roll out?
**Answer key:** gesture success 78.0% -> 81.2%, z = 35.4, overwhelmingly
significant at 1% (|z| > 2.58). Device mix is identical across arms, so
randomization holds. Guardrails are clean: pinch false positives flat
(4.5% vs 4.3%), crash rate flat (0.81% vs 0.74%). Session minutes tick up
22.2 -> 23.0 (directional, not the decision metric). Ship it.
Skills: two-proportion z-test, randomization checks, guardrail metrics.
**Visualization challenge:** sketch the readout chart by hand before you open
matplotlib. AI will offer a pie chart and a truncated y-axis; your job is the
chart that makes the 3.2-point lift honest.

Starter:
```python
import pandas as pd, numpy as np
df = pd.read_csv("ht_ab_holdout.csv")
g = df.groupby("group")
rate = g["gesture_successes"].sum() / g["gesture_attempts"].sum()
print(rate.round(4))
p1, p2 = rate["treatment"], rate["control"]
n1, n2 = g["gesture_attempts"].sum()["treatment"], g["gesture_attempts"].sum()["control"]
z = (p1-p2)/np.sqrt(p1*(1-p1)/n1 + p2*(1-p2)/n2)
print("z =", round(z,1))
```

### 2. `ht_model_eval.csv` (8,000 samples) — TAKE-HOME
Model quality by condition: was 6 weeks of field data collection worth it?
Columns: sample_id, model (ht_v1 | ht_v2), lighting (bright | dim | low),
motion (slow | fast), joint_group (fingertips | knuckles | palm_wrist),
mpjpe_mm (mean per-joint position error vs motion-capture ground truth),
tracked (0/1, POSITION_TRACKED_BIT analogue), pinch_correct (0/1).

MPJPE in one line: the average distance in mm between the predicted joint
positions and the true ones. Lower is better. It needs ground-truth joint
positions, which online data never has, so it is scored on an **offline mocap
set** with known truth. Field data trains the model; mocap data grades it.

**Question:** did HT-v2 get better, and where does it still fail?
**Answer key:** overall MPJPE 13.7 -> 9.8 mm. Biggest gains exactly where the
field data was collected: dim+fast 18.6 -> 11.3 mm, low+fast 22.2 -> 13.4 mm.
Fingertips remain the hardest joint group (14.9 -> 11.0 mm); the ranking is
unchanged, so the next collection should overweight fingertip-heavy gestures.
Tracked rate 90.3% -> 94.3%, pinch accuracy 86.8% -> 91.6%. All three metrics
agree.
Skills: groupby segmentation, error heat maps, reading a table like a DS.
**Visualization challenge:** build the heatmap (lighting x motion, faceted by
model). The groupby table hides the story; the heatmap shows where the field
data paid off. Annotate the two cells that justify the 6-week spend.

Starter:
```python
df = pd.read_csv("ht_model_eval.csv")
print(df.groupby(["model","lighting","motion"])["mpjpe_mm"].mean().round(1))
```

### 3. `release_perf.csv` (9,000 sessions) — TAKE-HOME
Release regression hunt: v69 shipped Tuesday, users say tracking feels worse.
Columns: session_id, release (v68 | v69 | v70), device (quest3 | quest3s),
jitter_mm, tracking_loss_per_hr, mtp_latency_ms (motion-to-photon, guardrail),
battery_drain_pct_hr (guardrail).

**Question:** is the regression real, which devices, and did v70 fix it?
**Answer key:** overall jitter rose +21% in v69 (2.09 -> 2.53 mm), but the real
story is in the device cut: Quest 3S went 2.09 -> 3.01 mm (+44%) while Quest 3
sat flat near 2.1. Tracking loss agrees (3S: 1.20 -> 1.87 events/hr). The flat
Quest 3 majority diluted the 3S regression — aggregation bias in the
Simpson's-paradox family (dilution, not a full reversal). v70 fixed the
regression (3S jitter back to 2.17 mm). Latency flat at 42 ms throughout.
Trade-off: v70 battery drain rose 18.0 -> 18.9 %/hr. Ship v70, file the
battery cost as known, investigate the 3S-specific path in v69.
Skills: before/after comparison, segment cuts, guardrail trade-offs, writing
the one-paragraph recommendation.
**Visualization challenge:** plot jitter by release, faceted by device (small
multiples). The overall-average line understates the problem; the faceted view
is the entire case. Put the battery trade-off on the same figure so the
decision is visible at a glance.

Starter:
```python
df = pd.read_csv("release_perf.csv")
print(df.groupby(["release","device"])["jitter_mm"].mean().round(2))
```

## Why the chart is yours to draw

AI writes the SQL, picks a chart type, and drafts the dashboard in seconds,
and the chart it picks is often wrong for the message. Chart choice is a
judgment call: what to encode, what to omit, where the axis starts, what the
title claims. That judgment is the durable human skill in an AI-assisted
workflow, and it is teachable. Start here:

- *Storytelling with Data* by Cole Nussbaumer Knaflic
- *The Visual Display of Quantitative Information* by Edward Tufte
- *Show Me the Numbers* by Stephen Few

## Ground rules

- Say "fake data" out loud on every case study. The *questions* are real; the numbers are not.
- OpenXR terms and the published accuracy figures are public. Everything else is invented.
