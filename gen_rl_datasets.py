"""Generate 3 Reality Labs synthetic datasets for the DISC talk (seed 20261001).

All data is SYNTHETIC. Shapes mimic hand-tracking telemetry concepts from the
public OpenXR XR_EXT_hand_tracking extension (26 joints/hand, joint locations
with pose+radius, per-joint validity flags, isActive) and published Quest
accuracy figures (~1.1 cm fingertip error, ~9.6 deg joint-angle error, ~45 ms
delay; Abdlkarim et al., Behavior Research Methods 2023). Nothing here is
real Meta data.

Usage:
    python gen_rl_datasets.py            # writes CSVs next to this script
    python gen_rl_datasets.py out_dir/   # writes CSVs to out_dir/
"""
import os
import sys

import numpy as np
import pandas as pd

rng = np.random.default_rng(20261001)
OUT = sys.argv[1] if len(sys.argv) > 1 else os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(OUT, "")  # ensure trailing separator
os.makedirs(OUT, exist_ok=True)

# ---------------------------------------------------------------- 1. A/B holdout
# HT-v2 hand-tracking model, 1% holdout: control vs treatment.
n_arm = 10000
df = pd.DataFrame({
    "user_id": [f"u{i:06d}" for i in range(2 * n_arm)],
    "group": ["control"] * n_arm + ["treatment"] * n_arm,
})
df["device"] = rng.choice(["quest3", "quest3s"], size=len(df), p=[0.6, 0.4])
is_t = (df["group"] == "treatment").to_numpy()

df["gesture_attempts"] = rng.poisson(40, len(df)).clip(min=5)
p_succ = np.where(is_t, 0.812, 0.780)
df["gesture_successes"] = rng.binomial(df["gesture_attempts"], p_succ)

df["pinch_attempts"] = rng.poisson(25, len(df)).clip(min=3)
p_fp = np.where(is_t, 0.043, 0.045)
df["pinch_false_positives"] = rng.binomial(df["pinch_attempts"], p_fp)

df["session_min"] = rng.normal(np.where(is_t, 23.0, 22.0), 8, len(df)).clip(min=1).round(1)
df["crashed"] = rng.binomial(1, np.where(is_t, 0.0082, 0.0080))

df.to_csv(OUT + "ht_ab_holdout.csv", index=False)

print("== ht_ab_holdout.csv ==")
g = df.groupby("group")
succ = g["gesture_successes"].sum() / g["gesture_attempts"].sum()
print("gesture success rate:\n", succ.round(4))
fp = g["pinch_false_positives"].sum() / g["pinch_attempts"].sum()
print("pinch false-positive rate:\n", fp.round(4))
print("crash rate:\n", g["crashed"].mean().round(4))
print("mean session_min:\n", g["session_min"].mean().round(2))
print("device mix by group:\n", pd.crosstab(df["group"], df["device"], normalize="index").round(3))
p1, p2 = succ["treatment"], succ["control"]
n1, n2 = g["gesture_attempts"].sum()["treatment"], g["gesture_attempts"].sum()["control"]
z = (p1 - p2) / np.sqrt(p1 * (1 - p1) / n1 + p2 * (1 - p2) / n2)
print(f"z-test on gesture success: z = {z:.1f} (need |z| > 2.58 for 1% significance)")

# ---------------------------------------------------------------- 2. model eval
# Did 6 weeks of field data collection actually improve the model? By condition.
n = 8000
ev = pd.DataFrame({
    "sample_id": [f"s{i:06d}" for i in range(n)],
    "model": rng.choice(["ht_v1", "ht_v2"], size=n),
    "lighting": rng.choice(["bright", "dim", "low"], size=n, p=[0.5, 0.3, 0.2]),
    "motion": rng.choice(["slow", "fast"], size=n, p=[0.6, 0.4]),
    "joint_group": rng.choice(["fingertips", "knuckles", "palm_wrist"], size=n, p=[0.4, 0.4, 0.2]),
})
base = {  # (lighting, motion) -> v1 mean MPJPE in mm
    ("bright", "slow"): 9.0, ("bright", "fast"): 12.0,
    ("dim", "slow"): 13.0, ("dim", "fast"): 18.0,
    ("low", "slow"): 16.0, ("low", "fast"): 22.0,
}
improve = {  # v2 improvement in mm (bigger where field data was collected)
    ("bright", "slow"): 1.5, ("bright", "fast"): 3.0,
    ("dim", "slow"): 3.5, ("dim", "fast"): 7.0,
    ("low", "slow"): 5.0, ("low", "fast"): 9.0,
}
joint_adj = {"fingertips": 1.5, "knuckles": 0.0, "palm_wrist": -1.5}
means = []
for _, r in ev.iterrows():
    m = base[(r["lighting"], r["motion"])] + joint_adj[r["joint_group"]]
    if r["model"] == "ht_v2":
        m -= improve[(r["lighting"], r["motion"])]
    means.append(m)
means = np.array(means)
ev["mpjpe_mm"] = np.maximum(1.0, rng.normal(means, means * 0.22)).round(2)

p_track = 0.93 - 0.06 * (ev["lighting"] == "low") - 0.04 * (ev["motion"] == "fast")
p_track = np.where(ev["model"] == "ht_v2", p_track + 0.04, p_track).clip(0.80, 0.999)
ev["tracked"] = rng.binomial(1, p_track)  # POSITION_TRACKED_BIT analogue

p_pinch = np.where(ev["model"] == "ht_v2", 0.93, 0.88) - 0.05 * (ev["lighting"] == "low")
ev["pinch_correct"] = rng.binomial(1, p_pinch.clip(0.5, 0.99))
ev.to_csv(OUT + "ht_model_eval.csv", index=False)

print("\n== ht_model_eval.csv ==")
print("mean MPJPE by model:\n", ev.groupby("model")["mpjpe_mm"].mean().round(2))
print("mean MPJPE by model x lighting x motion:\n",
      ev.groupby(["model", "lighting", "motion"])["mpjpe_mm"].mean().round(1).unstack("motion"))
print("mean MPJPE by model x joint_group:\n",
      ev.groupby(["model", "joint_group"])["mpjpe_mm"].mean().round(2))
print("tracked rate by model:\n", ev.groupby("model")["tracked"].mean().round(3))
print("pinch_correct rate by model:\n", ev.groupby("model")["pinch_correct"].mean().round(3))

# ---------------------------------------------------------------- 3. releases
# v69 shipped Tuesday. Users say tracking feels worse. Regression? Did v70 fix it?
n_r = 9000
rl = pd.DataFrame({
    "session_id": [f"sess{i:06d}" for i in range(n_r)],
    "release": rng.choice(["v68", "v69", "v70"], size=n_r),
    "device": rng.choice(["quest3", "quest3s"], size=n_r, p=[0.55, 0.45]),
})
jit = {("v68", "quest3"): 2.10, ("v68", "quest3s"): 2.10,
       ("v69", "quest3"): 2.15, ("v69", "quest3s"): 3.00,
       ("v70", "quest3"): 2.10, ("v70", "quest3s"): 2.15}
loss = {("v68", "quest3"): 1.20, ("v68", "quest3s"): 1.20,
        ("v69", "quest3"): 1.25, ("v69", "quest3s"): 1.90,
        ("v70", "quest3"): 1.20, ("v70", "quest3s"): 1.25}
bat = {"v68": 18.0, "v69": 18.1, "v70": 18.9}
rl["jitter_mm"] = [max(0.2, rng.normal(jit[(r, d)], 0.5)) for r, d in zip(rl["release"], rl["device"])]
rl["jitter_mm"] = rl["jitter_mm"].round(2)
rl["tracking_loss_per_hr"] = [max(0.0, rng.normal(loss[(r, d)], 0.4)) for r, d in zip(rl["release"], rl["device"])]
rl["tracking_loss_per_hr"] = rl["tracking_loss_per_hr"].round(2)
rl["mtp_latency_ms"] = rng.normal(42, 4, n_r).clip(min=20).round(1)
rl["battery_drain_pct_hr"] = rng.normal(rl["release"].map(bat), 1.5).clip(min=10).round(1)
rl.to_csv(OUT + "release_perf.csv", index=False)

print("\n== release_perf.csv ==")
print("mean jitter by release x device:\n",
      rl.groupby(["release", "device"])["jitter_mm"].mean().round(2).unstack("device"))
print("mean tracking_loss by release x device:\n",
      rl.groupby(["release", "device"])["tracking_loss_per_hr"].mean().round(2).unstack("device"))
print("mean mtp_latency by release:\n", rl.groupby("release")["mtp_latency_ms"].mean().round(1))
print("mean battery drain by release:\n", rl.groupby("release")["battery_drain_pct_hr"].mean().round(2))
print("\ndone.")
