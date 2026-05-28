#!/usr/bin/env python3

"""Diagnose a 'wavefront' (cliff) propagation speed in FiVo/FV snapshots.

This targets the visually striking sharp drop in v(r) that you see in plots.
Rather than using the cell center where |dv/dr| is largest (which can jitter),
we locate the *midpoint of the drop*:

  1) find an outer, strong negative dv/dr with a large v drop
  2) estimate v_in (inside) and v_out (outside) via local window averages
  3) define v_mid = (v_in + v_out)/2
  4) interpolate the radius where v crosses v_mid near the cliff

This produces a much more stable r_front(tau) and therefore a more meaningful
dr/dtau.

Usage:
  python3 tools/analyze_fivo_wavefront_speed.py snapshots/snapshots_ideal_diff_visc_phi

Outputs:
  snapshot_dir/wavefront_speed_summary.csv
"""

from __future__ import annotations

import argparse
import csv
import glob
import math
import os
import re
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple


TAU_RE = re.compile(r"snapshot_tau_(\d+\.\d+)\.csv$")


def parse_tau(path: str) -> float:
    m = TAU_RE.search(os.path.basename(path))
    if not m:
        raise ValueError(f"Could not parse tau from filename: {path}")
    return float(m.group(1))


def finite(x: float) -> bool:
    return math.isfinite(x)


def load_rv(path: str) -> Tuple[List[float], List[float]]:
    r: List[float] = []
    v: List[float] = []
    with open(path, newline="") as f:
        rd = csv.DictReader(f)
        for row in rd:
            if row.get("ok", "true").strip().lower() != "true":
                continue
            rr = float(row["r"])
            if rr < 0.0:
                continue
            r.append(rr)
            v.append(float(row["v"]))
    return r, v


@dataclass
class FrontPoint:
    tau: float
    r_front: float
    v_in: float
    v_out: float
    v_mid: float
    dvdr_at_cliff: float


def find_v_cliff_midpoint(
    r: Sequence[float],
    v: Sequence[float],
    *,
    min_r: float = 0.5,
    window: int = 6,
    min_drop: float = 0.3,
    strong_frac: float = 0.8,
) -> Tuple[float, float, float, float, float]:
    """Return (r_front, v_in, v_out, v_mid, dvdr_at_cliff).

    Returns NaNs if no suitable cliff is found.
    """
    n = len(r)
    if n < 2 * window + 3:
        nan = float("nan")
        return nan, nan, nan, nan, nan

    dvdr = [0.0] * n
    for i in range(1, n - 1):
        dr = r[i + 1] - r[i - 1]
        dvdr[i] = 0.0 if dr == 0.0 else (v[i + 1] - v[i - 1]) / dr

    candidates = [i for i in range(window, n - window - 1) if r[i] >= min_r]
    if not candidates:
        nan = float("nan")
        return nan, nan, nan, nan, nan

    scored: List[Tuple[float, float, int, float, float]] = []
    for i in candidates:
        if dvdr[i] >= 0.0:
            continue
        v_in = sum(v[i - window : i]) / window
        v_out = sum(v[i + 1 : i + 1 + window]) / window
        if (v_in - v_out) < min_drop:
            continue
        scored.append((r[i], -dvdr[i], i, v_in, v_out))

    if not scored:
        nan = float("nan")
        return nan, nan, nan, nan, nan

    steep_max = max(s[1] for s in scored)
    strong = [s for s in scored if s[1] >= strong_frac * steep_max]
    # Choose the *outermost* strong cliff.
    r0, _, i0, v_in, v_out = max(strong, key=lambda s: s[0])
    v_mid = 0.5 * (v_in + v_out)

    # Find crossing near i0 (above->below).
    jstart = max(1, i0 - 2 * window)
    jend = min(n - 1, i0 + 2 * window)
    r_cross = float("nan")
    for j in range(jstart + 1, jend + 1):
        if v[j - 1] >= v_mid and v[j] < v_mid:
            v1, v2 = v[j - 1], v[j]
            r1, r2 = r[j - 1], r[j]
            t = (v_mid - v1) / (v2 - v1) if v2 != v1 else 0.5
            r_cross = r1 + t * (r2 - r1)

    return r_cross, v_in, v_out, v_mid, dvdr[i0]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "snapshot_dir",
        nargs="?",
        default="snapshots/snapshots_ideal_diff_visc_phi",
        help="Directory with snapshot_tau_*.csv (excluding *_meta.csv)",
    )
    ap.add_argument("--min-r", type=float, default=0.5)
    ap.add_argument("--window", type=int, default=6)
    ap.add_argument("--min-drop", type=float, default=0.3)
    ap.add_argument("--strong-frac", type=float, default=0.8)
    args = ap.parse_args()

    files = [
        f
        for f in glob.glob(os.path.join(args.snapshot_dir, "snapshot_tau_*.csv"))
        if (not f.endswith("_meta.csv")) and TAU_RE.search(os.path.basename(f))
    ]
    files.sort(key=parse_tau)
    if not files:
        raise SystemExit(f"No snapshot files found in {args.snapshot_dir}")

    points: List[FrontPoint] = []
    for f in files:
        tau = parse_tau(f)
        r, v = load_rv(f)
        rf, v_in, v_out, v_mid, dvdr = find_v_cliff_midpoint(
            r,
            v,
            min_r=args.min_r,
            window=args.window,
            min_drop=args.min_drop,
            strong_frac=args.strong_frac,
        )
        points.append(
            FrontPoint(
                tau=tau,
                r_front=rf,
                v_in=v_in,
                v_out=v_out,
                v_mid=v_mid,
                dvdr_at_cliff=dvdr,
            )
        )

    # Compute speeds dr/dtau between consecutive snapshots.
    vmax = -float("inf")
    vmax_seg: Optional[Tuple[float, float, float, float]] = None
    for a, b in zip(points, points[1:]):
        if not (finite(a.r_front) and finite(b.r_front)):
            continue
        dt = b.tau - a.tau
        if dt <= 0.0:
            continue
        vfront = (b.r_front - a.r_front) / dt
        if vfront > vmax:
            vmax = vfront
            vmax_seg = (a.tau, b.tau, a.r_front, b.r_front)

    out_path = os.path.join(args.snapshot_dir, "wavefront_speed_summary.csv")
    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["tau", "r_front", "v_in", "v_out", "v_mid", "dvdr_at_cliff"])
        for p in points:
            w.writerow(
                [
                    f"{p.tau:.12g}",
                    "" if not finite(p.r_front) else f"{p.r_front:.12g}",
                    "" if not finite(p.v_in) else f"{p.v_in:.12g}",
                    "" if not finite(p.v_out) else f"{p.v_out:.12g}",
                    "" if not finite(p.v_mid) else f"{p.v_mid:.12g}",
                    "" if not finite(p.dvdr_at_cliff) else f"{p.dvdr_at_cliff:.12g}",
                ]
            )

    print(f"Wrote {out_path}")
    if vmax_seg is None:
        print("No front detected (with current thresholds).")
        return 0

    print(
        "Max wavefront speed dr/dtau = "
        f"{vmax:.6f} between tau={vmax_seg[0]:.3f}->{vmax_seg[1]:.3f} "
        f"(r={vmax_seg[2]:.3f}->{vmax_seg[3]:.3f})"
    )
    if vmax > 1.0 + 1e-12:
        print("WARNING: dr/dtau > 1 for this front definition")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
