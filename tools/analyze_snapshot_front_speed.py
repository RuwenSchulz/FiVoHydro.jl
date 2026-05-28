#!/usr/bin/env python3

"""Analyze 1D radial snapshot CSVs for (a) max fluid speed and (b) an apparent
"front speed" of the pressure profile.

This is meant as a quick causality sanity-check:
  - Local fluid speed should satisfy |v| <= 1.
  - Characteristic/signal speed (roughly sound) should satisfy |lambda| <= 1.
  - A naive front tracker based on a threshold can *appear* > 1 due to
    discretization + changing normalization; interpret with care.

Usage:
  python3 tools/analyze_snapshot_front_speed.py snapshots/snapshots_ideal_fluidum \
    --fracs 0.1 0.01 0.001 --cs 0.57735026919

Outputs a CSV summary next to the snapshot directory.
"""

from __future__ import annotations

import argparse
import csv
import glob
import math
import os
import re
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


TAU_RE = re.compile(r"snapshot_tau_(\d+\.\d+)\.csv$")


def parse_tau(path: str) -> float:
    m = TAU_RE.search(os.path.basename(path))
    if not m:
        raise ValueError(f"Could not parse tau from filename: {path}")
    return float(m.group(1))


def clamp(x: float, lo: float, hi: float) -> float:
    return lo if x < lo else hi if x > hi else x


def char_speed_outward(v: float, cs: float) -> float:
    """Relativistic addition: lambda+ = (v + cs)/(1 + v*cs)."""
    denom = 1.0 + v * cs
    if denom == 0.0:
        return math.copysign(float("inf"), v + cs)
    return (v + cs) / denom


@dataclass
class SnapshotSummary:
    tau: float
    r_at_max_abs_v: float
    max_abs_v: float
    max_lambda_plus: float
    p0: float
    # r where P/P0 == frac (interpolated); NaN if not found
    rfront: Dict[float, float]


def read_snapshot(path: str, fracs: Sequence[float], cs: float) -> SnapshotSummary:
    tau = parse_tau(path)

    # We restrict to r >= 0 and ok==true.
    points: List[Tuple[float, float]] = []  # (r, P)

    p0: Optional[float] = None
    max_abs_v = -1.0
    r_at_max_abs_v = float("nan")
    max_lambda_plus = -1.0

    with open(path, newline="") as f:
        rdr = csv.DictReader(f)
        for row in rdr:
            ok = row.get("ok", "true").strip().lower() == "true"
            if not ok:
                continue
            r = float(row["r"])
            if r < 0.0:
                continue

            P = float(row["P"])
            v = float(row["v"])

            if p0 is None:
                p0 = P

            av = abs(v)
            if av > max_abs_v:
                max_abs_v = av
                r_at_max_abs_v = r

            lam = char_speed_outward(v, cs)
            if lam > max_lambda_plus:
                max_lambda_plus = lam

            points.append((r, P))

    if p0 is None:
        raise RuntimeError(f"No usable rows found in {path}")

    # Ensure sorted by r.
    points.sort(key=lambda t: t[0])

    # Interpolated radius where P/P0 crosses a given fraction.
    rfront: Dict[float, float] = {}
    for frac in fracs:
        target = p0 * frac
        rf = float("nan")
        # Find the first index where P drops below target.
        prev_r, prev_p = points[0]
        if prev_p < target:
            rf = prev_r
        else:
            for r, p in points[1:]:
                if p <= target:
                    # Linear interpolation between (prev_r, prev_p) and (r, p).
                    if p == prev_p:
                        rf = r
                    else:
                        t = (target - prev_p) / (p - prev_p)
                        rf = prev_r + t * (r - prev_r)
                    break
                prev_r, prev_p = r, p
        rfront[frac] = rf

    return SnapshotSummary(
        tau=tau,
        r_at_max_abs_v=r_at_max_abs_v,
        max_abs_v=max_abs_v,
        max_lambda_plus=max_lambda_plus,
        p0=p0,
        rfront=rfront,
    )


def finite(x: float) -> bool:
    return math.isfinite(x)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "snapshot_dir",
        help="Directory containing snapshot_tau_XX.XXX.csv files",
    )
    ap.add_argument(
        "--fracs",
        nargs="+",
        type=float,
        default=[0.1, 0.01, 0.001],
        help="Fractions of central pressure P0 used to define the front location",
    )
    ap.add_argument(
        "--cs",
        type=float,
        default=1.0 / math.sqrt(3.0),
        help="Assumed sound speed (natural units c=1). Default: 1/sqrt(3).",
    )
    args = ap.parse_args()

    fracs = [f for f in args.fracs if f > 0.0 and f < 1.0]
    if not fracs:
        raise SystemExit("Need at least one 0<frac<1")

    cs = clamp(args.cs, 0.0, 1.0)

    files = glob.glob(os.path.join(args.snapshot_dir, "snapshot_tau_*.csv"))
    files = [f for f in files if TAU_RE.search(os.path.basename(f))]
    files.sort(key=parse_tau)
    if not files:
        raise SystemExit(f"No snapshot files found in {args.snapshot_dir}")

    summaries = [read_snapshot(f, fracs=fracs, cs=cs) for f in files]

    # Report global maxima.
    g_max_v = max(s.max_abs_v for s in summaries)
    s_max_v = max(summaries, key=lambda s: s.max_abs_v)
    g_max_lam = max(s.max_lambda_plus for s in summaries)
    s_max_lam = max(summaries, key=lambda s: s.max_lambda_plus)

    print(f"Assumed cs = {cs:.6f}")
    print(
        f"Global max |v| = {g_max_v:.6f} at tau={s_max_v.tau:.3f}, r={s_max_v.r_at_max_abs_v:.3f}"
    )
    print(
        f"Global max lambda+ = {g_max_lam:.6f} at tau={s_max_lam.tau:.3f} (should be <= 1)"
    )
    if g_max_v > 1.0 + 1e-12:
        print("WARNING: Found |v| > 1 (acausal fluid speed)")
    if g_max_lam > 1.0 + 1e-12:
        print("WARNING: Found lambda+ > 1 (acausal characteristic speed for chosen cs)")

    # Compute apparent front speeds.
    print("\nFront speeds (using interpolated r where P/P0 = frac):")
    for frac in fracs:
        vmax = -float("inf")
        argmax: Optional[Tuple[float, float]] = None
        for i in range(1, len(summaries)):
            a, b = summaries[i - 1], summaries[i]
            ra, rb = a.rfront[frac], b.rfront[frac]
            if not (finite(ra) and finite(rb)):
                continue
            dt = b.tau - a.tau
            if dt <= 0.0:
                continue
            vfront = (rb - ra) / dt
            if vfront > vmax:
                vmax = vfront
                argmax = (a.tau, b.tau)
        if argmax is None:
            print(f"  frac={frac:g}: (no crossings)")
        else:
            print(f"  frac={frac:g}: max dr/dtau = {vmax:.6f} between tau={argmax[0]:.3f}->{argmax[1]:.3f}")

    # Write summary CSV
    out_path = os.path.join(args.snapshot_dir, "snapshot_speed_summary.csv")
    fieldnames = [
        "tau",
        "max_abs_v",
        "r_at_max_abs_v",
        "max_lambda_plus",
        "p0",
    ] + [f"rfront_{f:g}" for f in fracs]

    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for s in summaries:
            row = {
                "tau": f"{s.tau:.12g}",
                "max_abs_v": f"{s.max_abs_v:.12g}",
                "r_at_max_abs_v": f"{s.r_at_max_abs_v:.12g}",
                "max_lambda_plus": f"{s.max_lambda_plus:.12g}",
                "p0": f"{s.p0:.12g}",
            }
            for frac in fracs:
                rf = s.rfront[frac]
                row[f"rfront_{frac:g}"] = "" if not finite(rf) else f"{rf:.12g}"
            w.writerow(row)

    print(f"\nWrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
