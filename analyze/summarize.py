#!/usr/bin/env python3
"""Aggregate run.sh results into one CSV row per (cell, role).

Within a repetition, Pods with the same role are averaged and their throughput is
summed; mean and stdev are then taken across repetitions.

Note: the mean of per-Pod P99s is not the P99 of the merged distribution.

Interference is relative to the same mode, config, workload and rate at n=1.

Excess interference removes the expected fair share: when n closed-loop Pods share
a GPU, each is expected to run n times slower, so `p50_excess` = 1.0 means pure
sharing and anything above it is overhead or contention. It is only meaningful for
homogeneous closed-loop cells, and left empty otherwise. `total_throughput_loss`
is the fraction of solo throughput lost in total across those Pods.

A repetition is excluded, and counted in `excluded_reps`, if any Pod failed to
report, started late, landed on a different GPU, or if the measured windows of the
Pods do not overlap enough.
Out-of-memory Pods are counted in `oom_pods`; they are results, not errors. So are
`starved_pods`, which ran but completed nothing inside the measured window.

Usage: summarize.py [results_dir] [-o summary.csv]
"""

import argparse
import csv
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

MIN_OVERLAP = 0.95
MAX_LATE_S = 0.0

FIELDS = [
    "mode", "config", "workload", "noisy", "rate", "n", "role", "reps", "excluded_reps",
    "oom_pods", "overloaded_pods", "starved_pods",
    "throughput_mean", "throughput_stdev", "p50_us_mean", "p50_us_stdev",
    "p99_us_mean", "p99_us_stdev", "p50_interference", "p99_interference",
    "throughput_ratio", "p50_excess", "total_throughput_loss",
]


def window_overlap(pods):
    """Shared window of all Pods, as a fraction of the shortest Pod window."""
    windows = [(p["window_start_unix"], p["window_end_unix"]) for p in pods]
    shortest = min(end - start for start, end in windows)
    shared = min(end for _, end in windows) - max(start for start, _ in windows)
    return max(shared, 0.0) / shortest if shortest > 0 else 0.0


def load_rep(rep_dir):
    """Return (meta, pods, exclusion reason or None)."""
    meta = json.loads((rep_dir / "meta.json").read_text())
    pods = []
    for path in sorted(rep_dir.glob("*.json")):
        if path.name != "meta.json":
            pods.append(json.loads(path.read_text()))

    if meta.get("tag", "").startswith("cap"):
        return meta, pods, "capacity run, see analyze/capacity.py"
    if len(pods) != meta["n"]:
        return meta, pods, f"{len(pods)} of {meta['n']} Pods reported"
    ok = [p for p in pods if p["status"] == "ok" and p["summary"]["count"]]
    if any(p.get("late_start_s", 0) > MAX_LATE_S for p in ok):
        return meta, pods, "late start"
    uuids = {p["environment"].get("gpu_uuid") for p in ok}
    if len(uuids - {None}) > 1:
        return meta, pods, "Pods landed on different GPUs"
    if len(ok) > 1 and window_overlap(ok) < MIN_OVERLAP:
        return meta, pods, "windows do not overlap"
    return meta, pods, None


def collect(results_dir):
    cells = defaultdict(lambda: {"reps": defaultdict(list), "excluded": 0, "oom": 0,
                                 "overloaded": 0, "starved": 0})
    for meta_path in sorted(results_dir.rglob("meta.json")):
        meta, pods, reason = load_rep(meta_path.parent)
        if meta.get("tag", "").startswith("cap"):
            continue
        workload = "-".join(filter(None, (meta.get("tag"), meta["workload"], meta.get("size"))))
        cell_key = (meta["mode"], meta["config"], workload, meta["noisy"], meta["rate"], meta["n"])
        roles = {p["role"] for p in pods} or {"unknown"}
        if reason:
            print(f"excluded {meta_path.parent}: {reason}", file=sys.stderr)
        for role in roles:
            cell = cells[cell_key + (role,)]
            role_pods = [p for p in pods if p["role"] == role]
            if reason:
                cell["excluded"] += 1
                continue
            cell["oom"] += sum(p["status"] == "oom" for p in role_pods)
            cell["overloaded"] += sum(bool(p.get("overloaded")) for p in role_pods)
            ok = [p["summary"] for p in role_pods if p["status"] == "ok" and p["summary"]["count"]]
            cell["starved"] += sum(p["status"] == "ok" and not p["summary"]["count"] for p in role_pods)
            if ok:
                cell["reps"][meta["rep"]] = ok
    return cells


def spread(values):
    return statistics.mean(values), statistics.stdev(values) if len(values) > 1 else 0.0


def ratio(numerator, denominator):
    return numerator / denominator if numerator is not None and denominator else None


def build_rows(cells):
    rows = {}
    for key, cell in cells.items():
        mode, config, workload, noisy, rate, n, role = key
        row = dict(zip(FIELDS, (mode, config, workload, noisy, rate, n, role)))
        row.update(reps=len(cell["reps"]), excluded_reps=cell["excluded"],
                   oom_pods=cell["oom"], overloaded_pods=cell["overloaded"],
                   starved_pods=cell["starved"])
        reps = list(cell["reps"].values())
        if reps:
            row["throughput_mean"], row["throughput_stdev"] = spread(
                [sum(s["throughput_per_s"] for s in pods) for pods in reps])
            row["p50_us_mean"], row["p50_us_stdev"] = spread(
                [statistics.mean(s["p50_us"] for s in pods) for pods in reps])
            row["p99_us_mean"], row["p99_us_stdev"] = spread(
                [statistics.mean(s["p99_us"] for s in pods) for pods in reps])
        rows[key] = row

    for key, row in rows.items():
        mode, config, workload, noisy, rate, n, role = key
        base = rows.get((mode, config, workload, "", rate, 1, "solo"))
        if base and role in ("solo", "peer", "victim"):
            row["p50_interference"] = ratio(row.get("p50_us_mean"), base.get("p50_us_mean"))
            row["p99_interference"] = ratio(row.get("p99_us_mean"), base.get("p99_us_mean"))
            row["throughput_ratio"] = ratio(row.get("throughput_mean"), base.get("throughput_mean"))
            if role == "peer" and not rate and row.get("p50_interference"):
                row["p50_excess"] = row["p50_interference"] / n
                row["total_throughput_loss"] = 1 - row["throughput_ratio"]
    return sorted(rows.values(), key=lambda r: tuple(str(r[f]) for f in FIELDS[:7]))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("results", nargs="?", default="results", type=Path)
    p.add_argument("-o", "--output", type=Path, help="default: stdout")
    args = p.parse_args()

    rows = build_rows(collect(args.results))
    out = args.output.open("w", newline="") if args.output else sys.stdout
    writer = csv.DictWriter(out, fieldnames=FIELDS)
    writer.writeheader()
    writer.writerows(rows)
    if args.output:
        out.close()


if __name__ == "__main__":
    main()
