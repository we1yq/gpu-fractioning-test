#!/usr/bin/env python3
"""Report capacity experiments Pod by Pod: who got memory, who failed, who was evicted.

These runs are outcomes rather than distributions, so nothing is averaged away.

Usage: capacity.py [results_dir]
"""

import json
import sys
from pathlib import Path


def pod_row(exit_path):
    exit_code = exit_path.read_text().strip() or "?"
    json_path = exit_path.with_suffix(".json")
    if not json_path.exists():
        return {"exit": exit_code, "status": "no result (killed?)"}
    pod = json.loads(json_path.read_text())
    summary = pod.get("summary") or {}
    return {
        "exit": exit_code,
        "status": pod["status"],
        "target_mib": pod.get("hold_target_mib"),
        "held_mib": pod.get("held_mib"),
        "fell_short": pod.get("hold_fell_short"),
        "gpu": (pod["environment"].get("gpu_uuid") or "?")[:12],
        "p50_ms": round(summary["p50_us"] / 1000, 2) if summary.get("p50_us") else None,
        "p99_ms": round(summary["p99_us"] / 1000, 2) if summary.get("p99_us") else None,
        "error": (pod.get("error") or "")[:90],
    }


def main():
    results = Path(sys.argv[1] if len(sys.argv) > 1 else "results")
    for meta_path in sorted(results.rglob("meta.json")):
        meta = json.loads(meta_path.read_text())
        if not meta.get("tag", "").startswith("cap"):
            continue
        print(f"{meta['config']} {meta['tag']} rep{meta['rep']}")
        for exit_path in sorted(meta_path.parent.glob("gpu-bench-*.exit")):
            index = exit_path.stem.rsplit("-", 1)[1]
            print(f"  pod{index}: {pod_row(exit_path)}")


if __name__ == "__main__":
    main()
