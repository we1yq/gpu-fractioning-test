#!/usr/bin/env python3
"""Run one synthetic workload and report its latency distribution as JSON.

Run once per Pod. Concurrent Pods pass the same --start-at so their measured
windows overlap; the result records the actual window so analysis can verify it.

The result is written to --out, or otherwise printed to stdout as a single line
prefixed with RESULT_MARKER, which lets run.sh recover it from `kubectl logs`.

Exit codes: 0 ok, 2 out of memory (matches the smoke-test convention).
"""

import argparse
import json
import math
import os
import platform
import random
import socket
import subprocess
import sys
import time

import torch

from workloads import WORKLOADS

RESULT_MARKER = "BENCH_RESULT "
EXIT_OOM = 2
SCHEMA_VERSION = 2

# Log-spaced histogram: 20 buckets per decade from 1 us up to about 1000 s.
HIST_BUCKETS_PER_DECADE = 20
HIST_MIN_US = 1.0
HIST_DECADES = 9


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--workload", required=True, choices=sorted(WORKLOADS))
    p.add_argument("--size", type=int, help="workload-specific size, see workloads.py")
    p.add_argument("--duration", type=float, default=60.0, help="measured seconds")
    p.add_argument("--warmup", type=float, default=10.0, help="discarded seconds before measuring")
    p.add_argument("--rate", type=float, default=0.0,
                   help="open-loop Poisson arrivals per second; 0 = closed loop (saturating)")
    p.add_argument("--start-at", type=float, help="unix time at which warmup begins")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda", help="cuda or cpu (cpu is for smoke tests only)")
    p.add_argument("--cpu-threads", type=int, default=2,
                   help="host threads; match the Pod CPU limit to avoid CFS throttling")
    p.add_argument("--numa-node", type=int,
                   help="pin to this NUMA node, ideally the one the GPU is attached to")
    p.add_argument("--role", default="solo", help="free-form label, e.g. victim or noisy")
    p.add_argument("--out", help="result path; default is a marker line on stdout")
    p.add_argument("--raw", action="store_true", help="also store every latency sample")
    return p.parse_args(argv)


def percentile(sorted_values, q):
    """Nearest-rank percentile; q is in [0, 100]."""
    if not sorted_values:
        return None
    rank = math.ceil(q / 100.0 * len(sorted_values))
    return sorted_values[max(rank, 1) - 1]


def histogram(latencies_us):
    """Sparse log-spaced histogram, kept small enough to go through pod logs."""
    counts = {}
    top = HIST_DECADES * HIST_BUCKETS_PER_DECADE - 1
    for value in latencies_us:
        ratio = max(value, HIST_MIN_US) / HIST_MIN_US
        bucket = min(int(math.log10(ratio) * HIST_BUCKETS_PER_DECADE), top)
        counts[bucket] = counts.get(bucket, 0) + 1
    return {
        "min_us": HIST_MIN_US,
        "buckets_per_decade": HIST_BUCKETS_PER_DECADE,
        "counts": {str(k): counts[k] for k in sorted(counts)},
    }


def summarize(latencies_us, window_s):
    ordered = sorted(latencies_us)
    n = len(ordered)
    mean = sum(ordered) / n if n else None
    stdev = None
    if n > 1:
        stdev = math.sqrt(sum((x - mean) ** 2 for x in ordered) / (n - 1))
    return {
        "count": n,
        "throughput_per_s": n / window_s if window_s > 0 else None,
        "mean_us": mean,
        "stdev_us": stdev,
        "min_us": ordered[0] if n else None,
        "p50_us": percentile(ordered, 50),
        "p90_us": percentile(ordered, 90),
        "p99_us": percentile(ordered, 99),
        "p999_us": percentile(ordered, 99.9),
        "max_us": ordered[-1] if n else None,
    }


def make_sync(device):
    """Timing is meaningless without this: CUDA kernels run asynchronously."""
    if device.startswith("cuda"):
        return torch.cuda.synchronize
    return lambda: None


def wait_until(deadline):
    """Sleep until close to the deadline, then spin so open-loop arrivals stay accurate."""
    while True:
        remaining = deadline - time.perf_counter()
        if remaining <= 0:
            return
        if remaining > 0.002:
            time.sleep(remaining - 0.001)


def run_closed_loop(workload, sync, warmup_end, measure_end):
    latencies_us = []
    window_start = None
    while True:
        begin = time.perf_counter()
        if begin >= measure_end:
            break
        workload.step()
        sync()
        end = time.perf_counter()
        if begin >= warmup_end:
            window_start = window_start or begin
            latencies_us.append((end - begin) * 1e6)
    return latencies_us, window_start


def run_open_loop(workload, sync, warmup_end, measure_end, rate, seed):
    """Latency runs from the *scheduled* arrival, so queueing behind a stall counts.

    Measuring from the actual start time would hide those delays (coordinated
    omission), which is exactly what a victim's P99 is supposed to expose.

    If the GPU cannot keep up, the backlog is abandoned at measure_end so the Pod
    still finishes, and the run is reported as overloaded.
    """
    arrivals = random.Random(seed)
    latencies_us = []
    window_start = None
    overloaded = False
    scheduled = time.perf_counter()
    while True:
        scheduled += arrivals.expovariate(rate)
        if scheduled >= measure_end:
            break
        wait_until(scheduled)
        if time.perf_counter() >= measure_end:
            overloaded = True
            break
        workload.step()
        sync()
        end = time.perf_counter()
        if scheduled >= warmup_end:
            window_start = window_start or scheduled
            latencies_us.append((end - scheduled) * 1e6)
    return latencies_us, window_start, overloaded


def parse_cpulist(text):
    """Expand a kernel cpulist such as '0,2,4-6' into a set of CPU numbers."""
    cpus = set()
    for part in text.strip().split(","):
        if part:
            first, _, last = part.partition("-")
            cpus.update(range(int(first), int(last or first) + 1))
    return cpus


def pin_to_numa_node(node):
    """Best effort. Returns the CPUs in use, or None if pinning was not possible."""
    if not hasattr(os, "sched_setaffinity"):
        return None
    try:
        with open(f"/sys/devices/system/node/node{node}/cpulist") as f:
            wanted = parse_cpulist(f.read())
        allowed = wanted & os.sched_getaffinity(0)
        if allowed:
            os.sched_setaffinity(0, allowed)
            return sorted(allowed)
    except OSError as e:
        print(f"NUMA pinning failed: {e}", file=sys.stderr)
    return None


def gpu_uuid():
    """Identifies the physical GPU, so a run that spans two GPUs can be rejected."""
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=uuid", "--format=csv,noheader"],
                             capture_output=True, text=True, timeout=30, check=True)
        return out.stdout.split()[0]
    except (OSError, IndexError, subprocess.SubprocessError) as e:
        print(f"could not read GPU UUID: {e}", file=sys.stderr)
        return None


def environment(device):
    info = {
        "hostname": socket.gethostname(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "device": device,
        "env": {k: v for k, v in os.environ.items()
                if k.startswith("CUDA_") or k.startswith("BENCH_")},
    }
    if device.startswith("cuda"):
        free, total = torch.cuda.mem_get_info()
        info["gpu_name"] = torch.cuda.get_device_name()
        info["gpu_uuid"] = gpu_uuid()
        info["gpu_mem_visible_bytes"] = total
        info["gpu_mem_free_at_start_bytes"] = free
    return info


def base_result(args, size):
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "ok",
        "workload": args.workload,
        "size": size,
        "role": args.role,
        "mode": "open_loop" if args.rate > 0 else "closed_loop",
        "rate_per_s": args.rate,
        "seed": args.seed,
        "duration_s": args.duration,
        "warmup_s": args.warmup,
        "requested_start_at": args.start_at,
        "environment": environment(args.device),
    }


def emit(result, out_path):
    payload = json.dumps(result, separators=(",", ":"))
    if out_path:
        with open(out_path, "w") as f:
            f.write(payload + "\n")
    else:
        print(RESULT_MARKER + payload, flush=True)


def measure(args, size, result):
    sync = make_sync(args.device)
    workload = WORKLOADS[args.workload](size, args.device, args.seed)

    # One step outside the window, so lazy initialization is not counted as
    # warmup that overruns a concurrent peer's start.
    workload.step()
    sync()

    wait_s = args.start_at - time.time() if args.start_at is not None else 0.0
    if wait_s > 0:
        time.sleep(wait_s)
    result["late_start_s"] = max(-wait_s, 0.0)

    # Time.time() is shared between Pods on a node; perf_counter() is not.
    wall_offset = time.time() - time.perf_counter()
    warmup_end = time.perf_counter() + args.warmup
    measure_end = warmup_end + args.duration

    if args.rate > 0:
        latencies_us, window_start, result["overloaded"] = run_open_loop(
            workload, sync, warmup_end, measure_end, args.rate, args.seed)
    else:
        latencies_us, window_start = run_closed_loop(workload, sync, warmup_end, measure_end)
    finished = time.perf_counter()

    # Closed-loop steps overrun measure_end, so use the real elapsed window.
    window_s = finished - window_start if window_start else 0.0
    result["window_start_unix"] = (window_start + wall_offset) if window_start else None
    result["window_end_unix"] = finished + wall_offset
    result["summary"] = summarize(latencies_us, window_s)
    result["histogram"] = histogram(latencies_us)
    if args.raw:
        result["latencies_us"] = latencies_us


def main(argv=None):
    args = parse_args(argv)
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        print("CUDA requested but not available", file=sys.stderr)
        return 1
    if args.rate < 0:
        print("--rate must be >= 0", file=sys.stderr)
        return 1

    # Before any allocation, so host memory lands on the pinned node.
    pinned_cpus = pin_to_numa_node(args.numa_node) if args.numa_node is not None else None
    torch.set_num_threads(args.cpu_threads)

    size = args.size or WORKLOADS[args.workload].default_size
    result = base_result(args, size)
    result["numa_node"] = args.numa_node
    result["numa_pinned"] = pinned_cpus is not None
    result["cpu_threads"] = args.cpu_threads
    exit_code = 0
    try:
        measure(args, size, result)
    except torch.cuda.OutOfMemoryError as e:
        result["status"] = "oom"
        result["error"] = str(e)
        exit_code = EXIT_OOM

    if args.device.startswith("cuda"):
        result["gpu_max_allocated_bytes"] = torch.cuda.max_memory_allocated()
        result["gpu_max_reserved_bytes"] = torch.cuda.max_memory_reserved()
    emit(result, args.out)
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
