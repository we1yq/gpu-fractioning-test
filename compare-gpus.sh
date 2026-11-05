#!/usr/bin/env bash
# Measure the same solo workloads on each physical GPU, to separate card differences
# from the effect of a memory limit.
#
#   ./compare-gpus.sh --gpu-confirmed --image IMAGE --results results/gpu-compare
#
# Workload Pods cannot name a GPU, so idle blocker Pods steer them: one blocker on
# each card, then removing one leaves that card as the only place a 10 GiB Pod fits.
# Blockers run sleep and do not touch the GPU. Results land under
# <results>/<gpu-uuid>/ and each repetition covers both cards.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
DEFAULT_KUBECTL=kubectl
[[ ! -x /var/lib/rancher/rke2/bin/kubectl ]] || DEFAULT_KUBECTL=/var/lib/rancher/rke2/bin/kubectl
read -r -a KUBE <<<"${KUBECTL:-$DEFAULT_KUBECTL}"

IMAGE="" RESULTS="" REPS=5 CONFIRMED=0 NAMESPACE=gpu-bench
WORKLOADS="gemm membw hostcopy smallkernel"
BLOCKER_REQUEST=35Gi

die() { echo "error: $*" >&2; exit 1; }

while [[ $# -gt 0 ]]; do
  case "$1" in
    --image) IMAGE="$2"; shift 2 ;;
    --results) RESULTS="$2"; shift 2 ;;
    --reps) REPS="$2"; shift 2 ;;
    --gpu-confirmed) CONFIRMED=1; shift ;;
    *) die "unknown argument: $1" ;;
  esac
done
[[ -n "$IMAGE" && -n "$RESULTS" ]] || die "--image and --results are required"
[[ "$CONFIRMED" == 1 ]] || die "pass --gpu-confirmed once the GPUs are ours"

start_blocker() { # name
  "$ROOT/run.sh" --dry-run --mode fractioning --compute sm-sharing \
      --gpu-memory "$BLOCKER_REQUEST" --gpu-limit "$BLOCKER_REQUEST" \
      --image "$IMAGE" --workload gemm \
    | sed -e "s/gpu-bench-dry-0/$1/" -e 's/bench-run: dry/bench-run: gpu-compare/' \
          -e 's|command: .*|command: ["sleep", "7200"]|' -e '/ args: /d' \
    | "${KUBE[@]}" apply -f - >/dev/null
}

# KAI records the card it picked on the Pod.
assigned_gpu() { # name
  "${KUBE[@]}" get pod "$1" -n "$NAMESPACE" -o json | python3 -c '
import json, sys
annotations = json.load(sys.stdin)["metadata"].get("annotations", {})
print(annotations.get("nvidia.com/container.bench.gpus.devices", ""))'
}

remove_blockers() {
  "${KUBE[@]}" delete pods -n "$NAMESPACE" -l bench-run=gpu-compare --ignore-not-found --wait=true >/dev/null
}
trap remove_blockers EXIT

for ((rep = 1; rep <= REPS; rep++)); do
  for free in 0 1; do
    keep=$((1 - free))
    start_blocker "gpu-compare-$free"
    start_blocker "gpu-compare-$keep"
    "${KUBE[@]}" wait -n "$NAMESPACE" --for=condition=Ready --timeout=120s \
      "pod/gpu-compare-$free" "pod/gpu-compare-$keep" >/dev/null

    free_gpu="$(assigned_gpu "gpu-compare-$free")"
    keep_gpu="$(assigned_gpu "gpu-compare-$keep")"
    [[ -n "$free_gpu" && "$free_gpu" != "$keep_gpu" ]] \
      || die "blockers did not land on separate cards: '$free_gpu' '$keep_gpu'"

    "${KUBE[@]}" delete pod "gpu-compare-$free" -n "$NAMESPACE" --wait=true >/dev/null
    echo "[rep $rep] measuring on $free_gpu"
    for workload in $WORKLOADS; do
      "$ROOT/run.sh" --gpu-confirmed --resume --only-rep "$rep" --reps "$REPS" \
        --mode fractioning --compute sm-sharing --gpu-memory 10Gi --gpu-limit 10Gi \
        --image "$IMAGE" --workload "$workload" --duration 20 --warmup 5 \
        --startup-slack 15 --cooldown 3 --results "$RESULTS/$free_gpu"
    done
    remove_blockers
  done
done
