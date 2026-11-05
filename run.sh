#!/usr/bin/env bash
# Run one cell of the experiment matrix: deploy Pods, wait, collect results.
#
#   ./run.sh --gpu-confirmed --mode fractioning --compute time-slicing \
#            --image IMAGE --workload gemm -n 2 --reps 5
#
# Interference (Pod 0 is the victim, the other n-1 Pods run --noisy):
#   ./run.sh ... --workload smallkernel --rate 200 --noisy gemm -n 4
#
# This script only creates its own Pods and one ConfigMap. It never changes
# GpuFractioningConfig, the device plugin, Helm releases or MIG partitions.
#
# Credentials come from k8s/setup-access.sh, so sudo is not needed. KUBECTL
# overrides the client. The RKE2 binary is preferred because it matches the
# server version.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
DEFAULT_KUBECTL=kubectl
[[ ! -x /var/lib/rancher/rke2/bin/kubectl ]] || DEFAULT_KUBECTL=/var/lib/rancher/rke2/bin/kubectl
read -r -a KUBE <<<"${KUBECTL:-$DEFAULT_KUBECTL}"

MODE="" COMPUTE="time-slicing" IMAGE="" WORKLOAD="" NOISY="" SIZE="" RATE=0
COUNT=1 REPS=5 DURATION=60 WARMUP=10 SLACK=120 COOLDOWN=10 TIMEOUT=600
GPU_MEMORY="4Gi" MIG_PROFILE="" CPU=2 QUEUE="default-queue" NAMESPACE="gpu-bench"
RESULTS="$ROOT/results" CONFIRMED=0 DRY_RUN=0

die() { echo "error: $*" >&2; exit 1; }

while [[ $# -gt 0 ]]; do
  case "$1" in
    --mode) MODE="$2"; shift 2 ;;            # fractioning | mps | mig
    --compute) COMPUTE="$2"; shift 2 ;;      # time-slicing | sm-sharing
    --image) IMAGE="$2"; shift 2 ;;
    --workload) WORKLOAD="$2"; shift 2 ;;
    --noisy) NOISY="$2"; shift 2 ;;
    --size) SIZE="$2"; shift 2 ;;
    --rate) RATE="$2"; shift 2 ;;            # open-loop rate of Pod 0 (or all, if no --noisy)
    -n) COUNT="$2"; shift 2 ;;
    --reps) REPS="$2"; shift 2 ;;
    --duration) DURATION="$2"; shift 2 ;;
    --warmup) WARMUP="$2"; shift 2 ;;
    --startup-slack) SLACK="$2"; shift 2 ;;  # seconds allowed for all Pods to get ready
    --cooldown) COOLDOWN="$2"; shift 2 ;;
    --timeout) TIMEOUT="$2"; shift 2 ;;
    --gpu-memory) GPU_MEMORY="$2"; shift 2 ;;
    --mig-profile) MIG_PROFILE="$2"; shift 2 ;;
    --cpu) CPU="$2"; shift 2 ;;
    --queue) QUEUE="$2"; shift 2 ;;
    --namespace) NAMESPACE="$2"; shift 2 ;;
    --results) RESULTS="$2"; shift 2 ;;
    --gpu-confirmed) CONFIRMED=1; shift ;;
    --dry-run) DRY_RUN=1; shift ;;
    *) die "unknown argument: $1" ;;
  esac
done

case "$MODE" in
  fractioning)
    [[ "$COMPUTE" == time-slicing || "$COMPUTE" == sm-sharing ]] || die "bad --compute: $COMPUTE"
    CONFIG="$COMPUTE-$GPU_MEMORY"
    ;;
  mps) CONFIG="default" ;;
  mig)
    [[ -n "$MIG_PROFILE" ]] || die "--mode mig needs --mig-profile"
    CONFIG="$MIG_PROFILE"
    ;;
  *) die "--mode must be fractioning, mps or mig" ;;
esac
[[ -n "$IMAGE" ]] || die "--image is required"
[[ -n "$WORKLOAD" ]] || die "--workload is required"
[[ -z "$NOISY" || "$COUNT" -ge 2 ]] || die "--noisy needs -n 2 or more"
if [[ "$CONFIRMED" != 1 && "$DRY_RUN" != 1 ]]; then
  die "the GPUs are shared: confirm colleagues have finished, then pass --gpu-confirmed"
fi

TEMPLATE="$ROOT/k8s/$MODE.yaml.tmpl"
RATE_TAG=""
[[ "$RATE" == 0 ]] || RATE_TAG="@$RATE"
CELL="$MODE-$CONFIG/$WORKLOAD$RATE_TAG${NOISY:++$NOISY}/n$COUNT"

render_pod() { # name run_id args_json
  sed -e "s|__COMPUTE_MODE__|$COMPUTE|" \
      -e "s|__NAME__|$1|; s|__RUN_ID__|$2|; s|__ARGS__|$3|" \
      -e "s|__IMAGE__|$IMAGE|; s|__NAMESPACE__|$NAMESPACE|; s|__QUEUE__|$QUEUE|" \
      -e "s|__GPU_MEMORY__|$GPU_MEMORY|g; s|__MIG_PROFILE__|$MIG_PROFILE|g" \
      -e "s|__CPU__|$CPU|g; s|__CONFIGMAP__|$CONFIGMAP|" \
      "$TEMPLATE"
}

pod_args() { # index start_at seed
  local workload="$WORKLOAD" role="solo" rate="$RATE"
  if [[ -n "$NOISY" ]]; then
    role="victim"
    if [[ "$1" -gt 0 ]]; then workload="$NOISY" role="noisy" rate=0; fi
  elif [[ "$COUNT" -gt 1 ]]; then
    role="peer"
  fi
  local args=(--workload "$workload" --role "$role" --seed "$3" --start-at "$2"
              --duration "$DURATION" --warmup "$WARMUP" --rate "$rate")
  [[ -z "$SIZE" || "$workload" != "$WORKLOAD" ]] || args+=(--size "$SIZE")
  printf '['
  local sep=""
  for a in "${args[@]}"; do printf '%s"%s"' "$sep" "$a"; sep=", "; done
  printf ']'
}

cleanup() {
  [[ -n "${ACTIVE_RUN:-}" ]] || return 0
  "${KUBE[@]}" delete pods -n "$NAMESPACE" -l "bench-run=$ACTIVE_RUN" --ignore-not-found --wait=false >/dev/null || true
}
trap cleanup EXIT

wait_for_pods() { # run_id -> returns nonzero on timeout
  # awk, because --warmup and --duration may be fractional.
  local deadline
  deadline="$(awk -v now="$(date +%s)" -v a="$SLACK" -v b="$WARMUP" -v c="$DURATION" -v d="$TIMEOUT" \
    'BEGIN { printf "%d", now + a + b + c + d + 1 }')"
  while [[ $(date +%s) -lt $deadline ]]; do
    local phases
    phases="$("${KUBE[@]}" get pods -n "$NAMESPACE" -l "bench-run=$1" -o 'jsonpath={.items[*].status.phase}')"
    local done_count=0 p
    for p in $phases; do
      [[ "$p" != Succeeded && "$p" != Failed ]] || done_count=$((done_count + 1))
    done
    [[ "$done_count" -lt "$COUNT" ]] || return 0
    sleep 5
  done
  return 1
}

collect_pod() { # name dir
  "${KUBE[@]}" logs -n "$NAMESPACE" "$1" >"$2/$1.log" 2>&1 || true
  sed -n 's/^BENCH_RESULT //p' "$2/$1.log" >"$2/$1.json"
  [[ -s "$2/$1.json" ]] || { rm -f "$2/$1.json"; echo "warning: no result from $1, see $2/$1.log" >&2; }
  "${KUBE[@]}" get pod -n "$NAMESPACE" "$1" \
    -o 'jsonpath={.status.containerStatuses[0].state.terminated.exitCode}' >"$2/$1.exit" 2>/dev/null || true
}

CONFIGMAP="gpu-bench-code"

if [[ "$DRY_RUN" == 1 ]]; then
  for ((i = 0; i < COUNT; i++)); do
    echo "---"
    render_pod "gpu-bench-dry-$i" dry "$(pod_args "$i" 0 "$i")"
  done
  exit 0
fi

if [[ "$MODE" == fractioning ]]; then
  ready="$("${KUBE[@]}" get gpufractioningconfigs -o 'jsonpath={.items[*].status.conditions[?(@.type=="Ready")].status}' || true)"
  [[ "$ready" == *True* ]] || die "no GpuFractioningConfig is Ready; restore it first (this script will not)"
fi

"${KUBE[@]}" create configmap "$CONFIGMAP" -n "$NAMESPACE" \
  --from-file="$ROOT/bench/bench.py" --from-file="$ROOT/bench/workloads.py" \
  --dry-run=client -o yaml | "${KUBE[@]}" apply -f - >/dev/null

for ((rep = 1; rep <= REPS; rep++)); do
  run_id="gpu-bench-$(date +%Y%m%d%H%M%S)-$rep"
  dir="$RESULTS/$CELL/rep$rep"
  [[ ! -e "$dir" ]] || die "$dir already exists; use another --results directory"
  mkdir -p "$dir"
  start_at=$(( $(date +%s) + SLACK ))

  cat >"$dir/meta.json" <<EOF
{"mode": "$MODE", "config": "$CONFIG", "workload": "$WORKLOAD", "noisy": "$NOISY", "rate": $RATE,
 "n": $COUNT, "rep": $rep, "run_id": "$run_id", "image": "$IMAGE", "start_at": $start_at}
EOF
  if [[ "$MODE" == fractioning ]]; then
    "${KUBE[@]}" get gpufractioningconfigs -o yaml >"$dir/gpufractioningconfig.yaml"
  fi

  ACTIVE_RUN="$run_id"
  for ((i = 0; i < COUNT; i++)); do
    # Seeds depend on rep and index only, so every config sees identical input.
    render_pod "$run_id-$i" "$run_id" "$(pod_args "$i" "$start_at" "$((rep * 1000 + i))")" \
      | "${KUBE[@]}" apply -f - >/dev/null
  done

  echo "[$CELL rep $rep/$REPS] $run_id: started, measuring at $start_at"
  wait_for_pods "$run_id" || echo "warning: $run_id timed out; collecting what finished" >&2
  for ((i = 0; i < COUNT; i++)); do collect_pod "$run_id-$i" "$dir"; done
  cleanup
  ACTIVE_RUN=""
  [[ "$rep" -eq "$REPS" ]] || sleep "$COOLDOWN"
done
