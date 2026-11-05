#!/usr/bin/env bash
# Run group C (traditional MPS, no memory enforcement) and put the cluster back.
# Start this on the server inside tmux. It asks for the sudo password once.
#
#   ./k8s/run-legacy-group.sh --image IMAGE --results results/v2 --reps 1
#   ./k8s/run-legacy-group.sh --image IMAGE --results results/c-smoke --smoke
#
# What it changes, and always restores on exit, including after a failure:
#   1. Deletes GpuFractioningConfig and waits for its MPS to exit.
#   2. Switches KAI to NonMemoryEnforced. kai-values.yaml is only read, never changed;
#      the switch is two --set overrides on top of it.
#   3. Lets the gpu-bench namespace use hostPath, needed to reach the node MPS.
#   4. Starts a node-level MPS with its pipe directory at /tmp/nvidia-mps.
#
# Make sure nobody else needs the GPUs before running it: the first step stops MPS for
# every Fractioning workload on this node.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
KUBECTL_BIN=/var/lib/rancher/rke2/bin/kubectl
ADMIN_CONFIG=/etc/rancher/rke2/rke2.yaml
HELM="$HOME/tools/helm"
KAI_VALUES="$HOME/gpu-fractioning/kai-values.yaml"
KAI_CHART=oci://ghcr.io/kai-scheduler/kai-scheduler/kai-scheduler
KAI_VERSION=v0.18.7
PIPE_DIR=/tmp/nvidia-mps
LOG_DIR=/tmp/nvidia-mps-log
MPS_ENV=(env "CUDA_MPS_PIPE_DIRECTORY=$PIPE_DIR" "CUDA_MPS_LOG_DIRECTORY=$LOG_DIR")

IMAGE="" RESULTS="" REPS=1 SMOKE=0
die() { echo "error: $*" >&2; exit 1; }
step() { echo; echo "=== $(date -u +%T) $*"; }

while [[ $# -gt 0 ]]; do
  case "$1" in
    --image) IMAGE="$2"; shift 2 ;;
    --results) RESULTS="$2"; shift 2 ;;
    --reps) REPS="$2"; shift 2 ;;
    --smoke) SMOKE=1; shift ;;
    *) die "unknown argument: $1" ;;
  esac
done
[[ -n "$IMAGE" && -n "$RESULTS" ]] || die "--image and --results are required"
[[ -n "${TMUX:-}" ]] || die "run this inside tmux, an SSH drop must not stop the restore"
[[ -x "$HELM" && -r "$KAI_VALUES" ]] || die "missing $HELM or $KAI_VALUES"

admin() { sudo "$KUBECTL_BIN" --kubeconfig "$ADMIN_CONFIG" "$@"; }

kai_upgrade() { # extra helm arguments
  sudo "$HELM" upgrade kai-scheduler "$KAI_CHART" --version "$KAI_VERSION" \
    --namespace kai-scheduler --kubeconfig "$ADMIN_CONFIG" \
    -f "$KAI_VALUES" --wait --timeout 5m "$@"
}

wait_for_no_mps() {
  local i
  for i in $(seq 1 60); do
    pgrep -f '^nvidia-cuda-mps-server' >/dev/null || pgrep -x mpsd >/dev/null || return 0
    sleep 3
  done
  return 1
}

fractioning_ready() {
  [[ "$(admin get gpufractioningconfigs -o 'jsonpath={.items[*].status.conditions[?(@.type=="Ready")].status}' 2>/dev/null)" == *True* ]]
}

CHANGED=0 KEEPALIVE=""

restore() {
  local status=$?
  trap - EXIT INT TERM HUP
  [[ -z "$KEEPALIVE" ]] || kill "$KEEPALIVE" 2>/dev/null || true
  if [[ "$CHANGED" == 0 ]]; then exit "$status"; fi
  step "restoring the cluster"
  local ok=1
  echo quit | sudo "${MPS_ENV[@]}" nvidia-cuda-mps-control 2>/dev/null || true
  wait_for_no_mps || { echo "warning: node MPS is still running" >&2; ok=0; }
  admin label namespace gpu-bench pod-security.kubernetes.io/enforce=baseline --overwrite || ok=0
  kai_upgrade || ok=0
  local i
  for i in $(seq 1 40); do fractioning_ready && break; sleep 5; done
  if fractioning_ready; then
    echo "restored: GpuFractioningConfig is Ready"
  else
    ok=0
  fi
  if [[ "$ok" == 0 ]]; then
    echo "RESTORE INCOMPLETE. Check the steps above, then run:" >&2
    echo "  sudo $HELM upgrade kai-scheduler $KAI_CHART --version $KAI_VERSION --namespace kai-scheduler --kubeconfig $ADMIN_CONFIG -f $KAI_VALUES --wait --timeout 5m" >&2
    echo "  $KUBECTL_BIN --kubeconfig $ADMIN_CONFIG label namespace gpu-bench pod-security.kubernetes.io/enforce=baseline --overwrite  (with sudo)" >&2
    exit 1
  fi
  exit "$status"
}
trap restore EXIT
trap 'exit 130' INT TERM HUP

step "checks"
sudo -v
# sudo credentials expire after about 15 minutes; a matrix runs for hours.
( while true; do sudo -n true; sleep 60; done ) &
KEEPALIVE=$!

[[ -z "$(admin get pods -n gpu-bench -o name 2>/dev/null)" ]] || die "gpu-bench still has Pods"
others="$(nvidia-smi --query-compute-apps=process_name --format=csv,noheader | grep -v nvidia-cuda-mps-server || true)"
[[ -z "$others" ]] || die "GPU processes are running: $others"
fractioning_ready || die "GpuFractioningConfig is not Ready, nothing to restore to"

step "1/4 deleting GpuFractioningConfig"
CHANGED=1
admin delete gpufractioningconfig default --wait=true --timeout=180s
wait_for_no_mps || die "MPS did not exit"

step "2/4 switching KAI to NonMemoryEnforced"
kai_upgrade --set global.nvFractions.set=false --set global.gpuSharingMode=NonMemoryEnforced

step "3/4 allowing hostPath in gpu-bench"
admin label namespace gpu-bench pod-security.kubernetes.io/enforce=privileged --overwrite

step "4/4 starting the node MPS"
sudo mkdir -p "$PIPE_DIR" "$LOG_DIR"
sudo "${MPS_ENV[@]}" nvidia-cuda-mps-control -d
sleep 2

step "running group C"
cd "$ROOT"
if [[ "$SMOKE" == 1 ]]; then
  ./run.sh --gpu-confirmed --mode legacy --image "$IMAGE" --workload gemm -n 2 \
    --duration 20 --warmup 5 --startup-slack 45 --reps 1 --results "$RESULTS"
else
  ./matrix.sh --gpu-confirmed --groups C --image "$IMAGE" --results "$RESULTS" --reps "$REPS"
fi
