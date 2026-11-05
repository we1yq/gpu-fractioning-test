#!/usr/bin/env bash
# Run the Fractioning experiment matrix, one repetition across all cells at a time
# so that slow drift (temperature, time of day) does not land on a single config.
#
#   ./matrix.sh --list
#   ./matrix.sh --gpu-confirmed --image IMAGE --results results/fractioning-v1 --reps 1
#
# Safe to rerun with the same --results: finished repetitions are skipped, so an
# interrupted matrix resumes, and raising --reps adds repetitions. Run it in tmux.
#
# A failed cell does not stop the matrix. Failures go to <results>/failures.log and
# the exit code is nonzero; rerun to retry them.
set -uo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
IMAGE="" RESULTS="" REPS=1 DURATION=30 WARMUP=5 SLACK=20 COOLDOWN=10 CONFIRMED=0 LIST=0
COMPUTE_MODES="time-slicing sm-sharing"
WORKLOADS="gemm membw smallkernel hostcopy"
# Open-loop victim rate. About 12% of solo smallkernel saturation (43k/s), so a
# victim with a fair 1/4 share (about 10.8k/s) still has headroom at n=4. Queueing
# then reflects interference, not a victim that was overloaded by design.
VICTIM_RATE=5000

die() { echo "error: $*" >&2; exit 1; }

while [[ $# -gt 0 ]]; do
  case "$1" in
    --image) IMAGE="$2"; shift 2 ;;
    --results) RESULTS="$2"; shift 2 ;;
    --reps) REPS="$2"; shift 2 ;;
    --duration) DURATION="$2"; shift 2 ;;
    --warmup) WARMUP="$2"; shift 2 ;;
    --startup-slack) SLACK="$2"; shift 2 ;;
    --cooldown) COOLDOWN="$2"; shift 2 ;;
    --workloads) WORKLOADS="$2"; shift 2 ;;
    --compute) COMPUTE_MODES="$2"; shift 2 ;;
    --victim-rate) VICTIM_RATE="$2"; shift 2 ;;
    --gpu-confirmed) CONFIRMED=1; shift ;;
    --list) LIST=1; shift ;;
    *) die "unknown argument: $1" ;;
  esac
done

# One line per cell: the run.sh arguments that vary.
cells() {
  local w noisy n
  # Homogeneous: pure sharing cost at each concurrency.
  for w in $WORKLOADS; do
    for n in 1 2 4; do echo "--workload $w -n $n"; done
  done
  # Cross matrix: which resource does each pair contend on?
  for w in $WORKLOADS; do
    for noisy in $WORKLOADS; do
      [[ "$w" == "$noisy" ]] || echo "--workload $w --noisy $noisy -n 2"
    done
  done
  # Latency-sensitive victim at a fixed request rate.
  echo "--workload smallkernel --rate $VICTIM_RATE -n 1"
  for noisy in $WORKLOADS; do
    [[ "$noisy" != smallkernel ]] || continue
    for n in 2 4; do
      echo "--workload smallkernel --rate $VICTIM_RATE --noisy $noisy -n $n"
    done
  done
}

if [[ "$LIST" == 1 ]]; then
  for compute in $COMPUTE_MODES; do cells | sed "s/^/$compute: /"; done
  per_mode=$(cells | wc -l | tr -d ' ')
  modes=$(wc -w <<<"$COMPUTE_MODES" | tr -d ' ')
  echo "$per_mode cells x $modes modes x $REPS reps = $((per_mode * modes * REPS)) runs"
  exit 0
fi

[[ -n "$IMAGE" ]] || die "--image is required"
[[ -n "$RESULTS" ]] || die "--results is required"
[[ "$CONFIRMED" == 1 ]] || die "pass --gpu-confirmed once the GPUs are ours"

mkdir -p "$RESULTS"
failed=0

for ((rep = 1; rep <= REPS; rep++)); do
  for compute in $COMPUTE_MODES; do
    while read -r -a cell; do
      if ! "$ROOT/run.sh" --gpu-confirmed --resume --only-rep "$rep" --reps "$REPS" \
          --mode fractioning --compute "$compute" --image "$IMAGE" \
          --duration "$DURATION" --warmup "$WARMUP" --startup-slack "$SLACK" --cooldown "$COOLDOWN" \
          --results "$RESULTS" "${cell[@]}" </dev/null; then
        failed=$((failed + 1))
        echo "$(date -u +%FT%TZ) rep=$rep compute=$compute ${cell[*]}" | tee -a "$RESULTS/failures.log" >&2
      fi
    done < <(cells)
  done
done

echo "matrix finished: $failed failed cell(s)"
[[ "$failed" -eq 0 ]]
