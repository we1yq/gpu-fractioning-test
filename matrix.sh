#!/usr/bin/env bash
# Run the experiment matrix described in docs/experiment-design.md.
#
#   ./matrix.sh --list
#   ./matrix.sh --gpu-confirmed --image IMAGE --results results/v2 --reps 1
#
# Groups A and B differ only in the hard limit, so they alternate cell by cell and
# slow drift (temperature, time of day) cannot land on one group. Each repetition
# covers every cell before the next one starts. Group C (--groups C) needs the node
# prepared by k8s/run-legacy-group.sh.
#
# Safe to rerun with the same --results: finished repetitions are skipped, so an
# interrupted matrix resumes, and raising --reps adds repetitions. Run it in tmux.
#
# A failed cell does not stop the matrix. Failures go to <results>/failures.log and
# the exit code is nonzero; rerun to retry them.
set -uo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
IMAGE="" RESULTS="" REPS=1 DURATION=30 WARMUP=5 SLACK=20 COOLDOWN=10 CONFIRMED=0 LIST=0
GROUPS_TO_RUN="A B" PHASES="runtime capacity alloc"
REQUEST=10Gi NO_LIMIT=40Gi LIMIT=10Gi FRACTION=0.25
# Below what a victim can sustain next to long kernels (390-1070 requests/s in round
# one), so queueing reflects interference rather than an overloaded victim.
VICTIM_RATE=100
# Next to 67 ms kernels the victim only reaches about 49 requests/s.
LONG_KERNEL_VICTIM_RATE=30
PROBE="--workload gemm --size 2048"

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
    --groups) GROUPS_TO_RUN="$2"; shift 2 ;;
    --phases) PHASES="$2"; shift 2 ;;
    --gpu-confirmed) CONFIRMED=1; shift ;;
    --list) LIST=1; shift ;;
    *) die "unknown argument: $1" ;;
  esac
done

# One line per cell: the run.sh arguments that vary.
runtime_cells() {
  local n
  for n in 1 2 4; do echo "--workload gemm -n $n"; done
  echo "--workload smallkernel --rate $VICTIM_RATE -n 1"
  echo "--workload smallkernel --rate $VICTIM_RATE --noisy gemm -n 2"
  echo "--workload smallkernel --rate $LONG_KERNEL_VICTIM_RATE -n 1"
  echo "--workload smallkernel --rate $LONG_KERNEL_VICTIM_RATE --noisy gemm --noisy-size 8192 -n 2"
  for n in 1 2; do echo "--workload membw -n $n"; done
  for n in 1 2; do echo "--workload hostcopy -n $n"; done
  echo "--workload gemm --noisy hostcopy -n 2"
  # Three 13 MiB arrays per Pod: combined working set well above a 40 MB L2.
  for n in 1 2; do echo "--tag l2 --workload membw --size 13 -n $n"; done
}

capacity_cells() {
  echo "--tag cap-fair $PROBE -n 4 --hold-list 15000,15000,15000,15000 --stagger 5"
  # Bystander first: it must already be measuring when the eviction happens.
  # Order: bystander (1 GiB), greedy Pod (38 GiB), then the victim (2 GiB) that
  # runs into the wall.
  echo "--tag cap-hog $PROBE -n 3 --hold-list 1000,38000,2000 --stagger 6"
}

alloc_cells() {
  echo "--workload alloccycle --size 64 -n 1"
  echo "--workload alloccycle --size 1024 -n 1"
  echo "--workload alloccycle --size 64 -n 2"
  echo "--tag near-limit --workload alloccycle --size 64 -n 1 --hold-list 9216"
}

cells() {
  local phase
  for phase in $PHASES; do "${phase}_cells"; done
}

group_args() { # group
  case "$1" in
    A) echo "--mode fractioning --compute sm-sharing --gpu-memory $REQUEST --gpu-limit $NO_LIMIT" ;;
    B) echo "--mode fractioning --compute sm-sharing --gpu-memory $REQUEST --gpu-limit $LIMIT" ;;
    C) echo "--mode legacy --gpu-fraction $FRACTION" ;;
    *) die "unknown group: $1" ;;
  esac
}

for group in $GROUPS_TO_RUN; do group_args "$group" >/dev/null; done

# Alternate which group goes first, so neither always follows the other.
ordered_groups() { # cell_index
  if (( $1 % 2 == 0 )); then
    echo "$GROUPS_TO_RUN"
  else
    tr ' ' '\n' <<<"$GROUPS_TO_RUN" | awk '{ a[NR] = $0 } END { for (i = NR; i > 0; i--) print a[i] }' | tr '\n' ' '
  fi
}

if [[ "$LIST" == 1 ]]; then
  cells
  count=$(cells | wc -l | tr -d ' ')
  groups=$(wc -w <<<"$GROUPS_TO_RUN" | tr -d ' ')
  echo "$count cells x $groups groups x $REPS reps = $((count * groups * REPS)) runs"
  exit 0
fi

[[ -n "$IMAGE" ]] || die "--image is required"
[[ -n "$RESULTS" ]] || die "--results is required"
[[ "$CONFIRMED" == 1 ]] || die "pass --gpu-confirmed once the GPUs are ours"

mkdir -p "$RESULTS"
failed=0

for ((rep = 1; rep <= REPS; rep++)); do
  index=0
  while read -r -a cell; do
    for group in $(ordered_groups "$index"); do
      read -r -a group_flags <<<"$(group_args "$group")"
      if ! "$ROOT/run.sh" --gpu-confirmed --resume --only-rep "$rep" --reps "$REPS" \
          --image "$IMAGE" --duration "$DURATION" --warmup "$WARMUP" \
          --startup-slack "$SLACK" --cooldown "$COOLDOWN" --results "$RESULTS" \
          "${group_flags[@]}" "${cell[@]}" </dev/null; then
        failed=$((failed + 1))
        echo "$(date -u +%FT%TZ) rep=$rep group=$group ${cell[*]}" | tee -a "$RESULTS/failures.log" >&2
      fi
    done
    index=$((index + 1))
  done < <(cells)
done

echo "matrix finished: $failed failed cell(s)"
[[ "$failed" -eq 0 ]]
