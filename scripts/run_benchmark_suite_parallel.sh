#!/usr/bin/env bash
# Parallel heap benchmark suite: runs the 3 seeds of each objective set concurrently
# (~2.6 GB VRAM / ~55% GPU per job on the RTX 4090, so 3 fit comfortably).
# Usage: run_benchmark_suite_parallel.sh [existing_tuned_seed1_pid]
# Sends an ntfy notification after each finished set (and on any failure).
set -u
cd "$(dirname "$0")/.."
# /opt/openrobots (python3.8) on PYTHONPATH/LD_LIBRARY_PATH shadows the venv's pinocchio
unset PYTHONPATH LD_LIBRARY_PATH
PY=.venv/bin/python
NTFY_URL="https://ntfy.fangnan.me/work_rsl"
EXISTING_PID="${1:-}"
mkdir -p logs/heap-eetracking

notify() { # title, message
  curl -s -m 20 -H "Title: $1" -d "$2" "$NTFY_URL" >/dev/null || true
}

summary() { # objective -> final-round episode_reward per seed
  local obj=$1 out=""
  for seed in 1 2 3; do
    local csv="logs/heap-eetracking/${obj}/${seed}/train.csv"
    [ -f "$csv" ] && out+="seed${seed}: $(tail -1 "$csv" | cut -d, -f4 | cut -c1-8)  "
  done
  echo "${out:-no results}"
}

run_seed() { # obj seed
  local obj=$1 seed=$2
  local log="logs/heap-eetracking/suite_${obj}_seed${seed}.log"
  echo "=== $obj seed $seed started $(date) ===" >>"$log"
  "$PY" scripts/train_heap_benchmark.py --objective "$obj" --seed "$seed" >>"$log" 2>&1
}

overall_rc=0

# ---- tuned set (seed 1 may already be running, handed over via EXISTING_PID) ----
t0=$(date +%s)
fails=""
pids=()
for seed in 2 3; do
  run_seed tuned "$seed" &
  pids+=($!)
done
i=2
for pid in "${pids[@]}"; do
  wait "$pid" || { overall_rc=1; fails+=" tuned_seed${i}"; notify "safe_mbrl heap benchmark: FAILURE" "tuned seed $i failed. Log tail: $(tail -c 300 logs/heap-eetracking/suite_tuned_seed${i}.log)"; }
  i=$((i + 1))
done
if [ -n "$EXISTING_PID" ]; then
  while kill -0 "$EXISTING_PID" 2>/dev/null; do sleep 30; done
  grep -q "^Done:" logs/heap-eetracking/suite_tuned_seed1.log || { overall_rc=1; fails+=" tuned_seed1"; notify "safe_mbrl heap benchmark: FAILURE" "tuned seed 1 (pre-existing job) did not finish cleanly. Log tail: $(tail -c 300 logs/heap-eetracking/suite_tuned_seed1.log)"; }
fi
mins=$((($(date +%s) - t0) / 60))
if [ -z "$fails" ]; then
  notify "safe_mbrl heap benchmark: tuned set done" "3 seeds (parallel) finished in ${mins} min. Final-round mean reward -> $(summary tuned)"
else
  notify "safe_mbrl heap benchmark: tuned set finished WITH FAILURES" "Failed:${fails}. Partial results -> $(summary tuned)"
fi

# ---- matching set ----
t0=$(date +%s)
fails=""
pids=()
for seed in 1 2 3; do
  run_seed matching "$seed" &
  pids+=($!)
done
i=1
for pid in "${pids[@]}"; do
  wait "$pid" || { overall_rc=1; fails+=" matching_seed${i}"; notify "safe_mbrl heap benchmark: FAILURE" "matching seed $i failed. Log tail: $(tail -c 300 logs/heap-eetracking/suite_matching_seed${i}.log)"; }
  i=$((i + 1))
done
mins=$((($(date +%s) - t0) / 60))
if [ -z "$fails" ]; then
  notify "safe_mbrl heap benchmark: matching set done" "3 seeds (parallel) finished in ${mins} min. Final-round mean reward -> $(summary matching)"
else
  notify "safe_mbrl heap benchmark: matching set finished WITH FAILURES" "Failed:${fails}. Partial results -> $(summary matching)"
fi

notify "safe_mbrl heap benchmark: suite complete" "Both objective sets done. Results in $(pwd)/logs/heap-eetracking/"
exit $overall_rc
