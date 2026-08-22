#!/usr/bin/env bash
# Sequential heap benchmark suite: 2 planner objectives x 3 seeds.
# Sends an ntfy notification after each finished set (and on any failure).
set -u
cd "$(dirname "$0")/.."
# /opt/openrobots (python3.8) on PYTHONPATH/LD_LIBRARY_PATH shadows the venv's pinocchio
unset PYTHONPATH LD_LIBRARY_PATH
PY=.venv/bin/python
# Personal push-notification endpoint - replace with your own ntfy topic, or leave:
# notify() degrades to a no-op if the URL is unreachable.
NTFY_URL="https://ntfy.fangnan.me/work_rsl"
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

overall_rc=0
for obj in tuned matching; do
  set_fail=0
  t0=$(date +%s)
  for seed in 1 2 3; do
    log="logs/heap-eetracking/suite_${obj}_seed${seed}.log"
    echo "=== $obj seed $seed started $(date) ===" | tee -a "$log"
    "$PY" scripts/train_heap_benchmark.py --objective "$obj" --seed "$seed" >>"$log" 2>&1
    rc=$?
    if [ $rc -ne 0 ]; then
      set_fail=1; overall_rc=1
      notify "safe_mbrl heap benchmark: FAILURE" \
        "$obj seed $seed exited rc=$rc after $((($(date +%s) - t0) / 60)) min. Log tail: $(tail -c 300 "$log")"
    fi
  done
  mins=$((($(date +%s) - t0) / 60))
  if [ $set_fail -eq 0 ]; then
    notify "safe_mbrl heap benchmark: $obj set done" \
      "3 seeds finished in ${mins} min. Final-round mean reward -> $(summary "$obj")"
  else
    notify "safe_mbrl heap benchmark: $obj set finished WITH FAILURES" \
      "See suite logs in logs/heap-eetracking/. Partial results -> $(summary "$obj")"
  fi
done
notify "safe_mbrl heap benchmark: suite complete" "Both objective sets done. Results in $(pwd)/logs/heap-eetracking/"
exit $overall_rc
