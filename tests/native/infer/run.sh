#!/usr/bin/env bash
# One host run of an infer port under the hard cap (user rule 2026-09-23):
#
#   bash tests/native/infer/run.sh <log> <exe> [args...]
#
# Runs <exe> under `timeout` (IDO_RUN_TIMEOUT, default 300 s; SIGKILL 10 s
# later), tees stdout+stderr to <log>, and appends one summary line:
#   RUN <exe> rc=<n> wall_s=<t> [TIMEOUT]
# A timeout is a FAIL (exit 124): the log keeps whatever profile the program
# printed so far (nodes, per-stage host time) -- print it as you go, not at
# the end. Do not raise the cap to make a CPU run finish; shrink the problem
# (fewer tokens/steps, smaller resolution) or run it on ggml-vulkan.
set -uo pipefail
[ $# -ge 2 ] || { echo "usage: run.sh <log> <exe> [args...]" >&2; exit 2; }
LOG="$1"; shift
CAP="${IDO_RUN_TIMEOUT:-300}"
T0=$(date +%s.%N)
timeout -k 10 "$CAP" "$@" 2>&1 | tee "$LOG"
RC=${PIPESTATUS[0]}
WALL=$(awk -v a="$T0" -v b="$(date +%s.%N)" 'BEGIN{printf "%.1f", b-a}')
TAG=""
if [ "$RC" -eq 124 ] || [ "$RC" -eq 137 ]; then TAG=" TIMEOUT (FAIL: hard cap ${CAP} s)"; RC=124; fi
echo "RUN $1 rc=$RC wall_s=$WALL$TAG" | tee -a "$LOG"
exit "$RC"
