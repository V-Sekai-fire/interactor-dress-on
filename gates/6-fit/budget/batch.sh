#!/usr/bin/env bash
# Run fit budget candidates natively, JOBS at a time (each fit_native is one
# thread). A candidate file has one "NAME args..." line per candidate.
#   gates/6-fit/budget/batch.sh candidates.txt
set -uo pipefail
HERE="$(cd "$(dirname "$0")" && pwd -W 2>/dev/null || pwd)"
JOBS="${JOBS:-3}"
while read -r name args; do
  [ -z "$name" ] && continue
  case "$name" in \#*) continue ;; esac
  while [ "$(jobs -rp | wc -l)" -ge "$JOBS" ]; do wait -n; done
  eval "bash \"$HERE/run_native.sh\" $name $args" > /dev/null &
done < "$1"
wait
echo BATCH DONE
