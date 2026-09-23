#!/usr/bin/env bash
# Install the own-built wheels in a runtime container (Dockerfile.test) and, on each GPU,
# run kernels.py (one real kernel per extension module, checked against a reference)
# and the service's own check (tools/services/pixal3d smoke.py --imports-only).
#
#   gates/7-pixal3d/own-wheels/test.sh [-w wheels_dir] [-t tag] [gpu ...]   # default GPUs: 0 1
#
# -w: the directory whose .whl files env/pixi.toml names by path (omit when it names the
# release URLs). Logs: logs/test-<tag>-gpu<N>.log. Exit 0 iff every run passes.
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
root=$(cd "$here/../../.." && pwd)
wheels="" tag=local
while getopts "w:t:" o; do
    case $o in w) wheels=$OPTARG ;; t) tag=$OPTARG ;; *) exit 2 ;; esac
done
shift $((OPTIND - 1))
gpus=${*:-0 1}
empty=$(mktemp -d)
[ -n "$wheels" ] || wheels=$empty
wheels=$(cd "$wheels" && pwd)
mkdir -p "$here/logs"
if pwd -W >/dev/null 2>&1; then  # Git Bash on Windows: hand Docker C:/... paths
    root=$(cd "$root" && pwd -W); wheels=$(cd "$wheels" && pwd -W)
    export MSYS_NO_PATHCONV=1
fi
image=pixal3d-own-wheels-test:$tag
docker build -f "$root/gates/7-pixal3d/own-wheels/Dockerfile.test" --build-context wheels="$wheels" -t "$image" "$root"
rm -rf "$empty"
rc=0
for g in $gpus; do
    log="$here/logs/test-$tag-gpu$g.log"
    # a container per GPU, one at a time
    docker run --rm --gpus "device=$g" "$image" bash -c '
        nvidia-smi --query-gpu=index,name,compute_cap,driver_version,memory.used --format=csv
        echo "== kernels.py"; python /app/kernels.py; k=$?
        echo "== smoke.py --imports-only (the service check)"; python smoke.py --imports-only; s=$?
        echo "== kernels rc=$k, check rc=$s"; exit $(( k | s ))' > "$log" 2>&1 || rc=1
    grep -E "^(device|  (ok|FAIL)|PASS|FAIL|== kernels rc)" "$log" | sed "s/^/[gpu$g] /"
done
exit $rc
