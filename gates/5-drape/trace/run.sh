#!/bin/sh
# Gate 5 trace diagnostic: drape.elf's sphere demo step by step against the
# native run (see ../README.md, "Where the sphere demo leaves native").
# Needs Godot 4.7.2 on PATH and the native output directory
# (../native/source_dir.txt under C:/cloth-dynamics-standalone/output or $NATIVE_OUT).
#   gates/5-drape/trace/run.sh            (GPU: GPU_INDEX, default 0)
set -e
HERE=$(cd "$(dirname "$0")" && pwd)
ROOT=$(cd "$HERE/../../.." && pwd)
OUT="$ROOT/build/trace"
mkdir -p "$OUT"
cd "$ROOT/project"
run() { godot --path . --script probe_drape_trace.gd --rendering-driver vulkan --xr-mode off --gpu-index "${GPU_INDEX:-0}" -- "$@" > /dev/null 2>&1; }
run out="$OUT/rd_mu0_exact.txt" backend=rd steps=350 mu=0.5397701956236457
run out="$OUT/rd_mu0_printed.txt" backend=rd steps=350 mu=0.539770
run out="$OUT/rd_mu03.txt" backend=rd steps=350 mu=0.3
run out="$OUT/rd_mu001.txt" backend=rd steps=350 mu=0.01
run out="$OUT/cpu_mu0_exact.txt" backend=cpu steps=150 mu=0.5397701956236457
cd "$HERE"
python compare_steps.py "$OUT/rd_mu0_exact.txt" iter0 > rd_mu0_exact.log
python compare_steps.py "$OUT/rd_mu0_printed.txt" iter0 > rd_mu0_printed.log
python compare_steps.py "$OUT/rd_mu03.txt" target > rd_mu03.log
python compare_steps.py "$OUT/rd_mu001.txt" iter1 > rd_mu001.log
python compare_steps.py "$OUT/cpu_mu0_exact.txt" iter0 > cpu_mu0_exact.log
python loss_attribution.py "$OUT/rd_mu001.txt" "$OUT/rd_mu03.txt" > loss_attribution.log
grep -H "^first" *.log
cat loss_attribution.log
