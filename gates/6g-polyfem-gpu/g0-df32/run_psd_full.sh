#!/bin/bash
# Gate 6g.0: the decisive arms over ALL phases with psd on: c0 double, c1 df48, c3 df48 + ACCD(df32, recomputed margins) with the audit.
set -u
export MSYS_NO_PATHCONV=1
S=C:/Users/ernest.lee/AppData/Local/Temp/claude/C--interactor-dress-on/5e2e38d1-70e6-4b5e-96a1-5253bf6ca9f5/scratchpad/g6g0
export LOGS=C:/b/g6g0-logs OUT=C:/b/g6g0-out CAP_S=900
export EXTRA_ARGS='--set /solver/nonlinear/Newton/force_psd_projection=true'
cd "$S"
run() { echo "== $1 $(date +%H:%M:%S)"; bash run_arm.sh "$@"; }
run f0-double-psd 99 G6G_FORMS=double
run f1-df48-psd   99 G6G_FORMS=df G6G_DF_BITS=48
run f3-df48-accd-psd 99 G6G_FORMS=df G6G_DF_BITS=48 G6G_CCD=accd_df G6G_AUDIT=1
python analyze.py "$LOGS/f0-double-psd.log" "$LOGS/f1-df48-psd.log" "$LOGS/f3-df48-accd-psd.log" > "$LOGS/f-analyze.txt" 2>&1
echo "analyze rc=$?"; grep -E '^== |phases|trajectory|MISSED|missed|extra' "$LOGS/f-analyze.txt" | cut -c1-220
for a in f0-double-psd f1-df48-psd f3-df48-accd-psd; do echo "-- $a"; grep -E '^phase [0-9]|^final:|MISSED|missed|extra|audit' "$LOGS/$a.log" | cut -c1-200 | head -12; done
echo DONE
