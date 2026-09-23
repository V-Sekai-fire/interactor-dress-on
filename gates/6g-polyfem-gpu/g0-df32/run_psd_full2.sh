#!/bin/bash
# Gate 6g.0, second run: the df32 arms over ALL phases with psd on, after the fit_df.cpp fix (src/).
set -u
export MSYS_NO_PATHCONV=1
S=C:/Users/ernest.lee/AppData/Local/Temp/claude/C--interactor-dress-on/5e2e38d1-70e6-4b5e-96a1-5253bf6ca9f5/scratchpad/g6g0
export G6G_REPO=C:/Users/ernest.lee/AppData/Local/Temp/claude/C--interactor-dress-on/5e2e38d1-70e6-4b5e-96a1-5253bf6ca9f5/scratchpad/ido-6g
export LOGS=C:/b/g6g0-logs OUT=C:/b/g6g0-out CAP_S=1200
export EXTRA_ARGS='--set /solver/nonlinear/Newton/force_psd_projection=true'
cd "$S"
run() { echo "== $1 $(date +%H:%M:%S)"; bash run_arm.sh "$@"; }
run g1-df48-psd   99 G6G_FORMS=df G6G_DF_BITS=48
run g3-df48-accd-psd 99 G6G_FORMS=df G6G_DF_BITS=48 G6G_CCD=accd_df G6G_AUDIT=1
run g0-double-psd 99 G6G_FORMS=double
echo DONE
