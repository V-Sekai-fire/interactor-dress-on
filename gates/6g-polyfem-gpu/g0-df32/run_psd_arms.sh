#!/bin/bash
# Gate 6g.0, done properly: the df32 arms with force_psd_projection on (stage A), foxgirl phase 0.
#   c0 double (control) | c1 df48 | c2 df44 | c3 df48 + additive CCD in df32 with the audit | c4 forms=compare
set -u
export MSYS_NO_PATHCONV=1  # keep the JSON pointer in --set from becoming a Windows path
S=C:/Users/ernest.lee/AppData/Local/Temp/claude/C--interactor-dress-on/5e2e38d1-70e6-4b5e-96a1-5253bf6ca9f5/scratchpad/g6g0
export LOGS=C:/b/g6g0-logs OUT=C:/b/g6g0-out CAP_S=300
export EXTRA_ARGS='--set /solver/nonlinear/Newton/force_psd_projection=true'
cd "$S"
run() { echo "== $1 $(date +%H:%M:%S)"; bash run_arm.sh "$@"; }
run c0-double-psd 1 G6G_FORMS=double
run c1-df48-psd   1 G6G_FORMS=df G6G_DF_BITS=48
run c2-df44-psd   1 G6G_FORMS=df G6G_DF_BITS=44
run c3-df48-accd-psd 1 G6G_FORMS=df G6G_DF_BITS=48 G6G_CCD="${ACCD_DF:-accd_df}" G6G_AUDIT=1
run c4-compare-psd 1 G6G_FORMS=compare
python analyze.py "$LOGS/c0-double-psd.log" "$LOGS/c1-df48-psd.log" "$LOGS/c2-df44-psd.log" "$LOGS/c3-df48-accd-psd.log" > "$LOGS/c-analyze.txt" 2>&1
echo "analyze rc=$?"; head -60 "$LOGS/c-analyze.txt"
# fit gap per arm against the avatar (usage: AVATAR NO_FIT VOXEL GARMENT...), c0 first
R=C:/Users/ernest.lee/AppData/Local/Temp/claude/C--interactor-dress-on/5e2e38d1-70e6-4b5e-96a1-5253bf6ca9f5/scratchpad/ido-6g0
G=""; for a in c0-double-psd c1-df48-psd c2-df44-psd c3-df48-accd-psd; do [ -f "$OUT/$a/garment_final.obj" ] && G="$G $OUT/$a/garment_final.obj"; done
python /c/interactor-dress-on/gates/6-fit/fit_gap.py "$R/vendor/cloth-fit/garment-data/assets/avatars/FoxGirl/avatar.obj" \
	"$R/vendor/cloth-fit/garment-data/assets/garments/LCL_Skirt_DressEvening_003/no-fit.txt" 0.01 $G > "$LOGS/c-gap.txt" 2>&1
echo "gap rc=$?"; cat "$LOGS/c-gap.txt"
echo DONE
