#!/bin/bash
# Gate 0H, pod side: does stock Godot + godot_sandbox get a Vulkan RenderingDevice on
# a RunPod NVIDIA GPU with no display, and does the loop run there?
#
# Runs in /work on a plain ubuntu:22.04 pod started by run_pod.py with
# NVIDIA_DRIVER_CAPABILITIES=all. /work/tree.tar is this repo's project/ (plus
# fit.elf), gates/2-avbd and vendor/xr-grid at the commit under test. Everything
# lands in /work/out; summary.txt is the verdict list.
#
#   V0  vulkaninfo with no display: is an NVIDIA Vulkan device there at all?
#   V1  godot --headless (control): expect no RenderingDevice, gate FAIL
#   V2  Xvfb + the NVIDIA ICD: Gate 1 (rd_compute) must PASS on the NVIDIA adapter
#   V3  Xvfb + lavapipe (control): the same gate on CPU Vulkan, to tell "the GPU
#       path is blocked" from "RenderingDevice never works here"
#   V4  if V2 passed: Gate 8, the loop, FIXTURE:infer,rig, on the NVIDIA adapter
set -u
OUT=/work/out
mkdir -p "$OUT"
: > "$OUT/summary.txt"
log() { echo "$(date -u +%H:%M:%S) $*" | tee -a "$OUT/summary.txt"; }
GODOT_ZIP=Godot_v4.7.2-stable_linux.x86_64.zip
G=/work/Godot_v4.7.2-stable_linux.x86_64

export DEBIAN_FRONTEND=noninteractive
apt-get update -qq > "$OUT/apt.log" 2>&1
apt-get install -y -qq xvfb libvulkan1 vulkan-tools mesa-vulkan-drivers libx11-6 libxcursor1 \
	libxinerama1 libxrandr2 libxi6 libxext6 libxrender1 libgl1 libegl1 libfontconfig1 \
	libxkbcommon0 unzip wget ca-certificates >> "$OUT/apt.log" 2>&1
log "apt rc=$?"
log "gpu $(nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader 2>&1)"
log "cpu $(nproc) x $(grep -m1 'model name' /proc/cpuinfo | cut -d: -f2 | sed 's/^ //')"
{ echo "NVIDIA_DRIVER_CAPABILITIES=${NVIDIA_DRIVER_CAPABILITIES:-}"; ls -la /usr/share/vulkan/icd.d /etc/vulkan/icd.d 2>&1;
  ldconfig -p | grep -i 'nvidia' ; } > "$OUT/icd.txt" 2>&1
if ! ls /etc/vulkan/icd.d/nvidia_icd.json /usr/share/vulkan/icd.d/nvidia_icd.json > /dev/null 2>&1; then
	if ldconfig -p | grep -q 'libGLX_nvidia.so.0'; then
		mkdir -p /etc/vulkan/icd.d
		echo '{"file_format_version":"1.0.0","ICD":{"library_path":"libGLX_nvidia.so.0","api_version":"1.3.0"}}' \
			> /etc/vulkan/icd.d/nvidia_icd.json
		log "nvidia_icd.json written by this script (the toolkit mounted libGLX_nvidia.so.0 but no ICD file)"
	else
		log "no libGLX_nvidia.so.0 in the container: the graphics driver capability was not granted"
	fi
fi
NV_ICD=$(ls /etc/vulkan/icd.d/nvidia_icd.json /usr/share/vulkan/icd.d/nvidia_icd.json 2> /dev/null | head -1)
LVP_ICD=$(ls /usr/share/vulkan/icd.d/lvp_icd*.json 2> /dev/null | head -1)

VK_ICD_FILENAMES="$NV_ICD" vulkaninfo --summary > "$OUT/v0-vulkaninfo-nodisplay.txt" 2>&1
log "V0 vulkaninfo, no display, NVIDIA ICD rc=$? $(grep -m2 'deviceName' "$OUT/v0-vulkaninfo-nodisplay.txt" | tr -s ' ' | tr '\n' ';')"

cd /work
wget -q "https://github.com/godotengine/godot/releases/download/4.7.2-stable/$GODOT_ZIP" && unzip -oq "$GODOT_ZIP"
log "godot $("$G" --headless --version 2>&1 | tail -1)"
tar xf tree.tar
timeout 900 "$G" --headless --path project --import > "$OUT/import.log" 2>&1
log "import rc=$?"

rd_gate() { # $1 tag, rest: env assignments
	local tag=$1; shift
	env "$@" timeout 300 "$G" --path project --rendering-driver vulkan --xr-mode off --audio-driver Dummy \
		--script gate_rd_compute.gd > "$OUT/$tag.log" 2>&1
	local rc=$?
	log "$tag rc=$rc $(grep -m1 -o 'Using Device.*' "$OUT/$tag.log") $(grep -m1 -E '^(PASS|FAIL)' "$OUT/$tag.log")"
	return $rc
}

timeout 120 "$G" --headless --path project --script gate_rd_compute.gd > "$OUT/v1-headless.log" 2>&1
log "V1 headless control rc=$? $(grep -m1 -E 'FAIL|PASS|null|RenderingDevice' "$OUT/v1-headless.log")"

Xvfb :99 -screen 0 1280x720x24 -nolisten tcp > "$OUT/xvfb.log" 2>&1 &
sleep 3
export DISPLAY=:99
VK_ICD_FILENAMES="$NV_ICD" vulkaninfo --summary > "$OUT/v2-vulkaninfo-xvfb.txt" 2>&1
log "V2 vulkaninfo on Xvfb rc=$? $(grep -m2 'deviceName' "$OUT/v2-vulkaninfo-xvfb.txt" | tr -s ' ' | tr '\n' ';')"
rd_gate v2-nvidia VK_ICD_FILENAMES="$NV_ICD"
V2=$?
rd_gate v3-lavapipe VK_ICD_FILENAMES="$LVP_ICD"

if [ "$V2" = 0 ]; then
	t0=$(date +%s)
	VK_ICD_FILENAMES="$NV_ICD" timeout 3300 "$G" --path project --rendering-driver vulkan --xr-mode off \
		--audio-driver Dummy --script gate_loop.gd -- --gate=loop --out=/work/out/v4-loop-results.txt \
		--wallclock=3200 --allow-fixture=infer,rig > "$OUT/v4-loop.log" 2>&1
	log "V4 loop rc=$? $(( $(date +%s) - t0 )) s $(grep -m1 '^RESULT' "$OUT/v4-loop-results.txt" 2> /dev/null)"
else
	log "V4 loop skipped: V2 did not pass"
fi
log DONE
