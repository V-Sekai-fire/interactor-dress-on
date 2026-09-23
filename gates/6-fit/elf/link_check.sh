#!/usr/bin/env bash
# What fit.elf links: size, sha256, sections, arch, and symbol counts for the
# libraries it must and must not contain. Writes gates/6-fit/elf/link.log.
#
#   bash gates/6-fit/elf/link_check.sh [path/to/fit.elf]
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$HERE/../../.." && pwd)"
ELF="${1:-$ROOT/project/fit.elf}"
BIN="${LLVM_BIN:-$HOME/scoop/apps/llvm/current/bin}"
OUT="$HERE/link.log"
NM="$(mktemp)"
"$BIN/llvm-nm" -C "$ELF" > "$NM"

{
	echo "fit.elf: $(stat -c %s "$ELF") bytes, sha256 $(sha256sum "$ELF" | cut -d' ' -f1)"
	echo "symbols: $(wc -l < "$NM")"
	echo
	echo "== sections (llvm-size -A) =="
	"$BIN/llvm-size" -A "$ELF" | sed -n '3,$p' | grep -v '^$'
	echo
	echo "== arch attribute (merged over every object) =="
	"$BIN/llvm-readobj" --arch-specific "$ELF" | grep -A1 "TagName: arch" | grep Value | sed 's/^ *//'
	echo
	echo "== must be present =="
	for p in "Eigen::" "polysolve::" "polyfem::" "ipc::" "igl::" "spdlog::" "nlohmann::" "fit::FitDriver" \
		"polyfem::solver::SdfGrid" "polyfem::solver::sdf_spline::hessian_batch" "main_0_Thread" \
		"__wrap_open" "__wrap_open64" "__wrap_openat" "__wrap_openat64" "__wrap_fopen" "__wrap_fopen64"; do
		printf '  %-45s %6d\n' "$p" "$(grep -cF -- "$p" "$NM" || true)"
	done
	echo
	echo "== must be absent =="
	for p in "tbb::detail" "tbb::v1" "r1::" "tbbmalloc" "scalable_malloc" "openvdb" "Imath" "boost::" "filib" "__real_open" "__real_fopen"; do
		printf '  %-45s %6d\n' "$p" "$(grep -cF -- "$p" "$NM" || true)"
	done
	echo
	echo "== tbb:: symbols: the serial stand-in's inline templates only =="
	grep -o 'tbb::[A-Za-z_0-9:]*' "$NM" | sed 's/<.*//' | sort | uniq -c | sed 's/^ */  /'
} > "$OUT"
rm -f "$NM"
cat "$OUT"
