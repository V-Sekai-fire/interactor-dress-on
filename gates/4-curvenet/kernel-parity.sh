#!/usr/bin/env bash
# Kernel parity: the cpp that kernels/cassie/gen.sh lowers from lean/ against
# the prebuilt .cpu.cpp the CASSIE module ships (entities-godot c165a519d2,
# modules/cassie/thirdparty/avbd), with the entities-godot post-processing
# (prelude include rewrite, SLANG_PRELUDE_EXPORT neutering, namespace wrap),
# #line directives, CRs and blank lines stripped from both.
#
#   CASSIE_AVBD=<entities-godot>/modules/cassie/thirdparty/avbd \
#     gates/4-curvenet/kernel-parity.sh > gates/4-curvenet/kernel-parity.log
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"
REF="${CASSIE_AVBD:-C:/contract-manifest/4-entities/godot/modules/cassie/thirdparty/avbd}"
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

norm() {
	# Drop the prelude: the reference #includes it, current slangc inlines it
	# (4.2k lines, guarded by SLANG_CPP_PRELUDE_H); both follow it with the
	# `using namespace SLANG_PRELUDE_NAMESPACE;` block, whose #endif is the cut.
	tr -d '\r' < "$1" \
	| awk 'body { print; next } /^using namespace SLANG_PRELUDE_NAMESPACE;$/ { u = 1; next } u && /^#endif$/ { body = 1 }' \
	| grep -v -E '^#line |^#include ".*slang-cpp-prelude\.h"|^#undef SLANG_PRELUDE_EXPORT$|^#define SLANG_PRELUDE_EXPORT$|^namespace cassie_slang_[a-z_0-9]+ \{$|^\} // namespace cassie_slang_' \
	| grep -v -E '^[[:space:]]*$'
}

echo "reference: entities-godot c165a519d2 modules/cassie/thirdparty/avbd/<ref>.cpu.cpp"
echo "emitted:   kernels/cassie/cpp/<k>_emit.cpp ($("${SLANGC:-slangc}" -v 2>&1 | head -1))"
echo
total=0
while read -r k ref; do
	norm "$REF/$ref.cpu.cpp" > "$TMP/ref"
	norm "$ROOT/kernels/cassie/cpp/${k}_emit.cpp" > "$TMP/new"
	n=$(diff "$TMP/ref" "$TMP/new" | grep -c '^[<>]')
	total=$((total + n))
	echo "== $k vs $ref.cpu.cpp: $(wc -l < "$TMP/ref") / $(wc -l < "$TMP/new") lines, $n differing =="
	diff "$TMP/ref" "$TMP/new"
	echo
done <<'EOF'
curve_casteljau curve_casteljau
curve_generate_bezier curve_generate_bezier
curve_newton curve_newton
curve_rdp curve_rdp
spmv_df32 spmv
EOF
echo "total differing lines: $total"

# Negative control: the same normalisation must not hide a real change. One
# float literal in a copy of the curve_generate_bezier emit, 3.0f -> 3.5f.
sed '0,/3\.0f/s//3.5f/' "$ROOT/kernels/cassie/cpp/curve_generate_bezier_emit.cpp" > "$TMP/ctl.cpp"
norm "$REF/curve_generate_bezier.cpu.cpp" > "$TMP/ref"
norm "$TMP/ctl.cpp" > "$TMP/new"
echo
echo "negative control (one 3.0f -> 3.5f in a copy of curve_generate_bezier_emit.cpp): $(diff "$TMP/ref" "$TMP/new" | grep -c '^[<>]') differing lines"
diff "$TMP/ref" "$TMP/new"
