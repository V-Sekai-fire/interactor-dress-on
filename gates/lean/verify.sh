#!/usr/bin/env bash
# Gate lean: the vendored lean/ subtree reproduces the committed AVBD kernels.
#   1. kernels/avbd/gen.sh (emit mode, now reading lean/) leaves kernels/avbd clean
#   2. independent: lake exe emit_shaders into a temp dir, CR-stripped cmp of each
#      kernel in kernels.txt against kernels/avbd/slang
#   3. negative control: numthreads(64 -> numthreads(32 in a copy of vbd_init.slang
#      must make the same loop print DIFF vbd_init (and only that)
# Run from anywhere; logs land next to this script. Needs lake, slangc, python.
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"
cd "$ROOT"
KERNELS=$(grep -v '^#' kernels/avbd/kernels.txt | tr '\n' ' ')

cmp_loop() { # $1 = dir of emitted .slang; prints SAME/DIFF per kernel, returns #DIFF
	local d="$1" n=0 k
	for k in $KERNELS; do
		if cmp -s <(tr -d '\r' < "$d/$k.slang") <(tr -d '\r' < "kernels/avbd/slang/$k.slang"); then
			echo "SAME $k"
		else
			echo "DIFF $k"; n=$((n + 1))
		fi
	done
	return $n
}

{
	echo "# AVBD_EMIT=1 BUILD_DIR=build bash kernels/avbd/gen.sh   ($(date -u +%FT%TZ))"
	t0=$(date +%s)
	AVBD_EMIT=1 BUILD_DIR=build bash kernels/avbd/gen.sh 2>&1
	echo "# exit $?  wall $(( $(date +%s) - t0 )) s"
	# With core.autocrlf=true a fresh checkout holds CRLF files while gen.sh
	# writes LF; git status then flags them on size alone although the content
	# is identical. The content check is git diff against HEAD; git status is
	# logged too, before and after git add (which stages nothing if clean).
	echo "# git diff --stat HEAD -- kernels/avbd (content; must be empty):"
	df="$(git diff --stat HEAD -- kernels/avbd 2>/dev/null)"
	echo "$df"
	echo "# git status --porcelain kernels/avbd:"
	git status --porcelain kernels/avbd
	git add kernels/avbd 2>/dev/null
	echo "# git status --porcelain kernels/avbd after git add (must be empty):"
	st="$(git status --porcelain kernels/avbd)"
	echo "$st"
	[ -z "$df" ] && [ -z "$st" ] && echo "RESULT CLEAN" || echo "RESULT DIRTY"
} > "$HERE/gen.log" 2>&1
tail -2 "$HERE/gen.log"

TMP="$(mktemp -d)"
NEG="$(mktemp -d)"
{
	echo "# lake exe emit_shaders $TMP   ($(date -u +%FT%TZ))"
	t0=$(date +%s)
	( cd lean && lake exe emit_shaders "$TMP" | tail -1 )
	echo "# exit $?  wall $(( $(date +%s) - t0 )) s; $(ls "$TMP"/*.slang | wc -l) .slang emitted"
	echo "# CR-stripped cmp against kernels/avbd/slang, $(echo $KERNELS | wc -w) kernels:"
	cmp_loop "$TMP"; nd=$?
	echo "RESULT $nd DIFF"
} > "$HERE/cmp.log" 2>&1
tail -1 "$HERE/cmp.log"

{
	cp "$TMP"/*.slang "$NEG"/
	echo "# negative control: numthreads(64 -> numthreads(32 in a copy of vbd_init.slang"
	echo "# occurrences before: $(grep -c 'numthreads(64' "$NEG/vbd_init.slang")"
	sed -i 's/numthreads(64/numthreads(32/' "$NEG/vbd_init.slang"
	echo "# occurrences after:  $(grep -c 'numthreads(32' "$NEG/vbd_init.slang")"
	cmp_loop "$NEG"; nd=$?
	echo "RESULT $nd DIFF (expect 1: vbd_init)"
} > "$HERE/negative-control.log" 2>&1
grep -E '^DIFF|^RESULT' "$HERE/negative-control.log"
rm -rf "$TMP" "$NEG"
