#!/usr/bin/env bash
# Gate lean: the vendored lean/ subtree builds and reproduces the committed AVBD kernels.
#   1. lake build (default target Cloth) and lake exe csr_falsify in lean/
#   2. kernels/avbd/gen.sh (emit mode, reading lean/) leaves kernels/avbd with no
#      content diff against HEAD
#   3. independent: lake exe emit_shaders into a temp dir, CR-stripped cmp of each
#      kernel in kernels.txt against kernels/avbd/slang
#   4. negative control: numthreads(64 -> numthreads(32 in a copy of vbd_init.slang
#      must make the same loop print DIFF vbd_init (and only that)
# Run from anywhere; logs land next to this script. Needs lake, slangc, python.
#
# Side-effect free: it stages nothing, and it refuses to run if kernels/avbd
# already has a content diff against HEAD (gen.sh rewrites it in place). When
# the regenerated tree matches HEAD it checks kernels/avbd back out, so a
# CRLF checkout is left byte-identical to how it was found; when it does not
# match, the regenerated files are left in place as the evidence. Temp dirs
# are removed. Logs name paths relative to the repo root (<tmp> for temp dirs),
# never the checkout's absolute location. The build dir (build/, ignored) and
# lean/.lake (ignored) are the only other things written.
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"
cd "$ROOT"
KERNELS=$(grep -v '^#' kernels/avbd/kernels.txt | tr '\n' ' ')

winpath() { command -v cygpath >/dev/null 2>&1 && cygpath -m "$1" || echo "$1"; }
TMP="$(mktemp -d)"
NEG="$(mktemp -d)"
trap 'rm -rf "$TMP" "$NEG"' EXIT
TMPROOT="$(dirname "$TMP")"
# scrub: absolute checkout and temp paths (both the POSIX and the Windows
# spelling, since lake and python print the latter) -> repo-relative / <tmp>
scrub() {
	sed -e "s#$(winpath "$ROOT")/##g" -e "s#$ROOT/##g" \
		-e "s#$(winpath "$ROOT")#.#g" -e "s#$ROOT#.#g" \
		-e "s#$(winpath "$TMPROOT")/tmp\.[A-Za-z0-9]*#<tmp>#g" \
		-e "s#$TMPROOT/tmp\.[A-Za-z0-9]*#<tmp>#g"
}

if ! git diff --quiet HEAD -- kernels/avbd; then
	echo "refusing: kernels/avbd has uncommitted content changes; commit or stash them first" >&2
	git diff --stat HEAD -- kernels/avbd >&2
	exit 2
fi

{
	echo "# lake build (default target Cloth) in lean/"
	echo "# $(date -u +%FT%TZ)  $(cd lean && lake --version)  toolchain $(cat lean/lean-toolchain)"
	t0=$(date +%s)
	( cd lean && lake build 2>&1 ); rc=$?
	echo "# exit $rc  wall $(( $(date +%s) - t0 )) s"
} 2>&1 | scrub > "$HERE/lake-build.log"
tail -2 "$HERE/lake-build.log"

{
	echo "# lake exe csr_falsify in lean/"
	echo "# $(date -u +%FT%TZ)"
	t0=$(date +%s)
	( cd lean && lake exe csr_falsify 2>&1 ); rc=$?
	echo "# exit $rc  wall $(( $(date +%s) - t0 )) s"
} 2>&1 | scrub > "$HERE/csr-falsify.log"
tail -1 "$HERE/csr-falsify.log"

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
	# is identical. The verdict is the content diff against HEAD; git status is
	# logged before and after the restoring checkout, which must leave it empty.
	echo "# git diff --stat HEAD -- kernels/avbd (content; must be empty):"
	df="$(git diff --stat HEAD -- kernels/avbd 2>/dev/null)"
	echo "$df"
	echo "# git status --porcelain kernels/avbd (stat only; LF over CRLF shows here):"
	git status --porcelain kernels/avbd
	if [ -z "$df" ]; then
		git checkout -- kernels/avbd
		echo "# git status --porcelain kernels/avbd after git checkout -- kernels/avbd (must be empty):"
		st="$(git status --porcelain kernels/avbd)"
		echo "$st"
		[ -z "$st" ] && echo "RESULT CLEAN" || echo "RESULT DIRTY"
	else
		echo "# content differs; regenerated files left in the worktree"
		echo "RESULT DIRTY"
	fi
} 2>&1 | scrub > "$HERE/gen.log"
tail -2 "$HERE/gen.log"

{
	echo "# lake exe emit_shaders <tmp>   ($(date -u +%FT%TZ))"
	t0=$(date +%s)
	( cd lean && lake exe emit_shaders "$(winpath "$TMP")" | tail -1 )
	echo "# exit $?  wall $(( $(date +%s) - t0 )) s; $(ls "$TMP"/*.slang | wc -l) .slang emitted"
	echo "# CR-stripped cmp against kernels/avbd/slang, $(echo $KERNELS | wc -w) kernels:"
	cmp_loop "$TMP"; nd=$?
	echo "RESULT $nd DIFF"
} 2>&1 | scrub > "$HERE/cmp.log"
tail -1 "$HERE/cmp.log"

{
	cp "$TMP"/*.slang "$NEG"/
	echo "# negative control: numthreads(64 -> numthreads(32 in a copy of vbd_init.slang"
	echo "# occurrences before: $(grep -c 'numthreads(64' "$NEG/vbd_init.slang")"
	sed -i 's/numthreads(64/numthreads(32/' "$NEG/vbd_init.slang"
	echo "# occurrences after:  $(grep -c 'numthreads(32' "$NEG/vbd_init.slang")"
	cmp_loop "$NEG"; nd=$?
	echo "RESULT $nd DIFF (expect 1: vbd_init)"
} 2>&1 | scrub > "$HERE/negative-control.log"
grep -E '^DIFF|^RESULT' "$HERE/negative-control.log"

echo "# git status --porcelain (whole tree, after the gate):"
git status --porcelain
