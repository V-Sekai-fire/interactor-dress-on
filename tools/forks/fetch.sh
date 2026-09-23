#!/usr/bin/env bash
# Clone the pinned V-Sekai-fire forks from tools/forks.tsv into .forks/<name>
# (gitignored), each checked out detached at its pinned rev, so a CMake build
# can point CPM at them with -DCPM_<pkg>_SOURCE=.forks/<name> instead of
# letting CPM fetch the upstream URL the recipe names (AGENTS.md rule 1).
#
# Usage: tools/forks/fetch.sh [consumer-prefix]   (default: cloth-fit)
#   Only rows whose consumer column starts with the prefix are fetched; the
#   default selects the five CPM packages of vendor/cloth-fit (json v3.11.2,
#   polysolve, libigl, ipc-toolkit, openvdb). FORKS_DIR overrides .forks,
#   FORKS_TSV the table.
#
# Per row: the ref to fetch is the tag named in pinned_rev ("(tag v2.5.0)")
# or else the branch named in the verified column ("branch polyfem"); the
# pinned SHA must be that ref or an ancestor of it on the org fork, so a rev
# reachable only through GitHub's fork-network store is refused. Clones are
# blobless (--filter=blob:none); blobs arrive with the checkout. A clone
# already at its pinned SHA is left alone. nlohmann/json is checked out
# sparse (include/, single_include/, LICENSE.MIT, meson.build), the same
# layout as the release include.zip the polysolve and ipc-toolkit recipes
# download: with no CMakeLists.txt, CPM does not add_subdirectory it and the
# recipes' own nlohmann_json INTERFACE target still gets created.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
TSV="${FORKS_TSV:-$ROOT/tools/forks.tsv}"
DEST="${FORKS_DIR:-$ROOT/.forks}"
PREFIX="${1:-cloth-fit}"

die() { echo "fetch.sh: $*" >&2; exit 1; }
[ -f "$TSV" ] || die "no $TSV"
mkdir -p "$DEST"

n=0
while IFS=$'\t' read -r -u 3 upstream url pinned consumer verified; do
  [ "$upstream" = "upstream" ] && continue
  case "$consumer" in "$PREFIX"*) ;; *) continue ;; esac

  name="$(basename "$url" .git)"
  sha="$(grep -oE '\b[0-9a-f]{40}\b' <<<"$pinned" | head -1)" || true
  [ -n "$sha" ] || die "$name: no 40-hex SHA in pinned_rev: $pinned"

  if tag="$(grep -oP '\(tag \K[^)]+' <<<"$pinned")"; then
    ref="refs/tags/$tag"
  elif branch="$(grep -oP '\bbranch \K[A-Za-z0-9_./-]+[A-Za-z0-9_/-]' <<<"$verified" | head -1)"; then
    ref="refs/heads/$branch"
  else
    die "$name: neither a tag in pinned_rev nor a branch in verified"
  fi

  dir="$DEST/$name"
  if [ -d "$dir/.git" ] && [ "$(git -C "$dir" rev-parse -q --verify HEAD 2>/dev/null)" = "$sha" ] \
     && [ -n "$(git -C "$dir" ls-files -t | grep -v '^S ' | head -1)" ]; then
    echo "$name: at ${sha:0:8} already"
    n=$((n + 1))
    continue
  fi

  echo "$name: $url $ref -> ${sha:0:8}"
  [ -d "$dir/.git" ] || git init -q "$dir"
  git -C "$dir" remote remove origin 2>/dev/null || true
  git -C "$dir" remote add origin "$url"
  git -C "$dir" config remote.origin.promisor true
  git -C "$dir" config remote.origin.partialclonefilter blob:none
  git -C "$dir" fetch -q --filter=blob:none --no-tags origin "+$ref:refs/forks/pin"
  git -C "$dir" merge-base --is-ancestor "$sha" refs/forks/pin \
    || die "$name: ${sha:0:8} is not on $ref of $url"

  if [ "$name" = "json" ]; then
    # LF, as in the zip (the other forks follow the user's core.autocrlf, as
    # CPM's own clones do). Patterns on stdin: as arguments, MSYS bash
    # rewrites "/include/" into a Windows path under the Git install.
    git -C "$dir" config core.autocrlf false
    printf '%s\n' /include/ /single_include/ /LICENSE.MIT /meson.build \
      | git -C "$dir" sparse-checkout set --no-cone --stdin
  fi
  git -C "$dir" -c advice.detachedHead=false checkout -q --detach "$sha"
  [ "$(git -C "$dir" rev-parse HEAD)" = "$sha" ] || die "$name: checkout is not at $sha"
  n=$((n + 1))
done 3<"$TSV"

[ "$n" -gt 0 ] || die "no rows in $TSV with a consumer starting '$PREFIX'"
echo "fetch.sh: $n forks under $DEST"
