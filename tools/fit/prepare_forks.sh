#!/usr/bin/env bash
# Fetch the cloth-fit forks (tools/forks/fetch.sh) and prepare
# .forks/polysolve-guest: a second worktree of the polysolve fork at its pin
# with vendor/cloth-fit-patches/polysolve-embedded-specs.patch applied, so the
# fit builds (fit_native, fit.elf) take polysolve's rules from the embedded
# specs while .forks/polysolve stays the unpatched source the oracle uses.
#
# Idempotent: an already-patched worktree is left alone; one with other local
# changes is refused. Files are checked out LF (the patch is LF).
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && (pwd -W 2>/dev/null || pwd))"
FORKS="${FORKS_DIR:-$ROOT/.forks}"
PATCH="$ROOT/vendor/cloth-fit-patches/polysolve-embedded-specs.patch"

FORKS_DIR="$FORKS" bash "$ROOT/tools/forks/fetch.sh" cloth-fit >/dev/null

src="$FORKS/polysolve"
dst="$FORKS/polysolve-guest"
sha="$(git -C "$src" rev-parse HEAD)"

if [ ! -e "$dst/.git" ]; then
  git -C "$src" -c core.autocrlf=false worktree add -q --detach "$dst" "$sha"
fi
[ "$(git -C "$dst" rev-parse HEAD)" = "$sha" ] \
  || { echo "prepare_forks.sh: $dst is not at $sha" >&2; exit 1; }

if git -C "$dst" -c core.autocrlf=false apply --reverse --check "$PATCH" 2>/dev/null; then
  echo "polysolve-guest: patch already applied at ${sha:0:8}"
elif git -C "$dst" -c core.autocrlf=false diff --quiet; then
  git -C "$dst" -c core.autocrlf=false apply "$PATCH"
  echo "polysolve-guest: patched at ${sha:0:8}"
else
  echo "prepare_forks.sh: $dst has local changes that are not the patch" >&2
  exit 1
fi
