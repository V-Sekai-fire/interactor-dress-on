# Gate lean — the vendored `lean/` subtree reproduces the AVBD kernels

**Result: PASS.** `lean/` (cloth-dynamics' `lean/` at e361584, a squashed
subtree) builds with its dependencies fetched from the V-Sekai-fire forks,
`csr_falsify` finds its witnesses, and the 24 AVBD kernels it emits are
byte-identical (CR-stripped) to the committed `kernels/avbd/slang/`. The
negative control flips.

| check | log | result | wall |
|---|---|---|---|
| `lake build` (default target `Cloth`), LeanSlang copied in | `lake-build.log` | 74 jobs, exit 0 | 94 s |
| `lake build` from an empty `lean/.lake` (copy of `lean/`) | `lake-build-clean.log` | 74 jobs, exit 0 | 134 s |
| `lake exe csr_falsify` | `csr-falsify.log` | both ladders `Outcome.found 0`, 7 plausible properties "Unable to find a counter-example", exit 0 | 9 s |
| `AVBD_EMIT=1 BUILD_DIR=build bash kernels/avbd/gen.sh` | `gen.log` | `git diff HEAD -- kernels/avbd` empty; `git status` empty | 70 s |
| `lake exe emit_shaders $tmp` + CR-stripped `cmp` of the 24 | `cmp.log` | 41 emitted, 24 `SAME`, 0 `DIFF` | 7 s |
| negative control: `numthreads(64` → `numthreads(32` in a copy of `vbd_init.slang` | `negative-control.log` | exactly one `DIFF vbd_init`, 23 `SAME` | — |

`verify.sh` runs the last three (from anywhere; it `cd`s to the repo root).
Toolchain: Lake 5.0.0 / Lean 4.30.0 (`lean/lean-toolchain`), slangc from the
scoop Vulkan SDK.

## What `csr_falsify` shows

The falsifiability run, not only the properties: a role-dropping CSR build is
caught by `rolesAddressTheRightVertex` and an all-one-colour colouring by
`separationHolds`, each at ladder level 0 with the first witness
(`mesh=(2, [[0, 1]])`). A property that no broken build can fail would pass
silently; these two show the properties can fail.

## Two findings on the way

- **The committed cpp emits named the checkout.** The first `gen.sh` run in
  this worktree left all 24 `cpp/*_emit.cpp` modified while every `.slang`
  matched: slangc writes its input path into `#line`, and `gen.sh` passed
  `C:/interactor-dress-on/kernels/avbd/slang/<k>.slang`. `gen.sh` now runs
  slangc from `kernels/avbd` with relative paths (commit "kernels/avbd:
  gen.sh gives slangc relative paths …"); the regenerated cpp differ from
  the old ones in one `#line` path each (24 lines over 24 files).
- **`git status` flags identical files on a fresh checkout.** With
  `core.autocrlf=true` the checkout holds CRLF and `gen.sh` writes LF; git
  status lists 25 files ` M` on size alone while `git diff HEAD` is empty.
  The gate therefore takes the content diff as the verdict and logs git
  status before and after `git add kernels/avbd` (which stages nothing). This
  run started from a fresh checkout of `kernels/avbd` (24 `w/crlf`) to show
  it.

## Pins

| dependency | repository | rev |
|---|---|---|
| toolchain | leanprover/lean4 | v4.30.0 |
| LeanSlang | V-Sekai-fire/lean-slang | v0.0.5 = 813d6c62b298bcd3f7177264179a7e1d16bd87cd |
| plausible | V-Sekai-fire/plausible (fork of leanprover-community, created 2026-09-22) | v4.30.0 = a456461b368b71d2accd95234832cd9c174b5437 |
| plausible-witness-dag | V-Sekai-fire/plausible-witness-dag | 160b94c9c6eed3bb9ebffce919fc6f989dcafba8 (pinned by SHA; the fork's `main/main` is 12 ahead) |

The manifest revs are the ones cloth-dynamics resolved; only the URLs and
the witness-dag `inputRev` changed, and `lake build` left the manifest
untouched. `lean/.lake/packages` for `lake-build.log` held a copy of the
standalone checkout's built LeanSlang (URL unchanged); plausible and
plausible-witness-dag were cloned fresh from the forks.

## Source

`git subtree split --prefix=lean` of a scratch clone of
`/c/cloth-dynamics-standalone` (branch `feature/vulkan-avbd-windows-backend`,
HEAD e361584, `lean/` clean): 55 files, 52 commits, split head f8f0d2c,
tree 9599170 = `e361584:lean`. The split walked 425 commits in ~2 min.
Added with `git subtree add --prefix=lean <clone> lean-split --squash`.
The source commit is on no remote yet (plan risk 7).

## Not in this gate

The LeanSlang bump to `emit-fp` (Cut L step 4) and a guest rebuild: the cpp
change is `#line` paths only, so the ELF's code is unaffected.
