# Gate lean — the vendored `lean/` subtree reproduces the AVBD kernels

**Result: PASS**, re-run after the LeanSlang bump to `emit-fp` (Cut L step
4). `lean/` (cloth-dynamics' `lean/` at e361584, a squashed subtree) builds
with its dependencies fetched from the V-Sekai-fire repos and forks,
`csr_falsify` finds its witnesses, and the 24 AVBD kernels it emits are
byte-identical (CR-stripped) to the committed `kernels/avbd/slang/`. The
negative control flips. The first run (LeanSlang v0.0.5) gave the same
verdicts; its timings are in the commit that added this gate.

| check | log | result | wall |
|---|---|---|---|
| `lake build` (default target `Cloth`), after `lake update LeanSlang` | `lake-build.log` | 75 jobs (LeanSlang and every `Cloth` module rebuilt), exit 0 | 15 s |
| `lake build` from an empty `lean/.lake` (copy of `lean/`) | `lake-build-clean.log` | 75 jobs, exit 0; manifest unchanged by the build | 50 s |
| `lake exe csr_falsify` | `csr-falsify.log` | both ladders `Outcome.found 0`, 7 plausible properties "Unable to find a counter-example", exit 0 | 1 s |
| `AVBD_EMIT=1 BUILD_DIR=build bash kernels/avbd/gen.sh` | `gen.log` | `git diff HEAD -- kernels/avbd` empty; `git status` empty after the restoring checkout | 20 s |
| `lake exe emit_shaders <tmp>` + CR-stripped `cmp` of the 24 | `cmp.log` | 41 emitted, 24 `SAME`, 0 `DIFF` | 2 s |
| negative control: `numthreads(64` → `numthreads(32` in a copy of `vbd_init.slang` | `negative-control.log` | exactly one `DIFF vbd_init`, 23 `SAME` | — |

`verify.sh` runs all but the clean build (from anywhere; it `cd`s to the repo
root). It is side-effect free: it stages nothing, refuses to start if
`kernels/avbd` already has a content diff against HEAD, checks
`kernels/avbd` back out when the regenerated tree matches (so a CRLF
checkout is left as it was found), removes its temp dirs, and writes logs
with repo-relative paths (`<tmp>` for temp dirs). Afterwards `git status`
shows only the logs it rewrote.
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
  The gate therefore takes the content diff as the verdict, logs git status,
  and then restores the checkout with `git checkout -- kernels/avbd` (the
  first version ran `git add`, which left the index touched).

## Pins

| dependency | repository | rev |
|---|---|---|
| toolchain | leanprover/lean4 | v4.30.0 |
| LeanSlang | V-Sekai-fire/contract-lean-slang (the renamed lean-slang) | branch `emit-fp` = 60532aef8ed70cc669ecab481182d0636c9e1ac3 (pinned by SHA; e0e96da plus the additive `litFloatExact`/`litDoubleExact`, AVBD emission unchanged; was lean-slang v0.0.5 = 813d6c6) |
| plausible | V-Sekai-fire/plausible (fork of leanprover-community, created 2026-09-22) | v4.30.0 = a456461b368b71d2accd95234832cd9c174b5437 |
| plausible-witness-dag | V-Sekai-fire/plausible-witness-dag | 160b94c9c6eed3bb9ebffce919fc6f989dcafba8 (pinned by SHA; the fork's `main/main` is 12 ahead) |

The plausible and plausible-witness-dag revs are the ones cloth-dynamics
resolved; only their URLs and the witness-dag `inputRev` changed. LeanSlang
was moved with `lake update LeanSlang` after deleting
`lean/.lake/packages/LeanSlang` (its cached URL was the old one); the
manifest diff is that one entry's `url`, `rev` and `inputRev`, and neither
`lake build` changed it.

`emit-fp` branches from v0.0.6 (8970c21, multi-entry-point accessors) and
adds the `half` and `double` scalars and the `litHalf`, `litInt` and `cast`
expressions, with `native_decide` fixtures in `LeanSlang.TestFp`. It is
additive, which is what the byte check confirms: the 24 AVBD kernels emit
unchanged. `main` is not used because it adds a libslang FFI `extern_lib` as
a default target (vendored SDK headers, Linux link flags), which breaks
`lake exe` on Windows.

## Source

`git subtree split --prefix=lean` of a scratch clone of
`/c/cloth-dynamics-standalone` (branch `feature/vulkan-avbd-windows-backend`,
HEAD e361584, `lean/` clean): 55 files, 52 commits, split head f8f0d2c,
tree 9599170 = `e361584:lean`. The split walked 425 commits in ~2 min.
Added with `git subtree add --prefix=lean <clone> lean-split --squash`.
The source commit is on no remote yet (plan risk 7).

## Not in this gate

A guest rebuild: the cpp change is `#line` paths only and the LeanSlang bump
emits identical Slang, so the ELF's code is unaffected.
