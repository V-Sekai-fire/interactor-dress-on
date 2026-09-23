# Gates 6g: PolyFEM's fit on the GPU (cut 6g-C)

State on `cut-6g` (WIP, parked 2026-09-23 at 99 % of the weekly budget).

| gate | question | state |
|---|---|---|
| `g0-df32/` | does float-float carry the whole fit? | **PASS** (committed `de409d979`): all four foxgirl phases with psd in df48 and df48+ACCD(df32) converge inside the Gate 6 guard, ACCD audit 0 of 6.0 M queries |
| `g1-rd-worker/` | rd_compute from the fit's worker Thread | PASS (merged in main) |
| `c1-hessian/` | SimilarityForm's Hessian on the device, df32 | **built, not yet gated**: fit.elf (`d2b7055e`, `runs/build-fit.log`) with the three kernels and their cpp twin compiles and loads; `fit_begin` reports 15,516 hinges, 2,682 slots, nnz 290,268, 14.5 MiB session data (`runs/check.txt`). The check arm (`fit_gpu_check`) was killed after 5 min inside the vmcall with no report: the cpp twin runs 15,516 lanes x 1,424 df32 ops serially on the guest CPU, six times, plus six CPU double passes; it needs `execution_timeout` raised (2,500,000 units may be short) or the twin arm cut to one comparison. Nothing has been read back from the GPU yet. |
| `c2-broad-phase/` | the CCD broad phase on the device | **written, not built**: `lean/Fit/{SweptAabb,BoxPair}.lean` emit and compile to SPIR-V and cpp; `guest/fit/fit_broad_phase.{h,cpp}`, the ContactForm hook and the `fit_set_gpu_broad` API are in the tree but fit.elf has NOT been rebuilt with them (the last build predates patch 4). |

## What is where

- Kernels (rule 2, all from Lean): `lean/Fit/Df32.lean` (float-float arithmetic, `precise` EFTs), `Hess12.lean` (a second-order forward tape over 12 variables, run at emit time), `SimilarityHessianBlock.lean` (the 12x12 block per hinge from the similarity energy, 1,424 tape statements, pinned), `ProjectPsd12.lean` (Jacobi PSD projection), `CsrGatherDf32.lean` (fixed-pattern gather, `expected` pinned), `SweptAabb.lean`, `BoxPair.lean`. `kernels/fit/gen.sh` emits, compiles both targets and embeds the SPIR-V into `$BUILD/fit_kernels.inc`; `build.sh` runs it with `BUILD_FIT=1` (`FIT_EMIT=1` re-emits).
- Guest: `guest/fit/fit_sim_hessian.{h,cpp}` (session data, the pattern with per-nonzero gather lists, the cpp twin, comparisons), `fit_gpu.{h,cpp}` (the device path: one compute list, sync inside the vmcall, `buffer_get_into` readback, `worker_syncs` counter in `rdc::Device`), `fit_broad_phase.{h,cpp}`, `main.cpp` (`fit_set_gpu` 0 CPU / 1 device / 2 twin, `fit_set_gpu_broad` 0 / 1 / 2 audited, `fit_gpu_check`, `fit_gpu_stats`, `fit_gpu_close`; a fallback inside a phase FAILs the step, never silent).
- Hooks in vendor/cloth-fit: `SimilarityForm::set_hessian_hook` + `hessian_blocks` (GarmentForm), `ContactForm::set_broad_phase_hook`. With no hook set (mode 0) both paths are the untouched CPU code.
- Host: `stages/stage_base.gd` gained a persistent worker Thread (`persistent_worker`, used by the fit stage: the device is bound to its thread); `fit_stage.gd` / `main.gd` wrappers (rule 8) and `fit_force_psd`; `gate_fit.gd` arms `gpucheck`, options `--gpu=`, `--broad=`, `--psd`.

## Resume

1. Rebuild with the broad phase: `cd <worktree> && BUILD_DIR=C:/b/ido6g BUILD_JOBS=12 BUILD_TARGETS=fit bash build.sh`, then `godot --path project --headless --import --xr-mode off`.
2. Raise the fit stage's `fit_execution_timeout` for the check (or pass `--timeout=20000000` to gate_fit), then `bash gates/6g-polyfem-gpu/c1-hessian/run.sh check`. Expect: blocks rd vs CPU double <= 1e-9, twin vs rd <= 1e-12, the sign-flipped control FAILS, pattern identical. If the twin is the slow part, drop it to one arm.
3. `bash gates/6g-polyfem-gpu/c1-hessian/run.sh gpu` then `cpu`: per-phase `gpu:` lines give round trips per Newton and their share (<= 10 %); the guard via `gates/6-fit/fit_gap.py` on `runs/*.garment.f64` against `C:/b/cf-up-out1/step_avatar_252.obj`.
4. `bash gates/6g-polyfem-gpu/c2-broad-phase/run.sh audit` (missed must be 0, extra <= 5 %), then `gpu` and `cpu` for the fit time before/after; `tests/probe_main_wrappers.gd` for rule 8.
5. Write each gate's README with the numbers, then commit; `gate_loop.gd --fit-mode=polyfem` for D.
