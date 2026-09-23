# Gate 6 (elf): fit.elf builds, loads and runs a phase in Godot

`fit.elf` is cloth-fit's garment solve for the guest. It contains PolyFEM,
polysolve, ipc-toolkit, libigl and the Lean SDF sampler, driven by
`guest/fit/fit_driver` one phase per vmcall. `build.sh` builds it
(`cmake/fit.cmake`; `BUILD_FIT=0` skips it). It is not committed. This page
records its hash and what it links, plus one smoke run in Godot.

This is not the Gate 6 guest-vs-native comparison over all four phases; that
is the next step (see the parent README).

## Result

**Build: pass.** **Smoke: pass**, with two guest-only bugs found and fixed.
**Phase 0 vs native: diverges after iteration 7.** It is chaotic from there,
as the parent README warns.

### The ELF (`link.log`, `link_check.sh`)

- **Size and hash:** 6,936,208 bytes, sha256 `f11ce4ca4c12ddcf904633817b0f97bee9bd17d2a5087b46fe91890373dabcdf`.
  - `.text` is 3.17 MB and `.eh_frame` plus `.gcc_except_table` 0.73 MB.
  - Not stripped: 16,406 symbols.
  - A full rebuild with `-Werror=absolute-value` (see below) gives the same bytes.
- **Compiled:** 206 objects, 27 of them PolyFEM, into 28 static libraries.
- **Present:** Eigen (2,192 symbols), polysolve (813), polyfem (668),
  ipc (964), igl (458), spdlog (1,010), nlohmann (481), `fit::FitDriver`,
  `SdfGrid`, and `sdf_spline::hessian_batch` over the emit's `main_0_Thread`.
- **Absent:** oneTBB (no `tbb::detail`, `r1::` or tbbmalloc), OpenVDB, Imath,
  boost, filib, and any `__real_open` / `__real_fopen`.
- **The only `tbb::` symbols** are the stand-in's inline templates:
  `blocked_range`, `blocked_range2d`, `enumerable_thread_specific` and
  `serial_detail::one_slot`.
- **Arch:** the merged attribute is sandbox-api's rv64gcv (from main.cpp and
  the API). Every solver target builds at `-march=rv64gc -ffp-contract=off`
  with `EIGEN_DONT_VECTORIZE`, as fit_native's guest numerics.

### The TBB stand-in (`guest/fit/tbb_serial`, `cmake/tbb_serial.cmake`)

ipc-toolkit and scalable-ccd link `TBB::tbb` unconditionally. PolyFEM, built
with `POLYFEM_THREADING=NONE`, calls no TBB at all. `TBB::tbb` is now an
INTERFACE target over a header-only, one-thread stand-in. Every onetbb recipe
returns early once that target exists, so oneTBB is never fetched.

These are the TBB names the compiled sources use, from the headers they
include:

| header | names | used by |
|---|---|---|
| `blocked_range.h` | `blocked_range<T>` (incl. CTAD) | ipc-toolkit, scalable-ccd |
| `blocked_range2d.h` | `blocked_range2d<T>`, `rows()`, `cols()` | ipc-toolkit brute_force, hash_grid |
| `parallel_for.h` | `parallel_for(range, body)`, `parallel_for(first, last, f)` | ipc-toolkit, scalable-ccd |
| `enumerable_thread_specific.h` | `enumerable_thread_specific<T>`: default, exemplar and args constructors; `local()`; `combine(f)`; const iteration | ipc-toolkit (potentials, collisions, broad phases, `merge_thread_local`), scalable-ccd |
| `parallel_sort.h` | `parallel_sort(first, last[, comp])` | ipc-toolkit hash_grid, collisions; scalable-ccd sort_and_sweep |
| `global_control.h` | `global_control(max_allowed_parallelism, n)` | fit_native, scalable-ccd |
| `info.h` | `info::default_concurrency()` | scalable-ccd |
| `task_arena.h` | `this_task_arena::current_thread_index()` | scalable-ccd |

- **Provided for completeness, unused here:** `parallel_reduce` (both forms),
  `combinable`, the partitioners and `task_arena`.
- **Not provided:** `concurrent_vector`, which nothing uses.
- **Why the stand-in gives oneTBB's result:** oneTBB on one thread runs the
  body on sub-ranges from left to right. The bodies here accumulate element by
  element into a single thread-local, so one call over the whole range gives
  the same bits.
- **Checked natively:** `fit_native` built with `FIT_TBB=serial` (the default
  for sdfgrid + guest numerics) reproduces its oneTBB f64 run bitwise.
  - `garment_final.f64` has sha256 ab9fb211…, and the `.obj` is identical.
  - `phases.tsv` is identical except for the wall and CPU columns.
  - All 184 Newton trace lines match except for timings.
  - Totals: 177 Newton (24/36/50/67), energy 0.0014288103789395555.
  - The oneTBB build has 1,485 `tbb::detail`/`r1::` symbols; the stand-in
    build has 0.
  - Evidence: `../foxgirl/run-sdf-kernel-tbbserial-f64.log`,
    `../foxgirl/phases-sdf-kernel-tbbserial-f64.tsv` and `../foxgirl/compare.log`.

### I/O counters

The link wraps `open`, `open64`, `openat`, `openat64`, `fopen` and `fopen64`
(`fit_tools.cpp`). Each call is counted, keeps its path, and is refused with
EACCES.

- **Probe:** `fopen`, `ifstream` and `open` are refused (errno 13), and 3 of 3
  are counted with their paths.
- **In the solve:** `fit_begin` and phase 0 make 0 attempts.

### Smoke (`smoke.txt`, `smoke.log`, `project/gate_fit_smoke.gd`)

Godot 4.7.2, `--rendering-driver vulkan --xr-mode off`, driven through
main.gd's no-argument wrappers.

| step | result |
|---|---|
| `fit_probe_exceptions` | PASS: typed catch through two `std::function` |
| `fit_probe_io` | PASS: all refused, 3/3 counted |
| `fit_probe_ldlt` | PASS: polysolve `Eigen::SimplicialLDLT` through the embedded spec, n = 200, relative residual 1.6e-16 |
| `fit_fixture_foxgirl` | 203 ms: 5,128 v / 10,171 f body, 2,682 v / 5,220 f garment, 15 joints / 14 bones, no-fit 2,005 (`project/util/obj_io.gd`) |
| `fit_begin` | 3.0 s. Normalisation bitwise equal to native f32 (target_scale 0.43371484168229041, center, source_scale). Heap 11.4 MiB, 79,255 live chunks |
| `fit_step` (phase 0, worker Thread) | 596 s (native: 28.9 s, about 21×). The main thread ran 35,840 frames meanwhile. 34 Newton, energy 0.0014895000331425965. Heap 122.9 MiB, io_attempts 0 |
| `fit_check` | none |
| `fit_preview` | equals `fit_result` |
| `fit_sdf` | 755 of 2,682 garment vertices inside; values −0.0100 … 0.2518 in solve units (−1 … 25.2 voxels) |

- **Repeatable:** two runs of phase 0 gave the same Newton count and bitwise
  the same energy.
- **Two formatting slips in this run's `smoke.txt`:**
  - The "phase 0 vs native" line printed its raw format string, because
    GDScript has no `%g` or `%e`. The numbers are in the phase 0 line above
    it. The script is fixed.
  - `fit_sdf` labelled world units as voxels. The wrapper now divides by the
    voxel size.

### What the smoke found (fixed)

1. **The guest heap caps live allocations.** `Sandbox.allocations_max`
   defaults to 10,000, and `fit_begin` failed with `Too many arena chunks
   (data: 10000)` inside an ipc-toolkit `robin_set` (79,255 chunks are live
   after begin). main.gd sets 4,000,000 before `program=`. (First run:
   `C:/b/ido6-smoke-run1.*`.)
2. **`abs(double)` meant `int abs` in the guest.**
   - **Where:** three unqualified calls in cloth-fit
     (`CurveCenterTargetForm.cpp` bone selection and
     `CurveConstraintForm.cpp` symmetric-vertex error).
   - **Why the builds differ:** the guest's glibc + libstdc++ bind them to
     C's `int abs`. llvm-mingw's libc++ binds them to the double overload.
   - **Effect in the guest:** no bone passed the 60° test, so the curve-center
     targets came from an uninitialised `bones` vector (bones "39 - 87" of a
     15-joint skeleton). Phase 0 started at f₀ 58.56 instead of 53.62, and
     ended after 4 Newton iterations at energy 4.94.
   - **The fix:** `std::abs`, and `bones` zeroed.
   - **Native is unchanged:** fit_native gives the same ab9fb211… after the
     fix.
   - **Guard:** `-Werror=absolute-value` now applies to every fit target.
     Two occurrences off the solve path stay warnings: finite-diff's
     `compare_gradient` log message, and polysolve's `verify_gradient` (spec
     default `apply_gradient_fd: "None"`).

### Phase 0 against native (`trace-vs-native.txt`)

The comparison is with native fit_native, f32 inputs
(`../foxgirl/run-sdf-guest-f32.log`).

- **What matches:** the first 39 trace lines are identical. That covers
  setup, the normalisation, the curve-center bones, f₀ = 53.6213, Newton
  iterations 0–7, and three IPC min distances printed to 16–17 digits.
- **First difference:** the min distance after iteration 7, at relative
  2e-13.
- **Where it ends:** phase 0 finishes at 34 Newton and energy 0.00148950,
  against native's 41 and 0.00033679.

This misses the guest-vs-native bound (±2 Newton, 1e-6 relative) that the
parent README proposes. What remains is a last-bit difference, which libm
(riscv glibc against ucrt) or the GDScript float parse of the OBJs (against
strtod) could cause. Which one it is has not been isolated.

The parent README's rule applies. The full four-phase guest run is judged by
the native-vs-upstream five-part criterion. Its log says so; the bound is not
loosened silently.

## Reproduce

```sh
BUILD_DIR=C:/b/ido6-guest bash build.sh                          # fit.elf + sha256
bash gates/6-fit/elf/link_check.sh                               # link.log
godot --path project --headless --import                         # first run after a new ELF
godot --path project --script gate_fit_smoke.gd --rendering-driver vulkan --xr-mode off
FIT_BUILD=C:/b/fit-native-tbbs FIT_SDF=sdfgrid pixi run --manifest-path tools/native/pixi.toml bash tests/native/fit/build.sh
C:/b/fit-native-tbbs/fit_native.exe --out C:/b/fit-out-tbbs64 --io-probe --control-push --no-round
cmp C:/b/fit-out-tbbs64/garment_final.f64 C:/b/fit-out-sdfk64/garment_final.f64
```

`CPM_SOURCE_CACHE` defaults to `C:/b/cpm-native` (build.sh). The org forks
come from `.forks` (`tools/fit/prepare_forks.sh`, which build.sh runs).
