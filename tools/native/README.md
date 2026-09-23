# tools/native — the upstream cloth-fit oracle for Cut 6

Builds upstream cloth-fit's `PolyFEM_bin` natively on the Windows desk
(llvm-mingw 20260826, pixi for cmake/ninja/boost/tbb headers), unmodified,
plus `tools/fit/openvdb_dump` against the same configure. The source is
`vendor/cloth-fit` as vendored, exported by `git archive` from the subtree's
squash commit 812ceb26 into `CF_SRC` (default `C:/b/cf-src-d2bd59a6`): the
working `vendor/cloth-fit` carries the `fit.elf` adaptations (OpenVDB replaced
by a brick-grid SDF; its `CITATION.cff` lists them) and no longer builds
`openvdb_dump`. `fit.elf` is judged
against this build: guest vs a native same-code build to 1e-6, native vs this
upstream binary by Hausdorff within 1 voxel (plan, Cut 6).

```sh
tools/forks/fetch.sh                      # .forks/{json,polysolve,libigl,ipc-toolkit,openvdb}
CF_BUILD=C:/b/cf-nat pixi run --manifest-path tools/native/pixi.toml \
  bash tools/native/build_upstream.sh     # PolyFEM_bin openvdb_dump (runs fetch.sh itself)
pwsh tools/native/run_oracle.ps1 -Threads 1 -OutDir C:/b/cf-up-out1   # CF_BUILD picks the exe
```

`foxgirl_oracle.json` is `vendor/cloth-fit/garment-data/foxgirl_skirt/setup.json`
with `output.skip_frame = 1` (every Newton step saved) and its asset paths
relative to the repo root; PolyFEM_bin opens them relative to its working
directory, so `run_oracle.ps1` runs it from the repo root. The build stages
cloth-fit's JSON specs into `<build>/json-specs` and bakes that path into the
binary.

## Dependencies and rule 1

Five CPM packages come from V-Sekai-fire forks (`tools/forks.tsv`), fetched by
`tools/forks/fetch.sh` into the gitignored `.forks/<name>` and handed to CPM as
`-DCPM_<pkg>_SOURCE=.forks/<name>`, so the vendored recipes stay unedited and
CPM never fetches their upstream URLs:

| CPM name | fork, ref fetched | pinned rev |
|---|---|---|
| `polysolve` | V-Sekai-fire/polysolve, branch `garment` | ee553dc6 |
| `ipc-toolkit` | V-Sekai-fire/ipc-toolkit, branch `polyfem` (tip 66c8924e) | 993e7e0f |
| `openvdb` | V-Sekai-fire/openvdb, branch `sample` | c96cb069 |
| `libigl` | V-Sekai-fire/libigl, tag `v2.5.0` | fdaac01b |
| `nlohmann_json` | V-Sekai-fire/json, tag `v3.11.2`, sparse (`include/`, `single_include/`, `LICENSE.MIT`, `meson.build`), LF | bc889afb |

`fetch.sh` refuses a pinned rev that is not the fetched ref or an ancestor of
it (control: ipc-toolkit 993e7e0f against the fork's `main` is refused). The
five checkouts are byte-identical to the CPM sources the `C:/b/cf-up` oracle
build used (`diff -r`, `.git` excluded), except that the json fork also has
`single_include/nlohmann/json_fwd.hpp`, which the release `include.zip` lacks
and no recipe includes. json is sparse because the recipes download that zip,
which has no `CMakeLists.txt`; a full checkout would be `add_subdirectory`'d
and collide with the recipes' own `nlohmann_json` target.

**oneTBB is an oracle-only exception, not forked.** `onetbb.cmake` pins
`gh:oneapi-src/oneTBB@2021.9.0` (a00cc3b8) and builds it static; it serves
only PolyFEM's `POLYFEM_THREADING=TBB` in this native oracle. `fit.elf` builds
with `POLYFEM_THREADING=NONE` and never links it. `pixi.toml` pins
`tbb-devel = 2021.9.*` so the headers OpenVDB and polyfem compile against (the
conda include dir precedes oneTBB's own) are the runtime's release; the
`C:/b/cf-up` build resolved cloth-fit's `>=2021.9,<2022` to 2021.13 headers
over the 2021.9 static runtime. `PolyFEM_bin.exe` imports no TBB DLL.

**Still fetched from upstream by CPM (no org fork yet; rule 1 asks before
forking these in).** Revs as the oracle resolved them in `C:/b/cpm-native`:

| package | upstream | rev | in fit.elf? |
|---|---|---|---|
| eigen | gitlab.com/libeigen/eigen 3.4.0 | 3147391d | yes (fit.elf only, rule 3) |
| spdlog | gabime/spdlog 1.12.0 | 7e635fca | yes |
| json-spec-engine | geometryprocessing/json-spec-engine | 49f1a30f | yes |
| finite-diff | zfergus/finite-diff 1.0.2 | 60f9fee0 | yes (polyfem links it) |
| LBFGSpp | yixuan/LBFGSpp | 00f0f1bd | yes (polysolve) |
| tight-inclusion | Continuous-Collision-Detection/Tight-Inclusion | 5113d41f | yes (ipc-toolkit) |
| scalable-ccd | continuous-collision-detection/scalable-ccd | 60692d7c | yes (ipc-toolkit) |
| SimpleBVH | geometryprocessing/SimpleBVH | e1a93133 | yes (ipc-toolkit) |
| robin-map | Tessil/robin-map | d37a4100 | yes (ipc-toolkit) |
| abseil-cpp | abseil/abseil-cpp | c2435f83 | yes (ipc-toolkit) |
| filib | zfergus/filib | 1cd377a7 | no (`IPC_TOOLKIT_WITH_FILIB=OFF`, Gate 0F: fesetround ignored) |
| CLI11 | CLIUtils/CLI11 2.3.2 | 291c5878 | no (only `src/polyfem/main.cpp`, PolyFEM_bin) |
| Catch2 | catchorg/Catch2 3.4.0 | 6e79e682 | no (tests) |
| oneTBB | oneapi-src/oneTBB v2021.9.0 | a00cc3b8 | no (oracle-only, above) |
| libigl-predicates, triangle | libigl/* via libigl's own FetchContent | 488242f, 3ee6cac | not yet known |

## The oracle of record

Built once from a checkout of cloth-fit at d2bd59a6 (byte-identical to
`vendor/cloth-fit`) with this recipe's predecessor (sources from upstream CPM
URLs at the same revs, tbb-devel 2021.13 headers): `C:/b/cf-up/PolyFEM_bin.exe`.
Not rebuilt from this recipe yet; the only differences are the fork source
paths (same bytes), the tbb header release and `openvdb_dump` now linking
`-static`. This recipe configures clean (`CF_BUILD=C:/b/cf6c
CPM_SOURCE_CACHE=C:/b/cpm-native CF_CONFIGURE_ONLY=1`, 9.5 s): all five forks
are taken from `.forks/`, and both `PolyFEM_bin` and `openvdb_dump` link
`-static`.

| run | wall s | peak WS MB | Newton iters | final |
|---|---|---|---|---|
| 1 thread (`cf-up-out1`) | 961.2 | 940.4 | 246 logged (248 counting the two 50-limit AL solves as 50) | step_garment_252.obj |
| 1 thread repeat (`cf-up-out1b`) | 371.6 | 940.0 | same | bitwise identical to the first |
| 16 threads (`cf-up-out16`) | 1164.4 | 1157.8 | 249 | step_garment_253.obj |
| 16 threads repeat (`cf-up-out16b`) | 174.3 | 953.8 | 184 | step_garment_188.obj |

One thread is bitwise reproducible; sixteen are not: the two 16-thread runs
first differ at frame 2 (2.6e-14) and end 8.42 voxels apart at worst, 2.98
mean (voxel 0.01), and each ends 4.93-11.63 voxels from the 1-thread result
(mean 3.11-5.99). The 1-thread run is therefore the reference. Wall times
vary with desk load (961 vs 372 s for identical 1-thread runs).

OpenVDB SDF as FitForm builds it (`openvdb_dump`, voxel 0.01, band 1 in / 150
out): 46,523,937 active voxels, 97,637 leaves, 413,824,624 bytes; its
zero-isosurface re-extraction matches the run's `sdf.obj` bitwise (47,852
verts, 95,700 tris).
