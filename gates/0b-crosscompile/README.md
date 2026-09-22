# Gate 0B — does the heavy C++ cross-compile for riscv64?

**Result: PASS**, with one scheduled replacement and one decided swap.

Everything that must live inside the sandbox ELF was cross-compiled with the
exact toolchain Gate 0A proved (`riscv64-sysroot/toolchain.cmake`, scoop llvm
clang, lld). Build dirs are gitignored; the logs beside this file are the
evidence.

| tree | result | log |
|---|---|---|
| ggml, CPU backend (`GGML_VULKAN=OFF`) | clean — `libggml{,-base,-cpu}.a`, `elf64-littleriscv` | `ggml.log` |
| PMP subset (SCsub allow-list, `PMP_SCALAR_TYPE_64=1`) | clean — `libpmp_subset.a` | `cassie-thirdparty.log` |
| Geogram subset (SCsub globs + excludes, **plus `delaunay_2d.cpp`**) | clean — `libgeogram_subset.a` | `cassie-thirdparty.log` |
| MWT + Cassie's `delaunay_geogram.cpp` | blocked only by `#include "core/math/delaunay_2d.h"` | `cassie-thirdparty.log` |
| cloth-fit / PolyFEM stack (`POLYFEM_THREADING=NONE`, OpenVDB stubbed) | 179 of 184 TUs, 11 libs linked; all 5 failures are `FitForm.hpp:10 openvdb/openvdb.h` | `clothfit.log` |

## What the two remaining items are

**OpenVDB** is used in exactly one file, `FitForm.{hpp,cpp}`:
`meshToSignedDistanceField` to build a grid, `SplineSampler::sampleHessian`
(tricubic B-spline value + gradient + Hessian) to query it, and one
`volumeToMesh` that is a debug export. libigl is already linked, so the
replacement is `igl::signed_distance` into a dense grid plus a tricubic
sampler — roughly 200 lines and no new dependency. Memory is not an issue
even in the guest: `normalize_meshes()` scales the skeleton's longest axis to
2.0, so at `voxel_size 0.01` the foxgirl body is ~104×30×200 cells ≈ 5 MB.

**Godot's Delaunay2D.** Despite its name, Cassie's `delaunay_geogram.cpp` is,
per its own header comment, *"backed by Godot's built-in Delaunay2D and
Delaunay3D"* — two `Delaunay2D::triangulate` calls. SCsub excluded Geogram's
`delaunay_2d.cpp` only for that reason. Decision: use Geogram's. It is
un-excluded here and compiles; the call-site swap is port work. Beyond that
the wrapper's Godot coupling is three type tokens — `Vector<T>`,
`PackedVector2Array`, `PackedVector3Array` — all `std::vector`
(`cassie-thirdparty/shim/`).

## What cost time, so it does not again

- **`-DPOLYFEM_WITH_TBB=OFF` is not a flag.** The switch is
  `POLYFEM_THREADING={CPP,TBB,NONE}` (`CMakeLists.txt:76`). CMake said the
  other name was unused; the early runs were still defaulting to TBB. `NONE`
  is also the *right* answer here, not merely the convenient one — see Gate 0C.
- **Windows MAX_PATH broke polysolve's checkout, not the pin.** A tracked file
  showed ` D`, and `git checkout` said *Filename too long*, because the CPM
  cache sat under a deep temp path. `CPM_SOURCE_CACHE` now points at the
  project-local, gitignored `.cpm-cache/`.
- **`CMAKE_PROJECT_INCLUDE_BEFORE` fires on every `project()` call**, including
  each CPM subproject's. The OpenVDB stub (`stub-openvdb.cmake`) is guarded
  with `if(NOT TARGET ...)`; unguarded it collides with itself inside
  polysolve.
- **Cassie's Geogram subset is the SCsub, not Geogram's CMake.** Geogram's own
  tree would try graphics, Lua, TetGen (AGPL), Triangle (non-commercial) and
  parallel Delaunay, none of which Cassie uses; their failures would say
  nothing. `cassie-thirdparty/CMakeLists.txt` transcribes SCsub's globs and
  exclude lists and is therefore the spec of what a Cassie port has to carry.
- **Python on Windows cannot open `/c/...` paths.** A shim-writing step failed
  silently on that and configure then listed a nonexistent source. Use
  `C:/...`.

Not yet measured: peak sandbox heap on a real scene. It needs the OpenVDB
replacement first to link a runnable binary.
