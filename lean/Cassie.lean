import LeanSlang
import Cassie.CurveCasteljau
import Cassie.CurveGenerateBezier
import Cassie.CurveNewton
import Cassie.CurveRdp

/-!
# `Cassie` — the curvenet stage's Lean-emitted kernels

The four editing-pipeline kernels of the CASSIE Godot module
(V-Sekai-fire/entities-godot `modules/cassie/lean/CassieAvbd/Curve*.lean`
at c165a519d2, see `Cassie/CITATION.cff`), renamespaced `Cassie.*`.
`lake exe emit_cassie <dir>` writes them as Slang; `kernels/cassie/gen.sh`
lowers each to `slangc -target cpp` (the in-guest CPU path, called by
`vendor/cassie/src/solver/slang_dispatch/*_dispatch.cpp`) and
`slangc -target spirv` (validation only, for now).

The fifth kernel the curvenet dispatchers need, `spmv_df32`, is DiffCloth's
`Cloth.SlangCodegen.SpmvDf32` and comes from `lake exe emit_shaders`.
-/

namespace Cassie

open LeanSlang

/-- Per-target kernel pair, as the entities-godot `CassieAvbd.Codegen` keeps
    it. `gpu` is lowered by `slangc -target spirv`; `cpu` by
    `slangc -target cpp`. They differ only where the GPU module uses
    `groupshared` + `GroupMemoryBarrierWithGroupSync`, which slangc's CPU
    backend rejects (E36107): such a kernel pairs its GPU module with a
    serial sibling (as `DotReduce` pairs with `DotReduceSerial`). The four
    curve kernels are single-thread and use no groupshared, so each pairs
    with itself. -/
structure KernelPair where
  name : String
  gpu  : SlangShaderModule
  cpu  : SlangShaderModule

/-- The Cassie kernels, by the file names `kernels/cassie/kernels.txt` uses. -/
def kernels : List KernelPair :=
  [ ⟨"curve_casteljau",       CurveCasteljau.shader,      CurveCasteljau.shader⟩
  , ⟨"curve_generate_bezier", CurveGenerateBezier.shader, CurveGenerateBezier.shader⟩
  , ⟨"curve_newton",          CurveNewton.shader,         CurveNewton.shader⟩
  , ⟨"curve_rdp",             CurveRdp.shader,            CurveRdp.shader⟩ ]

example : kernels.map (·.name) =
    ["curve_casteljau", "curve_generate_bezier", "curve_newton", "curve_rdp"] := by
  native_decide
-- No curve kernel shares memory across threads, so none needs a serial CPU
-- sibling: the CPU module is the GPU module, text for text.
example : kernels.all (fun k => LeanSlang.emit k.gpu == LeanSlang.emit k.cpu) := by
  native_decide
example : kernels.all (fun k => k.gpu.groupShared.isEmpty) := by native_decide

end Cassie
