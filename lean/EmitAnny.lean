import LeanSlang
import Anny

/-!
# `emit_anny` — write the ANNY forward and backward kernels as Slang

For each `Anny.SlangCodegen.*` kernel writes `<outDir>/<name>.slang`;
`kernels/anny/gen.sh` takes it from there (slangc cpp + spirv, binding
table, embedded SPIR-V).

    lake exe emit_anny /path/to/output/dir
-/

open LeanSlang

private def kernels : List (String × SlangShaderModule) :=
  [ ("anny_blend",             Anny.SlangCodegen.Blend.shader)
  , ("anny_blend_backward",    Anny.SlangCodegen.BlendBackward.shader)
  , ("anny_csr_gemv3",         Anny.SlangCodegen.CsrGemv3.shader)
  , ("anny_fk",                Anny.SlangCodegen.Fk.shader)
  , ("anny_fk_backward",       Anny.SlangCodegen.FkBackward.shader)
  , ("anny_lbs",               Anny.SlangCodegen.Lbs.shader)
  , ("anny_lbs_backward_bone", Anny.SlangCodegen.LbsBackwardBone.shader)
  , ("anny_lbs_backward_bind", Anny.SlangCodegen.LbsBackwardBind.shader)
  , ("anny_vert_residual",     Anny.SlangCodegen.VertResidual.shader)
  ]

def main (args : List String) : IO UInt32 := do
  let outDir := args.headD "."
  IO.FS.createDirAll outDir
  for (name, m) in kernels do
    let path := outDir ++ "/" ++ name ++ ".slang"
    IO.FS.writeFile path (LeanSlang.emit m ++ "\n")
    IO.println s!"wrote {path}"
  return 0
