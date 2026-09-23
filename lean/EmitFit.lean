import LeanSlang
import Fit

/-!
# `emit_fit` — write fit.elf's kernels as Slang

For each `Fit.SlangCodegen.*` kernel, writes `<outDir>/<name>.slang`.
`kernels/fit/gen.sh` takes it from there (slangc cpp, committed; slangc
spirv, validation only).

    lake exe emit_fit /path/to/output/dir
-/

open LeanSlang

private def kernels : List (String × SlangShaderModule) :=
  [ ("sdf_spline_hessian", Fit.SlangCodegen.SdfSplineHessian.shader)
  , ("similarity_hessian_block", Fit.SlangCodegen.SimilarityHessianBlock.shader)
  , ("project_psd12", Fit.SlangCodegen.ProjectPsd12.shader)
  , ("csr_gather_df32", Fit.SlangCodegen.CsrGatherDf32.shader)
  , ("swept_aabb", Fit.SlangCodegen.SweptAabb.shader)
  , ("box_pair", Fit.SlangCodegen.BoxPair.shader) ]

def main (args : List String) : IO UInt32 := do
  let outDir := args.headD "."
  IO.FS.createDirAll outDir
  for (name, m) in kernels do
    let path := outDir ++ "/" ++ name ++ ".slang"
    IO.FS.writeFile path (LeanSlang.emit m ++ "\n")
    IO.println s!"wrote {path}"
  return 0
