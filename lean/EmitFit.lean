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
  [ ("sdf_spline_hessian", Fit.SlangCodegen.SdfSplineHessian.shader) ]

def main (args : List String) : IO UInt32 := do
  let outDir := args.headD "."
  IO.FS.createDirAll outDir
  for (name, m) in kernels do
    let path := outDir ++ "/" ++ name ++ ".slang"
    IO.FS.writeFile path (LeanSlang.emit m ++ "\n")
    IO.println s!"wrote {path}"
  return 0
