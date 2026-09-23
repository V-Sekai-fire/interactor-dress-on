import LeanSlang
import Ggml

/-!
# `emit_ggml` — Lake exe writing the ggml-rd kernels and their params header

    lake exe emit_ggml <outDir> [<params-header-path>]

Writes `<outDir>/<name>.slang` for every `Ggml.kernels` and `Ggml.controls`
entry, and the C++
view of the params word layout (`Common.paramsHeader`) to
`<params-header-path>` when given. `kernels/ggml/gen.sh` runs it and
copies the kernels named in `kernels/ggml/kernels.txt`.
-/

def main (args : List String) : IO UInt32 := do
  let outDir := args.headD "."
  IO.FS.createDirAll outDir
  for (name, m) in Ggml.kernels ++ Ggml.controls do
    let path := outDir ++ "/" ++ name ++ ".slang"
    IO.FS.writeFile path (LeanSlang.emit m ++ "\n")
    IO.println s!"wrote {path}"
  match args with
  | [_, header] =>
    IO.FS.writeFile header Ggml.SlangCodegen.Common.paramsHeader
    IO.println s!"wrote {header}"
  | _ => pure ()
  return 0
