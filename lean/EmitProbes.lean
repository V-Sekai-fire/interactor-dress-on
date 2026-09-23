import LeanSlang
import Cloth.SlangCodegen
import Probes

/-!
# `emit_probes` — the Gate 0F probe kernels, one `.slang` per kernel

    lake exe emit_probes /path/to/output/dir

`saxpby` is the AVBD kernel itself (`Cloth.SlangCodegen.Saxpby`), listed here
too so the probe set is emitted by one command.
-/

open LeanSlang

private def kernels : List (String × SlangShaderModule) :=
  [ ("saxpby",      Cloth.SlangCodegen.Saxpby.shader)
  , ("half_load",   Probes.HalfLoad.shader)
  , ("probe_add",   Probes.Set0.add)
  , ("probe_scale", Probes.Set0.scale)
  , ("probe_acc",   Probes.Set0.acc)
  ]

def main (args : List String) : IO UInt32 := do
  let outDir := args.headD "."
  IO.FS.createDirAll outDir
  for (name, m) in kernels do
    let path := outDir ++ "/" ++ name ++ ".slang"
    IO.FS.writeFile path (LeanSlang.emit m ++ "\n")
    IO.println s!"wrote {path}"
  return 0
