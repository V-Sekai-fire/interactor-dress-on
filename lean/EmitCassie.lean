import LeanSlang
import Cassie

/-!
# `emit_cassie` — Lake exe writing the Cassie kernels as Slang

For each `Cassie.kernels` pair, writes the GPU module to `<outDir>/<name>.slang`
and, only when the CPU module emits different text (a groupshared kernel with
a serial sibling), the CPU module to `<outDir>/<name>.cpu.slang`.
`kernels/cassie/gen.sh` lowers `<name>.cpu.slang` to cpp when it exists and
`<name>.slang` otherwise.

Usage:

    lake exe emit_cassie /path/to/output/dir
-/

def main (args : List String) : IO UInt32 := do
  let outDir := args.headD "."
  IO.FS.createDirAll outDir
  for k in Cassie.kernels do
    let gpu := LeanSlang.emit k.gpu
    let path := outDir ++ "/" ++ k.name ++ ".slang"
    IO.FS.writeFile path (gpu ++ "\n")
    IO.println s!"wrote {path}"
    let cpu := LeanSlang.emit k.cpu
    if cpu != gpu then
      let cpath := outDir ++ "/" ++ k.name ++ ".cpu.slang"
      IO.FS.writeFile cpath (cpu ++ "\n")
      IO.println s!"wrote {cpath}"
  return 0
