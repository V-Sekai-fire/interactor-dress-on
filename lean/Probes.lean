import Probes.HalfLoad
import Probes.Set0

/-!
# `Probes` — Gate 0F probe kernels (interactor-dress-on `gates/0f-runtime`)

The kernels `probes.elf` dispatches beyond the AVBD `saxpby`, stated here so
that every kernel it runs comes from Lean in this repo (AGENTS.md rule 2).
`lake exe emit_probes <dir>` writes them; `kernels/probes/gen.sh` compares
the output with the committed `kernels/probes/slang/`.
-/
