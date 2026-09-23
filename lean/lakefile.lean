import Lake
open Lake DSL

package Cloth where

-- Every dependency is pinned in a V-Sekai-fire repo or fork
-- (interactor-dress-on AGENTS.md rule 1).
require LeanSlang from git
  "https://github.com/V-Sekai-fire/contract-lean-slang.git" @ "e0e96da4ba8116a3dcfd860f6732d29eee13fe24"

-- Property testing for the AVBD specs (Cloth.Avbd.*). The existing
-- `native_decide` examples pin single fixtures; plausible quantifies
-- the same invariants over generated meshes, which is where a role or
-- offset bug would actually show up. Pinned to v4.30.0, matching this
-- project's lean-toolchain, which was bumped to v4.30.0 precisely so
-- plausible-witness-dag's ladder could be adopted below.
require plausible from git
  "https://github.com/V-Sekai-fire/plausible" @ "v4.30.0"

@[default_target] lean_lib Cloth where

lean_exe emit_shaders where
  root := `EmitShaders

-- Gate 0F probe kernels (interactor-dress-on gates/0f-runtime).
lean_lib Probes

lean_exe emit_probes where
  root := `EmitProbes

-- Iterative-deepening witness search over the plausible ladder. Shares
-- the `Level` shape (walkSteps / finBound / numInst) that witness-cpp
-- mirrors on the C++ side, so a property stated here and a property
-- stated there escalate the same way.
require «plausible-witness-dag» from git
  "https://github.com/V-Sekai-fire/plausible-witness-dag" @ "160b94c9c6eed3bb9ebffce919fc6f989dcafba8"

lean_exe csr_falsify where
  root := `CsrFalsify
