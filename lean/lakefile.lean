import Lake
open Lake DSL

package Cloth where

-- Every dependency is pinned in a V-Sekai-fire repo or fork
-- (interactor-dress-on AGENTS.md rule 1).
require LeanSlang from git
  "https://github.com/V-Sekai-fire/contract-lean-slang.git" @ "60532aef8ed70cc669ecab481182d0636c9e1ac3"

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

-- The curvenet stage's kernels (Cut 4): CASSIE's four editing-pipeline
-- kernels from entities-godot, see Cassie/CITATION.cff.
lean_lib Cassie

lean_exe emit_cassie where
  root := `EmitCassie

-- drape.elf's L-BFGS-B kernels (Cut 5). A default target, so a bare
-- `lake build` checks their native_decide pins as well as Cloth's.
@[default_target] lean_lib Drape

lean_exe emit_drape where
  root := `EmitDrape

-- fit.elf's kernels (Cut 6): the SDF spline sampler FitForm evaluates.
-- A default target, so a bare `lake build` checks their native_decide
-- pins as well as Cloth's.
@[default_target] lean_lib Fit

lean_exe emit_fit where
  root := `EmitFit

-- The ANNY body model's forward and backward kernels (blendshapes, joint
-- regressor, 6D forward kinematics, skinning, vertex residual), for the
-- in-guest L-BFGS-B fit. A default target, so a bare `lake build` checks
-- their native_decide pins.
@[default_target] lean_lib Anny

lean_exe emit_anny where
  root := `EmitAnny

-- The ggml-rd op kernels (Cut 3): Ggml.SlangCodegen.*, one family per
-- module, all on the fixed layout of Ggml.SlangCodegen.Common.
lean_lib Ggml

lean_exe emit_ggml where
  root := `EmitGgml

-- Iterative-deepening witness search over the plausible ladder. Shares
-- the `Level` shape (walkSteps / finBound / numInst) that witness-cpp
-- mirrors on the C++ side, so a property stated here and a property
-- stated there escalate the same way.
require «plausible-witness-dag» from git
  "https://github.com/V-Sekai-fire/plausible-witness-dag" @ "160b94c9c6eed3bb9ebffce919fc6f989dcafba8"

lean_exe csr_falsify where
  root := `CsrFalsify
