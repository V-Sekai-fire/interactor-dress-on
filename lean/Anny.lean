import Anny.SlangCodegen.Common
import Anny.SlangCodegen.Blend
import Anny.SlangCodegen.BlendBackward
import Anny.SlangCodegen.CsrGemv3
import Anny.SlangCodegen.Fk
import Anny.SlangCodegen.FkBackward
import Anny.SlangCodegen.Lbs
import Anny.SlangCodegen.LbsBackwardBone
import Anny.SlangCodegen.LbsBackwardBind
import Anny.SlangCodegen.VertResidual

/-!
# `Anny` — the ANNY body model's forward and backward kernels

`Anny.SlangCodegen.*` holds the differentiable half of ANNY's shape and
pose (NAVER's ANNY; the forward stages restate `Sinew.SlangCodegen.Pheno`
and `Sinew.SlangCodegen.Lbs` on the drape kernels' layout): blendshapes,
the sparse joint regressor, forward kinematics from 6D joint rotations,
skinning, and the vertex residual, each with its vector-Jacobian
product, so the in-guest L-BFGS-B (`guest/drape/lbfgsb.h`, the
`lean/Drape` kernels) can fit coeffs, rot6 and trans to a target mesh
the way `AnnyInverter` does in PyTorch. `lake exe emit_anny` writes them
for `kernels/anny/gen.sh`.

The chain, forward then backward:

    anny_blend          coeffs            -> vanny
    anny_csr_gemv3      M · vanny         -> jpos
    anny_fk             rot6, trans, jpos -> world, bone
    anny_lbs            vanny, bone       -> verts
    anny_vert_residual  verts − target    -> dverts
    anny_lbs_backward_bone  dverts        -> dbone
    anny_lbs_backward_bind  dverts        -> dvanny (overwrite)
    anny_fk_backward    dbone             -> drot6, dtrans, djpos
    anny_csr_gemv3      Mᵀ · djpos        -> dvanny (accumulate)
    anny_blend_backward dvanny            -> dcoeffs

Not differentiated: the bind rotations `rb` (`pheno_skeleton_fit`'s
Newton–Schulz / Kabsch pass is a stop-gradient here) and the phenotype →
coeffs interpolation (host scalars).

Every kernel module exports `shader`, `expected` and two `native_decide`
examples pinning `emit shader = expected` and the entry point name.
`Common` is shared statement generators, not a kernel.
-/
