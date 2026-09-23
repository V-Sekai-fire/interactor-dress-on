import Fit.SdfSplineHessian
import Fit.Df32
import Fit.Hess12
import Fit.SimilarityHessianBlock
import Fit.ProjectPsd12
import Fit.CsrGatherDf32
import Fit.SweptAabb
import Fit.BoxPair

/-!
# `Fit` — fit.elf's Lean-emitted kernels

`Fit.SlangCodegen.*` holds the kernels of fit.elf (interactor-dress-on
plan, Cut 6), the dress-on stage built from cloth-fit (PolyFEM). The
first is `SdfSplineHessian`, the tricubic B-spline sampler FitForm
evaluates on its brick-grid SDF in place of OpenVDB's
`SplineSampler::sampleHessian`. `lake exe emit_fit` writes them for
`kernels/fit/gen.sh`.

Every kernel module exports `shader`, `expected` and two `native_decide`
examples pinning `emit shader = expected` and the entry point name.
-/
