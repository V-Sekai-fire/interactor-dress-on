import LeanSlang
import Ggml.SlangCodegen.Common
import Ggml.SlangCodegen.Unary

/-!
# `Ggml.SlangCodegen.Glu` — ggml GLU, op GEGLU_ERF, f32

`ggml_geglu_erf(a)`, `ggml_geglu_erf_swapped(a)` and
`ggml_geglu_erf_split(a, b)` (`GGML_OP_GLU` with `GGML_GLU_OP_GEGLU_ERF`
in op_params word 0 and the swapped flag in word 1): per row, the
activation of one half of the row times the other half,

    dst[i0, i1, i2, i3] = gelu_erf(x) * g
    x = s0[i0 + xo, i1, i2, i3]
    g = split ? s1[i0, i1, i2, i3] : s0[i0 + go, i1, i2, i3]

where in the non-split form `nc = dst.ne0 = a.ne0 / 2` and `(xo, go)` is
`(0, nc)`, or `(nc, 0)` when swapped; in the split form (`b` given, the
same shape as `a`) both halves are whole tensors and `(xo, go) = (0, 0)`.
The packer (`guest/ggml-rd/ops/glu.cpp`) writes `xo`, `go` and the split
flag as derived words 55, 56 and 57. One thread per destination element,
every operand strided (ggml-cpu, `ggml_compute_forward_geglu_erf_f32`,
walks rows by nb[1] and asserts `ggml_is_contiguous_1`).

`gelu_erf` is `Unary`'s (`0.5 x (1 + erf(x / sqrt 2))` with the A&S
7.1.26 erf, `ggml_vec_geglu_erf_f32`'s association
`0.5f * xi * (1.0f + erff(xi*SQRT_2_INV)) * g[i]`), so the result matches
ggml-cpu within test-backend-ops' NMSE as GELU_ERF does, not bit for bit.
-/

namespace Ggml.SlangCodegen.Glu

open LeanSlang
open Ggml.SlangCodegen.Common
open Ggml.SlangCodegen.Unary

/-- Threads per work group. -/
def threadgroup : Nat := 256

/-- Derived words: 55 = x's column offset in s0, 56 = the gate's, 57 =
    1 when the gate is s1 (the split form). -/
def wXOff : Nat := 55
def wGOff : Nat := 56
def wSplit : Nat := 57

example : wDerived ≤ wXOff ∧ wSplit < wDerived + nDerived ∧ wXOff ≠ wThreads ∧ wXOff ≠ wGroupsX := by
  native_decide

private def U : SlangType := .scalar .uint
private def F : SlangType := .scalar .float

def body : List SlangStmt :=
  [ .decl U "i0", .decl U "i1", .decl U "i2", .decl U "i3"
  , .expr (.call "unravel4" [v "e", u wDst, v "i0", v "i1", v "i2", v "i3"])
  , .declInit U "d" (.call "off4" [u wDst, v "i0", v "i1", v "i2", v "i3"])
  , .declInit F "x" (.index (v "s0") (.call "off4" [u (wSrc 0), add (v "i0") (pwN wXOff), v "i1", v "i2", v "i3"]))
  , .decl F "g"
  , .ifThen (.bin "!=" (pwN wSplit) (u 0))
      [ .assign (v "g") (.index (v "s1") (.call "off4" [u (wSrc 1), v "i0", v "i1", v "i2", v "i3"])) ]
      [ .assign (v "g") (.index (v "s0") (.call "off4" [u (wSrc 0), add (v "i0") (pwN wGOff), v "i1", v "i2", v "i3"])) ]
  , .assign (.index (v "dst") (v "d")) (mul (.call "gelu_erf" [v "x"]) (v "g")) ]

def gegluErfF32 : SlangShaderModule :=
  kernelModule .float .float .uint .float
    [fnPw, fnUnravel4, fnOff4, fnErf, fnGeluErf]
    (entry1D threadgroup body)

/-- The kernels this module contributes, by their kernels.txt names. -/
def kernels : List (String × SlangShaderModule) :=
  [ ("geglu_erf_f32", gegluErfF32) ]

/-! ## Pins -/

/-- The whole kernel after the shared helpers (`Unary.expectedSilu`'s
    prologue up to `off4`, then erf and gelu_erf as `gelu_erf_f32` has
    them): the entry is pinned in full. -/
def expectedEntry : String :=
"[shader(\"compute\")] [numthreads(256, 1, 1)]
void main(uint3 gid : SV_GroupID, uint3 lid : SV_GroupThreadID) {
  uint e = ((((gid.y * pw(54u)) + gid.x) * 256u) + lid.x);
  if ((e >= pw(53u))) {
    return;
  }
  uint i0;
  uint i1;
  uint i2;
  uint i3;
  unravel4(e, 1u, i0, i1, i2, i3);
  uint d = off4(1u, i0, i1, i2, i3);
  float x = s0[off4(10u, (i0 + pw(55u)), i1, i2, i3)];
  float g;
  if ((pw(57u) != 0u)) {
    g = s1[off4(19u, i0, i1, i2, i3)];
  } else {
    g = s0[off4(10u, (i0 + pw(56u)), i1, i2, i3)];
  }
  dst[d] = (gelu_erf(x) * g);
}"

/-- The whole text: silu_f32's declarations and shared helpers with s1
    declared float (the split gate), then erf and gelu_erf exactly as
    gelu_erf_f32 has them, then the entry above. -/
example : LeanSlang.emit gegluErfF32 =
    ((expectedSilu.splitOn "float silu(float x)").headD "").replace
        "RWStructuredBuffer<uint> s1;" "RWStructuredBuffer<float> s1;"
      ++ LeanSlang.emitFunction fnErf ++ "\n\n" ++ LeanSlang.emitFunction fnGeluErf ++ "\n\n"
      ++ expectedEntry := by native_decide

example : gegluErfF32.entryPointName = "main" := by native_decide

end Ggml.SlangCodegen.Glu
