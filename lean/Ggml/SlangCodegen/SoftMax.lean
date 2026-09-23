import LeanSlang
import Ggml.SlangCodegen.Common
import Ggml.SlangCodegen.Rows

/-!
# `Ggml.SlangCodegen.SoftMax` — ggml SOFT_MAX, f32, with scale (family K4)

Row kernels on `Rows` (one work group per row, grid-strided, tree-reduced
in group-shared memory; 256 threads, or 64 in `soft_max_f32_t64`, which the
packer picks for rows of at most 1024) and their Serial siblings for the
cpp target. ggml-cpu's arithmetic, per row:

    w   = x · scale                      scale = op_params[0] (word 37)
    mx  = max w                          (from -inf)
    sum = Σ exp(w - mx)
    y   = exp(w - mx) · (1 / sum)

The exponential is recomputed in the last pass rather than stored and read
back: three reads and one write per element instead of two reads and two
writes plus one read, and dst is never read (so in place it is as safe as
every other `Rows` map). No mask (src1), no sinks (src2) and max_bias = 0:
the census (Skin-Tokens, Pixal3D/DINO) has none of them, and the packer
(`guest/ggml-rd/ops/soft_max.cpp`) rejects them.
-/

namespace Ggml.SlangCodegen.SoftMax

open LeanSlang
open Ggml.SlangCodegen.Common
open Ggml.SlangCodegen.Rows

def x : SlangExpr := v "x"
def w : SlangExpr := mul x (v "scale")
def negInf : SlangExpr := fLit ((-1.0 : Float) / 0.0)

def body (tg : Nat) (var : Variant) : List SlangStmt :=
  [ .declInit fl "scale" (pfN wOpParams) ] ++
  reduce tg var "mx" .max negInf w ++
  reduce tg var "sume" .sum (fLit 0.0) (.call "exp" [sub w (v "mx")]) ++
  [ .declInit fl "inv" (fdiv (fLit 1.0) (v "sume")) ] ++
  mapRow tg var (mul (.call "exp" [sub w (v "mx")]) (v "inv"))

def softMax (tg : Nat) (var : Variant) : SlangShaderModule :=
  rowModule tg var [fnPw, fnPf, fnOff4] (body tg var)

def softMaxF32 : SlangShaderModule := softMax threadgroup .par
def softMaxF32Serial : SlangShaderModule := softMax threadgroup .ser
/-- 64 threads per row (the packer picks by row length, ops/rows.h). -/
def softMaxF32T64 : SlangShaderModule := softMax threadgroupShort .par
def softMaxF32T64Serial : SlangShaderModule := softMax threadgroupShort .ser

/-- The kernels this module contributes, by their kernels.txt names. -/
def kernels : List (String × SlangShaderModule) :=
  [ ("soft_max_f32", softMaxF32)
  , ("soft_max_f32_serial", softMaxF32Serial)
  , ("soft_max_f32_t64", softMaxF32T64)
  , ("soft_max_f32_t64_serial", softMaxF32T64Serial) ]

/-! ## Pins -/

/-- -inf is a bit pattern, not a decimal. -/
example : emitExpr negInf = "asfloat(0xFF800000u)" := by native_decide

/-- The head of both reductions and the whole map, in the parallel kernel. -/
example : ((LeanSlang.emit softMaxF32).splitOn
"  float scale = pf(37u);
  float acc_mx = asfloat(0xFF800000u);
  uint i_mx = t;
  while ((i_mx < n)) {
    float x = s0[(xb + (i_mx * xs))];
    acc_mx = max(acc_mx, (x * scale));
    i_mx = (i_mx + 256u);
  }
  sh[t] = acc_mx;
  GroupMemoryBarrierWithGroupSync();
  if ((t < 128u)) {
    sh[t] = max(sh[t], sh[(t + 128u)]);
  }").length = 2 := by native_decide

example : ((LeanSlang.emit softMaxF32).splitOn
"  float mx = sh[0u];
  GroupMemoryBarrierWithGroupSync();
  float acc_sume = 0.0f;
  uint i_sume = t;
  while ((i_sume < n)) {
    float x = s0[(xb + (i_sume * xs))];
    acc_sume = (acc_sume + exp(((x * scale) - mx)));
    i_sume = (i_sume + 256u);
  }").length = 2 := by native_decide

example : (LeanSlang.emit softMaxF32).endsWith
"  float sume = sh[0u];
  GroupMemoryBarrierWithGroupSync();
  float inv = (1.0f / sume);
  uint j = t;
  while ((j < n)) {
    float x = s0[(xb + (j * xs))];
    dst[(yb + (j * ys))] = (exp(((x * scale) - mx)) * inv);
    j = (j + 256u);
  }
}" := by native_decide

example : (LeanSlang.emit softMaxF32Serial).endsWith
"  float sume = sh[0u];
  float inv = (1.0f / sume);
  for (uint j = 0u; j < n; ++j) {
    float x = s0[(xb + (j * xs))];
    dst[(yb + (j * ys))] = (exp(((x * scale) - mx)) * inv);
  }
}" := by native_decide

/-- The t64 kernel is the 256-thread one with 64 threads, 64 partials, a
    stride of 64 and the tree's 128 and 64 steps gone (in both reductions). -/
example : LeanSlang.emit softMaxF32T64 =
    (((((((LeanSlang.emit softMaxF32).replace
      "  if ((t < 128u)) {
    sh[t] = max(sh[t], sh[(t + 128u)]);
  }
  GroupMemoryBarrierWithGroupSync();
" "").replace
      "  if ((t < 64u)) {
    sh[t] = max(sh[t], sh[(t + 64u)]);
  }
  GroupMemoryBarrierWithGroupSync();
" "").replace
      "  if ((t < 128u)) {
    sh[t] = (sh[t] + sh[(t + 128u)]);
  }
  GroupMemoryBarrierWithGroupSync();
" "").replace
      "  if ((t < 64u)) {
    sh[t] = (sh[t] + sh[(t + 64u)]);
  }
  GroupMemoryBarrierWithGroupSync();
" "").replace
      "[numthreads(256, 1, 1)]" "[numthreads(64, 1, 1)]").replace "sh[256];" "sh[64];").replace " + 256u)" " + 64u)" := by
  native_decide

/-- Two reductions: 20 barriers in the kernel, none in its sibling. -/
example : ((LeanSlang.emit softMaxF32).splitOn "GroupMemoryBarrierWithGroupSync();").length - 1 = 20 ∧
    ((LeanSlang.emit softMaxF32Serial).splitOn "GroupMemoryBarrier").length = 1 ∧
    ((LeanSlang.emit softMaxF32Serial).splitOn "groupshared").length = 1 := by
  native_decide

end Ggml.SlangCodegen.SoftMax
