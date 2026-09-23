import LeanSlang
import Ggml.SlangCodegen.Common

/-!
# `Ggml.SlangCodegen.Rope` — family K5: ROPE, NEOX mode, f32

ggml-cpu's `ggml_compute_forward_rope_flt` for `GGML_ROPE_TYPE_NEOX`
(`src/ggml-cpu/ops.cpp`), forward, f32 data, i32 positions, no frequency
factors; the packer (`guest/ggml-rd/ops/rope.cpp`) refuses every other
mode, a src2, and any other type.

**One thread per pair.** A row of `ne0` elements has `ne0 / 2` threads.
Thread `j < n_dims / 2` rotates the pair `(j, j + n_dims / 2)`; thread
`j >= n_dims / 2` copies the pair `(2j, 2j + 1)` (ggml-cpu's pass-through
of the channels past `n_dims`, which start at `n_dims = 2 (n_dims / 2)`).
Each thread reads the two elements it writes and no others, so the kernel
is safe in place (test-backend-ops runs `ggml_rope_ext_inplace`).

**The angle, as ggml-cpu computes it.** `ggml_rope_cache_init` starts from
`theta = (float) p` and multiplies by `theta_scale` once per pair, so pair
`j` has `theta_extrap = p * ts * ts ... (j times)`, rounded after every
product. The kernel runs the same loop (at most 63 products for
`n_dims = 128`), so `theta_extrap` is bit-identical to ggml-cpu's.
`rope_yarn` follows, line for line:

    theta_interp = freq_scale * theta_extrap
    theta        = theta_interp
    if ext_factor != 0:
        ramp_mix = rope_yarn_ramp(corr0, corr1, 2j) * ext_factor
        theta    = theta_interp * (1 - ramp_mix) + theta_extrap * ramp_mix
    cos_theta = cos(theta) * mscale,  sin_theta = sin(theta) * mscale
    rope_yarn_ramp(lo, hi, i0) = 1 - min(1, max(0, (i0/2 - lo) / max(0.001, hi - lo)))

then `dst[j] = x0 cos - x1 sin`, `dst[j + n/2] = x0 sin + x1 cos` with
`x0 = src[j]`, `x1 = src[j + n/2]`.

The per-op constants come from the packer, computed there with ggml-cpu's
own C expressions so they match it bit for bit whatever the GPU's
`pow`/`log` do: word 57 `theta_scale = powf(freq_base, -2.0f / n_dims)`,
58-59 `corr_dims` (`ggml_rope_yarn_corr_dims`), 60 `mscale` (`attn_factor`,
times `1 + 0.1 logf(1 / freq_scale)` when `ext_factor != 0`). Word 55 is
`ne0 / 2` (threads per row), 56 is `n_dims / 2`. `freq_scale` and
`ext_factor` are read from op_params (words 6 and 7, i.e. 43 and 44).
-/

namespace Ggml.SlangCodegen.Rope

open LeanSlang
open Ggml.SlangCodegen.Common

/-- Threads per work group. -/
def threadgroup : Nat := 256

/-! ## Derived words (the packer fills them) -/

/-- Threads per row: `ne0 / 2`. -/
def wPairs : Nat := wDerived + 2
/-- `n_dims / 2`. -/
def wHalf : Nat := wDerived + 3
/-- `theta_scale` (f32 bits). -/
def wThetaScale : Nat := wDerived + 4
/-- `corr_dims[0]`, `corr_dims[1]` (f32 bits). -/
def wCorr0 : Nat := wDerived + 5
def wCorr1 : Nat := wDerived + 6
/-- `mscale` (f32 bits). -/
def wMscale : Nat := wDerived + 7
/-- op_params words: `freq_scale` (6), `ext_factor` (7). -/
def wFreqScale : Nat := wOpParams + 6
def wExtFactor : Nat := wOpParams + 7

example : wPairs = 55 ∧ wHalf = 56 ∧ wThetaScale = 57 ∧ wCorr0 = 58 ∧ wCorr1 = 59 ∧ wMscale = 60
    ∧ wFreqScale = 43 ∧ wExtFactor = 44 := by native_decide
/-- 53 and 54 belong to `entry1D`'s grid; the rope words stay inside the
    derived block. -/
example : wThreads < wPairs ∧ wGroupsX < wPairs ∧ wMscale < wDerived + nDerived := by native_decide

def fl (x : Float) : SlangExpr := .litFloatExact x
def fsub (a b : SlangExpr) : SlangExpr := .bin "-" a b
def fdiv (a b : SlangExpr) : SlangExpr := .bin "/" a b
def uintD (n : String) (e : SlangExpr) : SlangStmt := .declInit (.scalar .uint) n e
def floatD (n : String) (e : SlangExpr) : SlangStmt := .declInit (.scalar .float) n e
def off (w : Nat) (i0 : SlangExpr) : SlangExpr := .call "off4" [u w, i0, v "i1", v "i2", v "i3"]

/-- The body after the 1-D prologue. -/
def body : List SlangStmt :=
  [ uintD "np" (pwN wPairs)
  , uintD "j" (urem (v "e") (v "np"))
  , uintD "r" (udiv (v "e") (v "np"))
  , uintD "i1" (urem (v "r") (pwN (wDst + tNe + 1)))
  , .assign (v "r") (udiv (v "r") (pwN (wDst + tNe + 1)))
  , uintD "i2" (urem (v "r") (pwN (wDst + tNe + 2)))
  , uintD "i3" (udiv (v "r") (pwN (wDst + tNe + 2)))
  , uintD "nh" (pwN wHalf)
  -- past n_dims: copy the pair (2j, 2j + 1)
  , .ifNoElse (.bin ">=" (v "j") (v "nh"))
      [ uintD "c0" (mul (u 2) (v "j"))
      , floatD "y0" (.index (v "s0") (off (wSrc 0) (v "c0")))
      , floatD "y1" (.index (v "s0") (off (wSrc 0) (add (v "c0") (u 1))))
      , .assign (.index (v "dst") (off wDst (v "c0"))) (v "y0")
      , .assign (.index (v "dst") (off wDst (add (v "c0") (u 1)))) (v "y1")
      , .ret none ]
  -- theta_extrap: ggml_rope_cache_init's running product, j steps
  , floatD "te" (.cast (.scalar .float)
      (.index (v "s1") (add (pwN (wSrc 1 + tOff)) (mul (v "i2") (pwN (wSrc 1 + tNb))))))
  , floatD "ts" (pfN wThetaScale)
  , .forCount "k" (u 0) (v "j") [ .assign (v "te") (mul (v "te") (v "ts")) ]
  -- rope_yarn
  , floatD "ext" (pfN wExtFactor)
  , floatD "ti" (mul (pfN wFreqScale) (v "te"))
  , floatD "th" (v "ti")
  , .ifNoElse (.bin "!=" (v "ext") (fl 0.0))
      [ floatD "lo" (pfN wCorr0)
      , floatD "y" (fdiv (fsub (.cast (.scalar .float) (v "j")) (v "lo"))
          (.call "max" [fl 0.001, fsub (pfN wCorr1) (v "lo")]))
      , floatD "rm" (mul (fsub (fl 1.0) (.call "min" [fl 1.0, .call "max" [fl 0.0, v "y"]])) (v "ext"))
      , .assign (v "th") (add (mul (v "ti") (fsub (fl 1.0) (v "rm"))) (mul (v "te") (v "rm"))) ]
  , floatD "ms" (pfN wMscale)
  , floatD "ct" (mul (.call "cos" [v "th"]) (v "ms"))
  , floatD "st" (mul (.call "sin" [v "th"]) (v "ms"))
  -- rotate the pair (j, j + n_dims / 2)
  , uintD "j1" (add (v "j") (v "nh"))
  , floatD "x0" (.index (v "s0") (off (wSrc 0) (v "j")))
  , floatD "x1" (.index (v "s0") (off (wSrc 0) (v "j1")))
  , .assign (.index (v "dst") (off wDst (v "j")))
      (fsub (mul (v "x0") (v "ct")) (mul (v "x1") (v "st")))
  , .assign (.index (v "dst") (off wDst (v "j1")))
      (add (mul (v "x0") (v "st")) (mul (v "x1") (v "ct"))) ]

/-- f32 data in s0 and dst, i32 positions in s1; s2 unused. -/
def ropeNeoxF32 : SlangShaderModule :=
  kernelModule .float .int .uint .float
    [fnPw, fnPf, fnOff4]
    (entry1D threadgroup body)

/-- The kernels this module contributes, by their kernels.txt names. -/
def kernels : List (String × SlangShaderModule) :=
  [ ("rope_neox_f32", ropeNeoxF32) ]

def expectedRopeNeox : String :=
"struct Slot {
  uint base;
  uint pad0;
  uint pad1;
  uint pad2;
};

[[vk::binding(0, 0)]]
StructuredBuffer<uint> params;
[[vk::binding(1, 0)]]
RWStructuredBuffer<float> s0;
[[vk::binding(2, 0)]]
RWStructuredBuffer<int> s1;
[[vk::binding(3, 0)]]
RWStructuredBuffer<uint> s2;
[[vk::binding(4, 0)]]
RWStructuredBuffer<float> dst;
[[vk::binding(0, 1)]]
ConstantBuffer<Slot> slot;

uint pw(uint k) {
  return params[(slot.base + k)];
}

float pf(uint k) {
  return asfloat(pw(k));
}

uint off4(uint w, uint i0, uint i1, uint i2, uint i3) {
  return ((((pw((w + 8u)) + (i0 * pw((w + 4u)))) + (i1 * pw((w + 5u)))) + (i2 * pw((w + 6u)))) + (i3 * pw((w + 7u))));
}

[shader(\"compute\")] [numthreads(256, 1, 1)]
void main(uint3 gid : SV_GroupID, uint3 lid : SV_GroupThreadID) {
  uint e = ((((gid.y * pw(54u)) + gid.x) * 256u) + lid.x);
  if ((e >= pw(53u))) {
    return;
  }
  uint np = pw(55u);
  uint j = (e % np);
  uint r = (e / np);
  uint i1 = (r % pw(2u));
  r = (r / pw(2u));
  uint i2 = (r % pw(3u));
  uint i3 = (r / pw(3u));
  uint nh = pw(56u);
  if ((j >= nh)) {
    uint c0 = (2u * j);
    float y0 = s0[off4(10u, c0, i1, i2, i3)];
    float y1 = s0[off4(10u, (c0 + 1u), i1, i2, i3)];
    dst[off4(1u, c0, i1, i2, i3)] = y0;
    dst[off4(1u, (c0 + 1u), i1, i2, i3)] = y1;
    return;
  }
  float te = float(s1[(pw(27u) + (i2 * pw(23u)))]);
  float ts = pf(57u);
  for (uint k = 0u; k < j; ++k) {
    te = (te * ts);
  }
  float ext = pf(44u);
  float ti = (pf(43u) * te);
  float th = ti;
  if ((ext != 0.0f)) {
    float lo = pf(58u);
    float y = ((float(j) - lo) / max(0.001f, (pf(59u) - lo)));
    float rm = ((1.0f - min(1.0f, max(0.0f, y))) * ext);
    th = ((ti * (1.0f - rm)) + (te * rm));
  }
  float ms = pf(60u);
  float ct = (cos(th) * ms);
  float st = (sin(th) * ms);
  uint j1 = (j + nh);
  float x0 = s0[off4(10u, j, i1, i2, i3)];
  float x1 = s0[off4(10u, j1, i1, i2, i3)];
  dst[off4(1u, j, i1, i2, i3)] = ((x0 * ct) - (x1 * st));
  dst[off4(1u, j1, i1, i2, i3)] = ((x0 * st) + (x1 * ct));
}"

example : LeanSlang.emit ropeNeoxF32 = expectedRopeNeox := by native_decide
example : ropeNeoxF32.entryPointName = "main" := by native_decide

end Ggml.SlangCodegen.Rope
