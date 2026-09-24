import LeanSlang
import Ggml.SlangCodegen.Common

/-!
# `Ggml.SlangCodegen.Unary` — family K1: element-wise ops, f32

SILU, GELU (tanh form), GELU_ERF, SIGMOID, NEG, RELU (`GGML_OP_UNARY`),
SCALE (`x * s + b`), DIAG_MASK_INF, LEAKY_RELU and CLAMP. One thread per
destination element, the shape of `Binary`:

    e            = this thread's linear element (threads >= word 53 return)
    (i0..i3)     = unravel(e) over dst's ne (ggml order, dim 0 fastest)
    x            = s0[off(s0, i)]
    dst[off(dst, i)] = f(x)

`dst` has src0's shape (the packer, `guest/ggml-rd/ops/unary.cpp`, checks
it); both may be strided or permuted views. In place (dst is src0) each
thread reads its own element before it writes it, and no thread reads an
element another thread writes. No barriers, no groupshared, so there is no
Serial sibling: the cpp emit runs the same module.

## The functions, against ggml-cpu (`src/ggml-cpu/vec.h`)

- `silu(x)    = x / (1 + exp(-x))`, `sigmoid(x) = 1 / (1 + exp(-x))`,
  `neg(x) = -x`: ggml-cpu's scalar formulas.
- `gelu(x) = 0.5 x (1 + tanh(sqrt(2/pi) x (1 + 0.044715 x^2)))`, with the
  tanh argument clamped to [-10, 10]: tanhf is exactly 1.0f from 9.01 on,
  so the clamp changes no result, and it keeps a driver whose tanh is
  `(e^y - e^-y) / (e^y + e^-y)` from returning inf/inf = NaN at
  |x| ~ 150 (test-backend-ops draws unary inputs from [-150, 150] to catch
  exactly that). ggml-cpu itself rounds GELU through an f16 table for
  |x| < 10 (`GGML_GELU_FP16`); this kernel computes it in f32.
- `gelu_erf(x) = 0.5 x (1 + erf(x / sqrt 2))`, with `erf` from
  Abramowitz & Stegun 7.1.26 (below): Slang has no erf on either target.
- `scale(x) = x * s` when `b == 0` (ggml-cpu's `ggml_vec_scale_f32`, so a
  -0 stays -0), else `x * s + b`; `s`, `b` are op_params words 0 and 1.
- `diag_mask_inf`: `-inf` where `i0 > n_past + i1`, else `x`; `n_past`
  is op_params word 0 (ggml asserts it is >= 0).
- `relu(x) = (x > 0) ? x : 0` (`ggml_vec_relu_f32`), so a -0 becomes +0
  and a NaN becomes 0, as on the CPU.
- `leaky_relu(x, ns) = ((x > 0) ? x : 0) + ns * ((x < 0) ? x : 0)`
  (`ggml_vec_leaky_relu_f32`, its two-term form kept: the sum is what
  makes a -0 input +0 and an x < 0 exactly `0 + ns * x`); `ns`
  (negative_slope) is op_params word 0.
- `clamp(x, lo, hi) = MAX(MIN(x, hi), lo)` written as ggml-cpu's macros
  (`(x < hi) ? x : hi`, then `(m > lo) ? m : lo`), so a NaN comes out as
  `lo` on both; `lo`, `hi` are op_params words 0 and 1.
-/

namespace Ggml.SlangCodegen.Unary

open LeanSlang
open Ggml.SlangCodegen.Common

/-- Threads per work group. -/
def threadgroup : Nat := 256

/-- An f32 literal, exactly (`litFloatExact`). -/
def fl (x : Float) : SlangExpr := .litFloatExact x
def fsub (a b : SlangExpr) : SlangExpr := .bin "-" a b
def fdiv (a b : SlangExpr) : SlangExpr := .bin "/" a b
def neg (a : SlangExpr) : SlangExpr := .un "-" a

private def floatParam (n : String) : SlangBinding :=
  ⟨n, .scalar .float, Semantic.none, none, none, .qIn⟩
private def float1 (name : String) (body : List SlangStmt) : SlangFunctionDecl :=
  { retType := .scalar .float, name := name, params := [floatParam "x"], body := body }

/-! ## The constants (ggml-cpu's, `src/ggml-cpu/vec.h`) -/

def geluCoefA : Float := 0.044715
def sqrt2OverPi : Float := 0.79788456080286535587989211986876
def sqrt2Inv : Float := 0.70710678118654752440084436210484
/-- The clamp on GELU's tanh argument (see the module doc). -/
def tanhClamp : Float := 10.0

/-! ## Abramowitz & Stegun 7.1.26

    erf(x) = 1 - (a1 t + a2 t^2 + a3 t^3 + a4 t^4 + a5 t^5) e^(-x^2),
    t = 1 / (1 + p x),  x >= 0,  |error| <= 1.5e-7,

odd-extended to x < 0. Horner form, as written below. -/

def asP  : Float := 0.3275911
def asA1 : Float := 0.254829592
def asA2 : Float := -0.284496736
def asA3 : Float := 1.421413741
def asA4 : Float := -1.453152027
def asA5 : Float := 1.061405429

/-- `float erf_as(float x)`: A&S 7.1.26, odd-extended. -/
def fnErf : SlangFunctionDecl :=
  float1 "erf_as"
    [ .declInit (.scalar .float) "z" (.call "abs" [v "x"])
    , .declInit (.scalar .float) "t" (fdiv (fl 1.0) (add (fl 1.0) (mul (fl asP) (v "z"))))
    , .declInit (.scalar .float) "q"
        (mul (v "t") (add (fl asA1) (mul (v "t") (add (fl asA2) (mul (v "t")
          (add (fl asA3) (mul (v "t") (add (fl asA4) (mul (v "t") (fl asA5))))))))))
    , .declInit (.scalar .float) "r"
        (fsub (fl 1.0) (mul (v "q") (.call "exp" [neg (mul (v "z") (v "z"))])))
    , .ret (some (.ternary (.bin "<" (v "x") (fl 0.0)) (neg (v "r")) (v "r"))) ]

/-- `float silu(float x) { return x / (1 + exp(-x)); }` -/
def fnSilu : SlangFunctionDecl :=
  float1 "silu" [ .ret (some (fdiv (v "x") (add (fl 1.0) (.call "exp" [neg (v "x")])))) ]

/-- `float sigmoid(float x) { return 1 / (1 + exp(-x)); }` -/
def fnSigmoid : SlangFunctionDecl :=
  float1 "sigmoid" [ .ret (some (fdiv (fl 1.0) (add (fl 1.0) (.call "exp" [neg (v "x")])))) ]

/-- `float gelu_tanh(float x)`: ggml_gelu_f32's order of operations,
    `SQRT_2_OVER_PI*x*(1.0f + GELU_COEF_A*x*x)`, tanh argument clamped. -/
def fnGelu : SlangFunctionDecl :=
  float1 "gelu_tanh"
    [ .declInit (.scalar .float) "y"
        (mul (mul (fl sqrt2OverPi) (v "x")) (add (fl 1.0) (mul (mul (fl geluCoefA) (v "x")) (v "x"))))
    , .ret (some (mul (mul (fl 0.5) (v "x"))
        (add (fl 1.0) (.call "tanh" [.call "clamp" [v "y", fl (-tanhClamp), fl tanhClamp]])))) ]

/-- `float gelu_erf(float x) { return 0.5f*x*(1.0f + erf(x*SQRT_2_INV)); }` -/
def fnGeluErf : SlangFunctionDecl :=
  float1 "gelu_erf"
    [ .ret (some (mul (mul (fl 0.5) (v "x"))
        (add (fl 1.0) (.call "erf_as" [mul (v "x") (fl sqrt2Inv)])))) ]

/-- `float relu(float x) { return (x > 0) ? x : 0; }` -/
def fnRelu : SlangFunctionDecl :=
  float1 "relu" [ .ret (some (.ternary (.bin ">" (v "x") (fl 0.0)) (v "x") (fl 0.0))) ]

/-- `float leaky_relu(float x, float ns)`: ggml_vec_leaky_relu_f32's sum. -/
def fnLeakyRelu : SlangFunctionDecl :=
  { retType := .scalar .float
  , name := "leaky_relu"
  , params := [floatParam "x", floatParam "ns"]
  , body := [ .ret (some (add (.ternary (.bin ">" (v "x") (fl 0.0)) (v "x") (fl 0.0))
                              (mul (v "ns") (.ternary (.bin "<" (v "x") (fl 0.0)) (v "x") (fl 0.0))))) ] }

/-- `float clamp_ggml(float x, float lo, float hi)`: `MAX(MIN(x, hi), lo)`
    with ggml's macros (`a < b ? a : b`, `a > b ? a : b`), not Slang's
    `clamp`, whose NaN result is the target's choice. -/
def fnClampGgml : SlangFunctionDecl :=
  { retType := .scalar .float
  , name := "clamp_ggml"
  , params := [floatParam "x", floatParam "lo", floatParam "hi"]
  , body := [ .declInit (.scalar .float) "m" (.ternary (.bin "<" (v "x") (v "hi")) (v "x") (v "hi"))
            , .ret (some (.ternary (.bin ">" (v "m") (v "lo")) (v "m") (v "lo"))) ] }

/-! ## Kernels -/

/-- The body after the 1-D prologue: unravel, two offsets, `x`, then
    `dst[d] = res` (`res` may read `x`, `i0..i3`, `a`). -/
def body (res : SlangExpr) : List SlangStmt :=
  [ .decl (.scalar .uint) "i0"
  , .decl (.scalar .uint) "i1"
  , .decl (.scalar .uint) "i2"
  , .decl (.scalar .uint) "i3"
  , .expr (.call "unravel4" [v "e", u wDst, v "i0", v "i1", v "i2", v "i3"])
  , .declInit (.scalar .uint) "d" (.call "off4" [u wDst, v "i0", v "i1", v "i2", v "i3"])
  , .declInit (.scalar .uint) "a" (.call "off4" [u (wSrc 0), v "i0", v "i1", v "i2", v "i3"])
  , .declInit (.scalar .float) "x" (.index (v "s0") (v "a"))
  , .assign (.index (v "dst") (v "d")) res ]

/-- A K1 kernel: f32 in s0, f32 out; `extra` helpers before the entry. -/
def shader (extra : List SlangFunctionDecl) (res : SlangExpr) : SlangShaderModule :=
  kernelModule .float .uint .uint .float
    ([fnPw, fnUnravel4, fnOff4] ++ extra)
    (entry1D threadgroup (body res))

def siluF32 : SlangShaderModule := shader [fnSilu] (.call "silu" [v "x"])
def geluF32 : SlangShaderModule := shader [fnGelu] (.call "gelu_tanh" [v "x"])
def geluErfF32 : SlangShaderModule := shader [fnErf, fnGeluErf] (.call "gelu_erf" [v "x"])
def sigmoidF32 : SlangShaderModule := shader [fnSigmoid] (.call "sigmoid" [v "x"])
def negF32 : SlangShaderModule := shader [] (neg (v "x"))
def reluF32 : SlangShaderModule := shader [fnRelu] (.call "relu" [v "x"])

/-- LEAKY_RELU: `negative_slope` = op_params word 0. -/
def leakyReluF32 : SlangShaderModule :=
  shader [fnPf, fnLeakyRelu] (.call "leaky_relu" [v "x", pfN wOpParams])

/-- CLAMP: `min` = op_params word 0, `max` = word 1. -/
def clampF32 : SlangShaderModule :=
  shader [fnPf, fnClampGgml] (.call "clamp_ggml" [v "x", pfN wOpParams, pfN (wOpParams + 1)])

/-- SCALE: `s` = op_params word 0, `b` = word 1. -/
def scaleF32 : SlangShaderModule :=
  let s := pfN wOpParams
  let b := pfN (wOpParams + 1)
  shader [fnPf]
    (.ternary (.bin "==" b (fl 0.0)) (mul (v "x") s) (add (mul (v "x") s) b))

/-- `-inf` as f32 bits. -/
def negInfBits : Nat := 0xFF800000

/-- DIAG_MASK_INF: `n_past` = op_params word 0. -/
def diagMaskInfF32 : SlangShaderModule :=
  shader []
    (.ternary (.bin ">" (v "i0") (add (pwN wOpParams) (v "i1")))
      (.call "asfloat" [u negInfBits]) (v "x"))

/-- The kernels this module contributes, by their kernels.txt names. -/
def kernels : List (String × SlangShaderModule) :=
  [ ("silu_f32", siluF32)
  , ("gelu_f32", geluF32)
  , ("gelu_erf_f32", geluErfF32)
  , ("sigmoid_f32", sigmoidF32)
  , ("neg_f32", negF32)
  , ("scale_f32", scaleF32)
  , ("diag_mask_inf_f32", diagMaskInfF32)
  , ("relu_f32", reluF32)
  , ("leaky_relu_f32", leakyReluF32)
  , ("clamp_f32", clampF32) ]

/-! ## Pins -/

/-- silu_f32 in full; every other K1 kernel is this text with the helper
    and the one result line changed (pinned below). -/
def expectedSilu : String :=
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
RWStructuredBuffer<uint> s1;
[[vk::binding(3, 0)]]
RWStructuredBuffer<uint> s2;
[[vk::binding(4, 0)]]
RWStructuredBuffer<float> dst;
[[vk::binding(0, 1)]]
ConstantBuffer<Slot> slot;

uint pw(uint k) {
  return params[(slot.base + k)];
}

void unravel4(uint e, uint w, out uint i0, out uint i1, out uint i2, out uint i3) {
  uint n0 = pw(w);
  uint n1 = pw((w + 1u));
  uint n2 = pw((w + 2u));
  i0 = (e % n0);
  uint r = (e / n0);
  i1 = (r % n1);
  r = (r / n1);
  i2 = (r % n2);
  i3 = (r / n2);
}

uint off4(uint w, uint i0, uint i1, uint i2, uint i3) {
  return ((((pw((w + 8u)) + (i0 * pw((w + 4u)))) + (i1 * pw((w + 5u)))) + (i2 * pw((w + 6u)))) + (i3 * pw((w + 7u))));
}

float silu(float x) {
  return (x / (1.0f + exp((-x))));
}

[shader(\"compute\")] [numthreads(256, 1, 1)]
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
  uint a = off4(10u, i0, i1, i2, i3);
  float x = s0[a];
  dst[d] = silu(x);
}"

example : LeanSlang.emit siluF32 = expectedSilu := by native_decide
example : siluF32.entryPointName = "main" := by native_decide

/-- The silu helper and its call, the only lines each sibling changes. -/
def siluFn : String := "float silu(float x) {
  return (x / (1.0f + exp((-x))));
}

"
def siluCall : String := "dst[d] = silu(x);"

/-- A sibling of silu_f32: its helpers' text and its result line. -/
def sibling (helpers res : String) : String :=
  (expectedSilu.replace siluFn helpers).replace siluCall res

example : LeanSlang.emit sigmoidF32 = sibling
"float sigmoid(float x) {
  return (1.0f / (1.0f + exp((-x))));
}

" "dst[d] = sigmoid(x);" := by native_decide

example : LeanSlang.emit geluF32 = sibling
"float gelu_tanh(float x) {
  float y = ((0.7978846f * x) * (1.0f + ((0.044715f * x) * x)));
  return ((0.5f * x) * (1.0f + tanh(clamp(y, (-10.0f), 10.0f))));
}

" "dst[d] = gelu_tanh(x);" := by native_decide

example : LeanSlang.emit geluErfF32 = sibling
"float erf_as(float x) {
  float z = abs(x);
  float t = (1.0f / (1.0f + (0.3275911f * z)));
  float q = (t * (0.2548296f + (t * ((-0.28449672f) + (t * (1.4214138f + (t * ((-1.4531521f) + (t * 1.0614054f)))))))));
  float r = (1.0f - (q * exp((-(z * z)))));
  return ((x < 0.0f) ? (-r) : r);
}

float gelu_erf(float x) {
  return ((0.5f * x) * (1.0f + erf_as((x * 0.70710677f))));
}

" "dst[d] = gelu_erf(x);" := by native_decide

example : LeanSlang.emit negF32 = sibling "" "dst[d] = (-x);" := by native_decide

example : LeanSlang.emit scaleF32 = sibling
"float pf(uint k) {
  return asfloat(pw(k));
}

" "dst[d] = ((pf(38u) == 0.0f) ? (x * pf(37u)) : ((x * pf(37u)) + pf(38u)));" := by native_decide

example : LeanSlang.emit diagMaskInfF32 = sibling ""
  "dst[d] = ((i0 > (pw(37u) + i1)) ? asfloat(4286578688u) : x);" := by native_decide

example : LeanSlang.emit reluF32 = sibling
"float relu(float x) {
  return ((x > 0.0f) ? x : 0.0f);
}

" "dst[d] = relu(x);" := by native_decide

example : LeanSlang.emit leakyReluF32 = sibling
"float pf(uint k) {
  return asfloat(pw(k));
}

float leaky_relu(float x, float ns) {
  return (((x > 0.0f) ? x : 0.0f) + (ns * ((x < 0.0f) ? x : 0.0f)));
}

" "dst[d] = leaky_relu(x, pf(37u));" := by native_decide

example : LeanSlang.emit clampF32 = sibling
"float pf(uint k) {
  return asfloat(pw(k));
}

float clamp_ggml(float x, float lo, float hi) {
  float m = ((x < hi) ? x : hi);
  return ((m > lo) ? m : lo);
}

" "dst[d] = clamp_ggml(x, pf(37u), pf(38u));" := by native_decide

end Ggml.SlangCodegen.Unary
