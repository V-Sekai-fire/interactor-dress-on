import LeanSlang
import Ggml.SlangCodegen.Common
import Ggml.SlangCodegen.Unary

/-!
# `Ggml.SlangCodegen.UnaryMath` — family K1: the f32 math unaries

ABS, SGN, STEP, TANH, EXP, FLOOR (`GGML_OP_UNARY`) and SQR, LOG, SIN, COS
(ops of their own in ggml), the element-wise ops interactor-nx-ggml's
graph builder and interactor-rf-detr-ggml's decoder, deformable attention
and loss build from. Each is `Unary.shader` (one thread per destination
element, `dst[off(dst, i)] = f(s0[off(s0, i)])`, both sides strided, in
place safe) with one function; the packer is
`guest/ggml-rd/ops/unary_math.cpp`.

## The functions, against ggml-cpu (`src/ggml-cpu/unary-ops.cpp`)

ggml-cpu computes every one of these in f32 (`op_abs` .. `op_floor`,
through `unary_op<op>` with `float` source and destination): none reads an
f16 table (`GGML_GELU_FP16` and `GGML_GELU_QUICK_FP16` are the only table
macros in `vec.h`, for GELU and GELU_QUICK), so the f32 formula is the
whole reference:

- `abs(x) = fabsf(x)`; `sqr(x) = x * x`; `floor(x) = floorf(x)`;
  `exp(x) = expf(x)`; `log(x) = logf(x)`; `sin(x) = sinf(x)`;
  `cos(x) = cosf(x)`. Slang's `abs`, `floor`, `exp`, `log`, `sin`, `cos`
  compile to `fabsf`, `floorf`, `expf`, `logf`, `sinf`, `cosf` on the cpp
  target (the prelude's `F32_*`), so the host emit is bit-exact against
  ggml-cpu; on the GPU `exp`, `log`, `sin`, `cos` are the driver's, within
  test-backend-ops' NMSE.
- `sgn(x) = (x > 0) ? 1 : ((x < 0) ? -1 : 0)`, `step(x) = (x > 0) ? 1 : 0`:
  ggml-cpu's ternaries, so a NaN gives 0 on both.
- `tanh(x) = tanhf(x)`, with the argument clamped to [-10, 10] by two
  ternaries (a NaN passes through, unlike Slang's `clamp`): tanhf is
  exactly 1.0f from 9.01 on, so the clamp changes no result, and it keeps a
  driver whose tanh is `(e^2y - 1) / (e^2y + 1)` from returning inf/inf at
  |x| ~ 150 (test-backend-ops draws unary inputs from [-150, 150]).
-/

namespace Ggml.SlangCodegen.UnaryMath

open LeanSlang
open Ggml.SlangCodegen.Common
open Ggml.SlangCodegen.Unary

private def floatParam (n : String) : SlangBinding :=
  ⟨n, .scalar .float, Semantic.none, none, none, .qIn⟩
private def float1 (name : String) (body : List SlangStmt) : SlangFunctionDecl :=
  { retType := .scalar .float, name := name, params := [floatParam "x"], body := body }

/-- `float sgn(float x) { return (x > 0) ? 1 : ((x < 0) ? -1 : 0); }` -/
def fnSgn : SlangFunctionDecl :=
  float1 "sgn"
    [ .ret (some (.ternary (.bin ">" (v "x") (fl 0.0)) (fl 1.0)
        (.ternary (.bin "<" (v "x") (fl 0.0)) (fl (-1.0)) (fl 0.0)))) ]

/-- `float step(float x) { return (x > 0) ? 1 : 0; }` -/
def fnStep : SlangFunctionDecl :=
  float1 "step_ggml" [ .ret (some (.ternary (.bin ">" (v "x") (fl 0.0)) (fl 1.0) (fl 0.0))) ]

/-- `float tanh_clamped(float x)`: tanh of x clamped to [-10, 10] (see the
    module doc); NaN stays NaN. -/
def fnTanhClamped : SlangFunctionDecl :=
  float1 "tanh_clamped"
    [ .declInit (.scalar .float) "y"
        (.ternary (.bin ">" (v "x") (fl tanhClamp)) (fl tanhClamp)
          (.ternary (.bin "<" (v "x") (fl (-tanhClamp))) (fl (-tanhClamp)) (v "x")))
    , .ret (some (.call "tanh" [v "y"])) ]

def absF32 : SlangShaderModule := shader [] (.call "abs" [v "x"])
def sgnF32 : SlangShaderModule := shader [fnSgn] (.call "sgn" [v "x"])
def stepF32 : SlangShaderModule := shader [fnStep] (.call "step_ggml" [v "x"])
def expF32 : SlangShaderModule := shader [] (.call "exp" [v "x"])
def tanhF32 : SlangShaderModule := shader [fnTanhClamped] (.call "tanh_clamped" [v "x"])
def floorF32 : SlangShaderModule := shader [] (.call "floor" [v "x"])
def sqrF32 : SlangShaderModule := shader [] (mul (v "x") (v "x"))
def logF32 : SlangShaderModule := shader [] (.call "log" [v "x"])
def sinF32 : SlangShaderModule := shader [] (.call "sin" [v "x"])
def cosF32 : SlangShaderModule := shader [] (.call "cos" [v "x"])

/-- The kernels this module contributes, by their kernels.txt names. -/
def kernels : List (String × SlangShaderModule) :=
  [ ("abs_f32", absF32)
  , ("sgn_f32", sgnF32)
  , ("step_f32", stepF32)
  , ("exp_f32", expF32)
  , ("tanh_f32", tanhF32)
  , ("floor_f32", floorF32)
  , ("sqr_f32", sqrF32)
  , ("log_f32", logF32)
  , ("sin_f32", sinF32)
  , ("cos_f32", cosF32) ]

/-! ## Pins: each kernel is `Unary.sibling` with its helpers and its result. -/

example : LeanSlang.emit absF32 = sibling "" "dst[d] = abs(x);" := by native_decide
example : LeanSlang.emit expF32 = sibling "" "dst[d] = exp(x);" := by native_decide
example : LeanSlang.emit floorF32 = sibling "" "dst[d] = floor(x);" := by native_decide
example : LeanSlang.emit sqrF32 = sibling "" "dst[d] = (x * x);" := by native_decide
example : LeanSlang.emit logF32 = sibling "" "dst[d] = log(x);" := by native_decide
example : LeanSlang.emit sinF32 = sibling "" "dst[d] = sin(x);" := by native_decide
example : LeanSlang.emit cosF32 = sibling "" "dst[d] = cos(x);" := by native_decide

example : LeanSlang.emit sgnF32 = sibling
"float sgn(float x) {
  return ((x > 0.0f) ? 1.0f : ((x < 0.0f) ? (-1.0f) : 0.0f));
}

" "dst[d] = sgn(x);" := by native_decide

example : LeanSlang.emit stepF32 = sibling
"float step_ggml(float x) {
  return ((x > 0.0f) ? 1.0f : 0.0f);
}

" "dst[d] = step_ggml(x);" := by native_decide

example : LeanSlang.emit tanhF32 = sibling
"float tanh_clamped(float x) {
  float y = ((x > 10.0f) ? 10.0f : ((x < (-10.0f)) ? (-10.0f) : x));
  return tanh(y);
}

" "dst[d] = tanh_clamped(x);" := by native_decide

end Ggml.SlangCodegen.UnaryMath
