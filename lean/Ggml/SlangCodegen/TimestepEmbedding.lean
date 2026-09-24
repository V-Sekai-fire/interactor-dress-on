import LeanSlang
import Ggml.SlangCodegen.Common

/-!
# `Ggml.SlangCodegen.TimestepEmbedding` — ggml TIMESTEP_EMBEDDING, f32

`ggml_timestep_embedding(timesteps, dim, max_period)`: dst is
`[dim, N]` for `N = timesteps.ne0` (op_params words 0 and 1 are `dim` and
`max_period`), and with `half = dim / 2`, for timestep `t = s0[i1]`:

    dst[j, i1]        = cos(t * freq(j))        j < half
    dst[half + j, i1] = sin(t * freq(j))        j < half
    dst[2 half, i1]   = 0                       dim odd
    freq(j) = exp((-log(max_period) * j) / half)

exactly ggml-cpu's `ggml_compute_forward_timestep_embedding_f32`
(`expf(-logf(max_period) * j / half)`: `max_period` converted to f32 for
the log, `j` for the product, `half` for the quotient, in that order).
One thread per destination element, dst through its strides, the
timesteps through theirs. exp, log, cos and sin are the target's: the
cpp emit's libm agrees with ggml-cpu's, a GPU's within its own precision.
-/

namespace Ggml.SlangCodegen.TimestepEmbedding

open LeanSlang
open Ggml.SlangCodegen.Common

/-- Threads per work group. -/
def threadgroup : Nat := 256

private def U : SlangType := .scalar .uint
private def F : SlangType := .scalar .float
private def uintParam (n : String) : SlangBinding :=
  ⟨n, .scalar .uint, Semantic.none, none, none, .qIn⟩

/-- `float freq(uint j, uint half)`: `exp((-log(float(max_period)) * float(j)) / float(half))`. -/
def fnFreq : SlangFunctionDecl :=
  { retType := F
  , name := "freq"
  , params := [uintParam "j", uintParam "half"]
  , body := [ .ret (some (.call "exp"
      [ .bin "/" (mul (.un "-" (.call "log" [.cast F (pwN (wOpParams + 1))])) (.cast F (v "j")))
                 (.cast F (v "half")) ])) ] }

def body : List SlangStmt :=
  let s0 := wSrc 0
  [ .decl U "i0", .decl U "i1", .decl U "i2", .decl U "i3"
  , .expr (.call "unravel4" [v "e", u wDst, v "i0", v "i1", v "i2", v "i3"])
  , .declInit U "d" (.call "off4" [u wDst, v "i0", v "i1", v "i2", v "i3"])
  , .declInit U "half" (udiv (pwN wOpParams) (u 2))
  , .declInit F "t" (.index (v "s0") (.call "off4" [u s0, v "i1", u 0, u 0, u 0]))
  , .ifThen (.bin "<" (v "i0") (v "half"))
      [ .assign (.index (v "dst") (v "d")) (.call "cos" [mul (v "t") (.call "freq" [v "i0", v "half"])]) ]
      [ .ifThen (.bin "<" (v "i0") (mul (u 2) (v "half")))
          [ .assign (.index (v "dst") (v "d"))
              (.call "sin" [mul (v "t") (.call "freq" [.bin "-" (v "i0") (v "half"), v "half"])]) ]
          [ .assign (.index (v "dst") (v "d")) (.litFloatExact 0.0) ] ] ]

def timestepEmbeddingF32 : SlangShaderModule :=
  kernelModule .float .uint .uint .float
    [fnPw, fnUnravel4, fnOff4, fnFreq]
    (entry1D threadgroup body)

/-- The kernels this module contributes, by their kernels.txt names. -/
def kernels : List (String × SlangShaderModule) :=
  [ ("timestep_embedding_f32", timestepEmbeddingF32) ]

def expected : String :=
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

float freq(uint j, uint half) {
  return exp((((-log(float(pw(38u)))) * float(j)) / float(half)));
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
  uint half = (pw(37u) / 2u);
  float t = s0[off4(10u, i1, 0u, 0u, 0u)];
  if ((i0 < half)) {
    dst[d] = cos((t * freq(i0, half)));
  } else {
    if ((i0 < (2u * half))) {
      dst[d] = sin((t * freq((i0 - half), half)));
    } else {
      dst[d] = 0.0f;
    }
  }
}"

example : LeanSlang.emit timestepEmbeddingF32 = expected := by native_decide
example : timestepEmbeddingF32.entryPointName = "main" := by native_decide

end Ggml.SlangCodegen.TimestepEmbedding
