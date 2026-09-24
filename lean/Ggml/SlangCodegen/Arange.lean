import LeanSlang
import Ggml.SlangCodegen.Common

/-!
# `Ggml.SlangCodegen.Arange` — ggml ARANGE, f32

`ggml_arange(start, stop, step)`: a 1-D f32 tensor of
`ceil((stop - start) / step)` elements (ggml computes the count and
asserts it against `ggml_nelements(dst)` in the op) with

    dst[i] = start + step * float(i)

exactly ggml-cpu's `ggml_compute_forward_arange_f32`
(`start + step * i`, `i` converted to f32 then one multiply and one add),
so bit-exact against it. `start`, `stop` and `step` are op_params words
0, 1 and 2 as floats; only `start` and `step` are read. No source: the
kernel writes dst through its own offset and stride (a new tensor, so
contiguous, but the words are used all the same). One thread per element.
-/

namespace Ggml.SlangCodegen.Arange

open LeanSlang
open Ggml.SlangCodegen.Common

/-- Threads per work group. -/
def threadgroup : Nat := 256

def body : List SlangStmt :=
  [ .assign (.index (v "dst") (add (pwN (wDst + tOff)) (mul (v "e") (pwN (wDst + tNb)))))
      (add (pfN wOpParams) (mul (pfN (wOpParams + 2)) (.cast (.scalar .float) (v "e")))) ]

def arangeF32 : SlangShaderModule :=
  kernelModule .uint .uint .uint .float
    [fnPw, fnPf]
    (entry1D threadgroup body)

/-- The kernels this module contributes, by their kernels.txt names. -/
def kernels : List (String × SlangShaderModule) :=
  [ ("arange_f32", arangeF32) ]

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
RWStructuredBuffer<uint> s0;
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

float pf(uint k) {
  return asfloat(pw(k));
}

[shader(\"compute\")] [numthreads(256, 1, 1)]
void main(uint3 gid : SV_GroupID, uint3 lid : SV_GroupThreadID) {
  uint e = ((((gid.y * pw(54u)) + gid.x) * 256u) + lid.x);
  if ((e >= pw(53u))) {
    return;
  }
  dst[(pw(9u) + (e * pw(5u)))] = (pf(37u) + (pf(39u) * float(e)));
}"

example : LeanSlang.emit arangeF32 = expected := by native_decide
example : arangeF32.entryPointName = "main" := by native_decide

end Ggml.SlangCodegen.Arange
