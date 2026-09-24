import LeanSlang
import Ggml.SlangCodegen.Common

/-!
# `Ggml.SlangCodegen.Upscale` — ggml UPSCALE, mode NEAREST, f32

`ggml_interpolate(a, ne0..ne3, GGML_SCALE_MODE_NEAREST)` (and `ggml_upscale`,
which is it with `ne = a.ne * factor`): dst has the shape it was given and
each destination element copies the source element at

    i_src[k] = (i[k] * src0.ne[k]) / dst.ne[k]      (integer division)

per dimension, a gather in the shape of `Repeat` (one thread per
destination element, `dst[off(dst, i)] = s0[off(s0, i_src)]`, both sides
strided). No arithmetic on the value, so it is a bit copy.

ggml-cpu (`ggml_compute_forward_upscale_f32`) writes the same index as
`(int64_t)(i / sf)` with `sf = (float) dst.ne / src0.ne`: a float divide
truncated. For an integer factor `sf` is exact and `i / sf` truncates to
`i div sf` for every i below 2^24, so the two agree; for a ratio that is
not an integer they agree wherever the float quotient does not fall on the
wrong side of an integer, which the L2 cases (`tests/ggml_rd_kernels/
cases/move.cpp`, test-backend-ops' [2,5,7,11] <-> [5,7,11,13]) check for
the shapes they run. The integer form is kept because a GPU divide may be
2.5 ULP off (the Vulkan precision floor), which for an exact integer
quotient could truncate one step low. The packer
(`guest/ggml-rd/ops/upscale.cpp`) refuses every mode but plain NEAREST
(no `GGML_SCALE_FLAG_ALIGN_CORNERS`, which changes `sf`, no ANTIALIAS)
and any dimension where `i * src0.ne` would overflow a word.
-/

namespace Ggml.SlangCodegen.Upscale

open LeanSlang
open Ggml.SlangCodegen.Common

/-- Threads per work group. -/
def threadgroup : Nat := 256

private def uintParam (n : String) : SlangBinding :=
  ⟨n, .scalar .uint, Semantic.none, none, none, .qIn⟩

/-- `uint src_idx(uint i, uint nd, uint ns) { return (i * ns) / nd; }`:
    the nearest source index of destination index `i` in a dimension of
    `nd` destination and `ns` source elements. -/
def fnSrcIdx : SlangFunctionDecl :=
  { retType := .scalar .uint
  , name := "src_idx"
  , params := [uintParam "i", uintParam "nd", uintParam "ns"]
  , body := [ .ret (some (udiv (mul (v "i") (v "ns")) (v "nd"))) ] }

/-- The body after the 1-D prologue: unravel, the source index per
    dimension, one copy. -/
def body : List SlangStmt :=
  let s0 := wSrc 0
  let si (k : Nat) : SlangExpr :=
    .call "src_idx" [v ("i" ++ toString k), pwN (wDst + tNe + k), pwN (s0 + tNe + k)]
  [ .decl (.scalar .uint) "i0"
  , .decl (.scalar .uint) "i1"
  , .decl (.scalar .uint) "i2"
  , .decl (.scalar .uint) "i3"
  , .expr (.call "unravel4" [v "e", u wDst, v "i0", v "i1", v "i2", v "i3"])
  , .declInit (.scalar .uint) "d" (.call "off4" [u wDst, v "i0", v "i1", v "i2", v "i3"])
  , .declInit (.scalar .uint) "a" (.call "off4" [u s0, si 0, si 1, si 2, si 3])
  , .assign (.index (v "dst") (v "d")) (.index (v "s0") (v "a")) ]

def upscaleNearestF32 : SlangShaderModule :=
  kernelModule .float .uint .uint .float
    [fnPw, fnUnravel4, fnOff4, fnSrcIdx]
    (entry1D threadgroup body)

/-- The kernels this module contributes, by their kernels.txt names. -/
def kernels : List (String × SlangShaderModule) :=
  [ ("upscale_nearest_f32", upscaleNearestF32) ]

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

uint src_idx(uint i, uint nd, uint ns) {
  return ((i * ns) / nd);
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
  uint a = off4(10u, src_idx(i0, pw(1u), pw(10u)), src_idx(i1, pw(2u), pw(11u)), src_idx(i2, pw(3u), pw(12u)), src_idx(i3, pw(4u), pw(13u)));
  dst[d] = s0[a];
}"

example : LeanSlang.emit upscaleNearestF32 = expected := by native_decide
example : upscaleNearestF32.entryPointName = "main" := by native_decide

end Ggml.SlangCodegen.Upscale
