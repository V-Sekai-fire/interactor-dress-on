import LeanSlang
import Ggml.SlangCodegen.Common

/-!
# `Ggml.SlangCodegen.Binary` — ggml ADD and MUL, f32, with broadcast

The reference kernel of the ggml-rd op library (family K2): every other
kernel copies its shape. One thread per destination element:

    e            = this thread's linear element (threads >= word 53 return)
    (i0..i3)     = unravel(e) over dst's ne (ggml order, dim 0 fastest)
    dst[off(dst, i)] = s0[off(s0, i)] OP s1[off(s1, i mod ne1)]

`off(t, i)` is `t.offset + sum_k i_k * t.nb[k]`, in elements, so every
operand may be a strided or permuted view. The modulo is ggml's broadcast
(`ggml_can_repeat(src1, src0)`: each of src1's dims divides src0's), and
dst has src0's shape (`ggml_are_same_shape(src0, dst)`), which the packer
(`guest/ggml-rd/ops/binary.cpp`) checks before choosing this kernel.

Each result is one IEEE f32 add or multiply of the same two operands the
CPU backend reads, so it is bit-exact against ggml-cpu. s2 is unused and
kept by `-preserve-params`.
-/

namespace Ggml.SlangCodegen.Binary

open LeanSlang
open Ggml.SlangCodegen.Common

/-- Threads per work group. -/
def threadgroup : Nat := 256

/-- The body after the 1-D prologue: unravel, three offsets, one op. -/
def body (op : String) : List SlangStmt :=
  let s1 := wSrc 1
  [ .decl (.scalar .uint) "i0"
  , .decl (.scalar .uint) "i1"
  , .decl (.scalar .uint) "i2"
  , .decl (.scalar .uint) "i3"
  , .expr (.call "unravel4" [v "e", u wDst, v "i0", v "i1", v "i2", v "i3"])
  , .declInit (.scalar .uint) "d" (.call "off4" [u wDst, v "i0", v "i1", v "i2", v "i3"])
  , .declInit (.scalar .uint) "a" (.call "off4" [u (wSrc 0), v "i0", v "i1", v "i2", v "i3"])
  , .declInit (.scalar .uint) "b" (.call "off4"
      [ u s1
      , urem (v "i0") (pwN (s1 + tNe))
      , urem (v "i1") (pwN (s1 + tNe + 1))
      , urem (v "i2") (pwN (s1 + tNe + 2))
      , urem (v "i3") (pwN (s1 + tNe + 3)) ])
  , .assign (.index (v "dst") (v "d"))
      (.bin op (.index (v "s0") (v "a")) (.index (v "s1") (v "b"))) ]

/-- The module for one operator (`"+"` or `"*"`). -/
def shader (op : String) : SlangShaderModule :=
  kernelModule .float .float .uint .float
    [fnPw, fnUnravel4, fnOff4]
    (entry1D threadgroup (body op))

def addF32 : SlangShaderModule := shader "+"
def mulF32 : SlangShaderModule := shader "*"

/-- The kernels this module contributes, by their kernels.txt names. -/
def kernels : List (String × SlangShaderModule) :=
  [ ("add_f32", addF32)
  , ("mul_f32", mulF32) ]

/-- The Gate 3 aliasing control: add_f32 with its sources read-only. Bound
    over one RD buffer (a ggml buffer), its first binding of that buffer is
    read-only, so Godot's render graph records every span as a read and does
    not order the next span after it; an in-place chain then loses
    increments. add_f32 itself must not. -/
def ctlAddF32RoSources : SlangShaderModule :=
  controlModuleRoSources .float .float .uint .float
    [fnPw, fnUnravel4, fnOff4]
    (entry1D threadgroup (body "+"))

def controls : List (String × SlangShaderModule) :=
  [ ("ctl_add_f32_rosrc", ctlAddF32RoSources) ]

def expectedAdd : String :=
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
RWStructuredBuffer<float> s1;
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
  uint b = off4(19u, (i0 % pw(19u)), (i1 % pw(20u)), (i2 % pw(21u)), (i3 % pw(22u)));
  dst[d] = (s0[a] + s1[b]);
}"

example : LeanSlang.emit addF32 = expectedAdd := by native_decide
example : addF32.entryPointName = "main" := by native_decide
/-- The control differs from add_f32 in the three source declarations only. -/
example : LeanSlang.emit ctlAddF32RoSources =
    ((expectedAdd.replace "RWStructuredBuffer<float> s0;" "StructuredBuffer<float> s0;").replace
      "RWStructuredBuffer<float> s1;" "StructuredBuffer<float> s1;").replace
      "RWStructuredBuffer<uint> s2;" "StructuredBuffer<uint> s2;" := by
  native_decide

/-- MUL is ADD with the one operator swapped, and nothing else. -/
example : LeanSlang.emit mulF32 =
    (expectedAdd.replace "dst[d] = (s0[a] + s1[b]);" "dst[d] = (s0[a] * s1[b]);") := by
  native_decide

end Ggml.SlangCodegen.Binary
