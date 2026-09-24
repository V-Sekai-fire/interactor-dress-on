import LeanSlang
import Ggml.SlangCodegen.Common

/-!
# `Ggml.SlangCodegen.Pad` — ggml PAD (`ggml_pad`, `ggml_pad_ext`,
`ggml_pad_ext_circular`), f32

dst has `src0.ne[k] + lp[k] + rp[k]` elements in each dimension
(op_params words 0..7: `lp0 rp0 lp1 rp1 lp2 rp2 lp3 rp3`; word 8 is the
circular flag). One thread per destination element, dst through its
strides, src0 through its (test-backend-ops runs a strided view and a
permuted source):

    zero fill   dst[i] = every k: lp[k] <= i[k] < lp[k] + src0.ne[k]
                           ? src0[i - lp]  :  0
    circular    dst[i] = src0[(i[k] + src0.ne[k] - lp[k]) mod src0.ne[k]  per k]

exactly ggml-cpu's `ggml_compute_forward_pad_f32<false>` (the inside test
`i0 >= lp0 && i0 < ne0 - rp0`, which is `i0 - lp0 < ne00` since
`ne0 = ne00 + lp0 + rp0`) and `<true>` (`ggml_wrap_around(i - lp, n)` =
`(i - lp + n) % n`). Both are copies, so bit-exact. The tests run in
unsigned arithmetic, so the packer (`guest/ggml-rd/ops/pad.cpp`) refuses
negative padding, and for the circular form a left pad wider than the
source (where ggml-cpu's own `(coord + size) % size` goes negative).
-/

namespace Ggml.SlangCodegen.Pad

open LeanSlang
open Ggml.SlangCodegen.Common

/-- Threads per work group. -/
def threadgroup : Nat := 256

private def U : SlangType := .scalar .uint
private def uintParam (n : String) : SlangBinding :=
  ⟨n, .scalar .uint, Semantic.none, none, none, .qIn⟩

/-- op_params word of `lp[k]`; `rp[k]` is the next one. -/
def wLp (k : Nat) : Nat := wOpParams + 2 * k
/-- op_params word 8: circular. -/
def wCircular : Nat := wOpParams + 8

/-- `bool pad_in(uint i, uint lp, uint n)`: `lp <= i < lp + n`. -/
def fnPadIn : SlangFunctionDecl :=
  { retType := .scalar .bool
  , name := "pad_in"
  , params := [uintParam "i", uintParam "lp", uintParam "n"]
  , body := [ .ret (some (.bin "&&" (.bin ">=" (v "i") (v "lp")) (.bin "<" (.bin "-" (v "i") (v "lp")) (v "n")))) ] }

/-- `uint pad_wrap(uint i, uint lp, uint n)`: `(i + n - lp) % n`. -/
def fnPadWrap : SlangFunctionDecl :=
  { retType := U
  , name := "pad_wrap"
  , params := [uintParam "i", uintParam "lp", uintParam "n"]
  , body := [ .ret (some (urem (.bin "-" (add (v "i") (v "n")) (v "lp")) (v "n"))) ] }

def body : List SlangStmt :=
  let s0 := wSrc 0
  let i (k : Nat) : SlangExpr := v ("i" ++ toString k)
  let lp (k : Nat) : SlangExpr := pwN (wLp k)
  let n (k : Nat) : SlangExpr := pwN (s0 + tNe + k)
  let inside (k : Nat) : SlangExpr := .call "pad_in" [i k, lp k, n k]
  let wrap (k : Nat) : SlangExpr := .call "pad_wrap" [i k, lp k, n k]
  let src (k : Nat) : SlangExpr := .bin "-" (i k) (lp k)
  [ .decl U "i0", .decl U "i1", .decl U "i2", .decl U "i3"
  , .expr (.call "unravel4" [v "e", u wDst, v "i0", v "i1", v "i2", v "i3"])
  , .declInit U "d" (.call "off4" [u wDst, v "i0", v "i1", v "i2", v "i3"])
  , .ifThen (.bin "!=" (pwN wCircular) (u 0))
      [ .assign (.index (v "dst") (v "d")) (.index (v "s0") (.call "off4" [u s0, wrap 0, wrap 1, wrap 2, wrap 3])) ]
      [ .ifThen (.bin "&&" (.bin "&&" (.bin "&&" (inside 0) (inside 1)) (inside 2)) (inside 3))
          [ .assign (.index (v "dst") (v "d")) (.index (v "s0") (.call "off4" [u s0, src 0, src 1, src 2, src 3])) ]
          [ .assign (.index (v "dst") (v "d")) (.litFloatExact 0.0) ] ] ]

def padF32 : SlangShaderModule :=
  kernelModule .float .uint .uint .float
    [fnPw, fnUnravel4, fnOff4, fnPadIn, fnPadWrap]
    (entry1D threadgroup body)

/-- The kernels this module contributes, by their kernels.txt names. -/
def kernels : List (String × SlangShaderModule) :=
  [ ("pad_f32", padF32) ]

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

bool pad_in(uint i, uint lp, uint n) {
  return ((i >= lp) && ((i - lp) < n));
}

uint pad_wrap(uint i, uint lp, uint n) {
  return (((i + n) - lp) % n);
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
  if ((pw(45u) != 0u)) {
    dst[d] = s0[off4(10u, pad_wrap(i0, pw(37u), pw(10u)), pad_wrap(i1, pw(39u), pw(11u)), pad_wrap(i2, pw(41u), pw(12u)), pad_wrap(i3, pw(43u), pw(13u)))];
  } else {
    if ((((pad_in(i0, pw(37u), pw(10u)) && pad_in(i1, pw(39u), pw(11u))) && pad_in(i2, pw(41u), pw(12u))) && pad_in(i3, pw(43u), pw(13u)))) {
      dst[d] = s0[off4(10u, (i0 - pw(37u)), (i1 - pw(39u)), (i2 - pw(41u)), (i3 - pw(43u)))];
    } else {
      dst[d] = 0.0f;
    }
  }
}"

example : LeanSlang.emit padF32 = expected := by native_decide
example : padF32.entryPointName = "main" := by native_decide

end Ggml.SlangCodegen.Pad
