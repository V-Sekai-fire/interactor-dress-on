import LeanSlang
import Ggml.SlangCodegen.Common

/-!
# `Ggml.SlangCodegen.Conv` — ggml IM2COL and CONV_3D (family K7)

Four kernels, one thread per output, no barriers, no groupshared (so no
Serial sibling):

    im2col_f32     IM2COL, f32 image -> f32 columns
    im2col_f16     IM2COL, f32 image -> f16 columns (two halves per thread)
    conv3d_f32     CONV_3D (ggml_conv_3d_direct), f32 kernel, f32 input
    conv3d_f16     CONV_3D, f16 kernel, f32 input

## IM2COL (`ggml_im2col`, op_params s0 s1 p0 p1 d0 d1 is_2D at words 37-43)

src0 is the kernel, read for its shape only (KW = ne00, KH = ne01 in 2-D);
src1 the image. dst is `[IC*KH*KW, OW, OH, N]` (2-D) or `[IC*KW, OW, N, 1]`
(1-D), so for dst index (i0..i3):

    iic = i0 / (KH*KW), ikh = (i0 mod KH*KW) / KW, ikw = i0 mod KW
    iow = i1, ioh = 2-D ? i2 : 0, in = 2-D ? i3 : i2
    x = iow*s0 + ikw*d0 - p0, y = ioh*s1 + ikh*d1 - p1
    dst = (x, y) inside the image ? src1[x, y, iic, in] (2-D) | src1[x, iic, in] (1-D) : 0

exactly `ggml_compute_forward_im2col_f32/_f16` in ggml-cpu, with the image
read through its strides (the packer requires ggml-cpu's own layout
assumptions: nb10 = 4, and rows contiguous in 2-D). The padding test runs
in unsigned arithmetic: `x + p0 = iow*s0 + ikw*d0` is in range iff it is
`>= p0` and `- p0 < IW` (the packer refuses negative padding and
non-positive stride or dilation).

The f16 kernel writes a `uint` word buffer (no 16-bit storage capability)
and one thread owns one 32-bit word of dst: the two halves in it, each
computed if it belongs to dst and kept from the old word otherwise (only
the first and last word of an odd-offset or odd-length dst are read). The
conversion is `f32_to_f16`: round to nearest even in integer arithmetic
(normals, subnormals, overflow to inf, NaN -> 0x7E00 with the sign), the
same bits as ggml's `ggml_compute_fp32_to_fp16`, on every target (slangc's
C++ `f32tof16` rounds ties away from zero and SPIR-V leaves the rounding of
a half conversion to the driver).

## CONV_3D (`ggml_conv_3d_direct`, op_params s0 s1 s2 p0 p1 p2 d0 d1 d2 c n oc)

src0 is the kernel `[KW, KH, KD, IC*OC]`, src1 the input `[W, H, D, IC*N]`,
dst `[OW, OH, OD, OC*N]`. For dst index (ox, oy, oz, ocn), b = ocn / OC,
oc = ocn mod OC:

    dst = sum over ic < IC, kz < KD, ky < KH, kx < KW, (x, y, z) inside the input, of
          src1[x, y, z, b*IC + ic] * src0[kx, ky, kz, oc*IC + ic]
    x = ox*s0 + kx*d0 - p0 (and y, z alike)

accumulated in f32 in ggml-cpu's order (ic, kz, ky, kx). ggml-cpu computes
it as an im2col into the kernel's type and a matrix product, so for an f16
kernel the input is rounded to f16 first: conv3d_f16 does the same
(`f16tof32(f32_to_f16(x))`).
-/

namespace Ggml.SlangCodegen.Conv

open LeanSlang
open Ggml.SlangCodegen.Common

/-- Threads per work group. -/
def threadgroup : Nat := 256

private def uintParam (n : String) : SlangBinding :=
  ⟨n, .scalar .uint, Semantic.none, none, none, .qIn⟩
private def floatParam (n : String) : SlangBinding :=
  ⟨n, .scalar .float, Semantic.none, none, none, .qIn⟩

private def U : SlangType := .scalar .uint
private def F : SlangType := .scalar .float

def lt (a b : SlangExpr) : SlangExpr := .bin "<" a b
def ge (a b : SlangExpr) : SlangExpr := .bin ">=" a b
def gt (a b : SlangExpr) : SlangExpr := .bin ">" a b
def eq (a b : SlangExpr) : SlangExpr := .bin "==" a b
def ne (a b : SlangExpr) : SlangExpr := .bin "!=" a b
def lor (a b : SlangExpr) : SlangExpr := .bin "||" a b
def land (a b : SlangExpr) : SlangExpr := .bin "&&" a b
def sub (a b : SlangExpr) : SlangExpr := .bin "-" a b
def band (a b : SlangExpr) : SlangExpr := .bin "&" a b
def bor (a b : SlangExpr) : SlangExpr := .bin "|" a b
def shr (a b : SlangExpr) : SlangExpr := .bin ">>" a b
def shl (a b : SlangExpr) : SlangExpr := .bin "<<" a b

/-- `inside(c, lo, n)`: `c - lo` is in `[0, n)` for an unsigned `c` that
    stands for `c - lo` before the padding is taken off. -/
def inside (c lo n : SlangExpr) : SlangExpr := land (ge c lo) (lt (sub c lo) n)

/-! ## f32 -> f16, round to nearest even, in integer arithmetic -/

/-- `uint f32_to_f16(float f)`: the f16 bits of `f`, round to nearest even. -/
def fnF32ToF16 : SlangFunctionDecl :=
  { retType := U
  , name := "f32_to_f16"
  , params := [floatParam "f"]
  , body :=
      [ .declInit U "u" (.call "asuint" [v "f"])
      , .declInit U "s" (band (shr (v "u") (u 16)) (u 32768))
      , .declInit U "a" (band (v "u") (u 2147483647))
        -- NaN: a quiet NaN with the sign, as ggml.
      , .ifNoElse (gt (v "a") (u 2139095040)) [ .ret (some (bor (v "s") (u 32256))) ]
        -- |f| >= 65520 (65504 + half an ulp, a tie that rounds up): inf.
      , .ifNoElse (ge (v "a") (u 1199566848)) [ .ret (some (bor (v "s") (u 31744))) ]
        -- |f| >= 2^-14: a normal half. Rebias the exponent (127 -> 15),
        -- keep 10 mantissa bits, round on the 13 dropped ones.
      , .ifNoElse (ge (v "a") (u 947912704))
          [ .declInit U "h" (shr (sub (v "a") (u 939524096)) (u 13))
          , .declInit U "r" (band (v "a") (u 8191))
          , .ifNoElse (lor (gt (v "r") (u 4096)) (land (eq (v "r") (u 4096)) (ne (band (v "h") (u 1)) (u 0))))
              [ .assign (v "h") (add (v "h") (u 1)) ]
          , .ret (some (bor (v "s") (v "h"))) ]
        -- |f| <= 2^-25 (half the smallest subnormal, a tie to even 0): zero.
      , .ifNoElse (lt (v "a") (u 855638017)) [ .ret (some (v "s")) ]
        -- A subnormal half: the value in units of 2^-24, rounded.
      , .declInit U "m" (bor (band (v "a") (u 8388607)) (u 8388608))
      , .declInit U "sh" (sub (u 126) (shr (v "a") (u 23)))
      , .declInit U "h" (shr (v "m") (v "sh"))
      , .declInit U "r" (band (v "m") (sub (shl (u 1) (v "sh")) (u 1)))
      , .declInit U "hw" (shl (u 1) (sub (v "sh") (u 1)))
      , .ifNoElse (lor (gt (v "r") (v "hw")) (land (eq (v "r") (v "hw")) (ne (band (v "h") (u 1)) (u 0))))
          [ .assign (v "h") (add (v "h") (u 1)) ]
      , .ret (some (bor (v "s") (v "h"))) ] }

/-! ## IM2COL -/

def wS1 : Nat := wSrc 1
/-- op_params words: s0 s1 p0 p1 d0 d1 is_2D. -/
def wIm (k : Nat) : Nat := wOpParams + k

/-- `float im2col_at(i0, i1, i2, i3)`: the column value at dst index i. -/
def fnIm2colAt : SlangFunctionDecl :=
  { retType := F
  , name := "im2col_at"
  , params := [uintParam "i0", uintParam "i1", uintParam "i2", uintParam "i3"]
  , body :=
      [ .declInit (.scalar .bool) "is2d" (eq (pwN (wIm 6)) (u 1))
      , .declInit U "kw" (pwN (wSrc 0 + tNe))
      , .declInit U "kh" (.ternary (v "is2d") (pwN (wSrc 0 + tNe + 1)) (u 1))
      , .declInit U "iw" (pwN (wS1 + tNe))
      , .declInit U "ih" (.ternary (v "is2d") (pwN (wS1 + tNe + 1)) (u 1))
      , .declInit U "kk" (mul (v "kh") (v "kw"))
      , .declInit U "iic" (udiv (v "i0") (v "kk"))
      , .declInit U "r" (urem (v "i0") (v "kk"))
      , .declInit U "ikh" (udiv (v "r") (v "kw"))
      , .declInit U "ikw" (urem (v "r") (v "kw"))
      , .declInit U "ioh" (.ternary (v "is2d") (v "i2") (u 0))
      , .declInit U "inn" (.ternary (v "is2d") (v "i3") (v "i2"))
      , .declInit U "x" (add (mul (v "i1") (pwN (wIm 0))) (mul (v "ikw") (pwN (wIm 4))))
      , .declInit U "y" (add (mul (v "ioh") (pwN (wIm 1))) (mul (v "ikh") (pwN (wIm 5))))
      , .ifNoElse (.un "!" (land (inside (v "x") (pwN (wIm 2)) (v "iw")) (inside (v "y") (pwN (wIm 3)) (v "ih"))))
          [ .ret (some (.litFloat 0.0)) ]
      , .declInit U "xx" (sub (v "x") (pwN (wIm 2)))
      , .declInit U "yy" (sub (v "y") (pwN (wIm 3)))
        -- 2-D: [x, y, iic, in]; 1-D: [x, iic, in, 0].
      , .ret (some (.index (v "s1") (.call "off4"
          [ u wS1, v "xx"
          , .ternary (v "is2d") (v "yy") (v "iic")
          , .ternary (v "is2d") (v "iic") (v "inn")
          , .ternary (v "is2d") (v "inn") (u 0) ]))) ] }

/-- im2col_f32: one thread per dst element, through dst's strides. -/
def im2colF32Body : List SlangStmt :=
  [ .decl U "i0", .decl U "i1", .decl U "i2", .decl U "i3"
  , .expr (.call "unravel4" [v "e", u wDst, v "i0", v "i1", v "i2", v "i3"])
  , .assign (.index (v "dst") (.call "off4" [u wDst, v "i0", v "i1", v "i2", v "i3"]))
      (.call "im2col_at" [v "i0", v "i1", v "i2", v "i3"]) ]

/-- `uint im2col_h(uint h, uint off, uint n, uint old)`: the f16 bits of
    half `h` of the (contiguous) dst buffer: its column value if `h` is
    one of dst's `n` elements from `off`, else `old`. -/
def fnIm2colH : SlangFunctionDecl :=
  { retType := U
  , name := "im2col_h"
  , params := [uintParam "h", uintParam "off", uintParam "n", uintParam "old"]
  , body :=
      [ .ifNoElse (.un "!" (inside (v "h") (v "off") (v "n"))) [ .ret (some (v "old")) ]
      , .decl U "i0", .decl U "i1", .decl U "i2", .decl U "i3"
      , .expr (.call "unravel4" [sub (v "h") (v "off"), u wDst, v "i0", v "i1", v "i2", v "i3"])
      , .ret (some (.call "f32_to_f16" [.call "im2col_at" [v "i0", v "i1", v "i2", v "i3"]])) ] }

/-- im2col_f16: thread `e` owns word `(off >> 1) + e` of dst. -/
def im2colF16Body : List SlangStmt :=
  [ .declInit U "off" (pwN (wDst + tOff))
  , .declInit U "n" (mul (mul (pwN (wDst + tNe)) (pwN (wDst + tNe + 1)))
      (mul (pwN (wDst + tNe + 2)) (pwN (wDst + tNe + 3))))
  , .declInit U "wd" (add (shr (v "off") (u 1)) (v "e"))
  , .declInit U "h0" (mul (v "wd") (u 2))
  , .declInit U "old" (u 0)
  , .ifNoElse (.un "!" (land (inside (v "h0") (v "off") (v "n")) (inside (add (v "h0") (u 1)) (v "off") (v "n"))))
      [ .assign (v "old") (.index (v "dst") (v "wd")) ]
  , .declInit U "lo" (.call "im2col_h" [v "h0", v "off", v "n", band (v "old") (u 65535)])
  , .declInit U "hi" (.call "im2col_h" [add (v "h0") (u 1), v "off", v "n", shr (v "old") (u 16)])
  , .assign (.index (v "dst") (v "wd")) (bor (v "lo") (shl (v "hi") (u 16))) ]

def im2colF32 : SlangShaderModule :=
  kernelModule .uint .float .uint .float
    [fnPw, fnUnravel4, fnOff4, fnIm2colAt]
    (entry1D threadgroup im2colF32Body)

def im2colF16 : SlangShaderModule :=
  kernelModule .uint .float .uint .uint
    [fnPw, fnUnravel4, fnOff4, fnIm2colAt, fnF32ToF16, fnIm2colH]
    (entry1D threadgroup im2colF16Body)

/-! ## CONV_3D -/

/-- op_params words: s0 s1 s2 p0 p1 p2 d0 d1 d2 c n oc. -/
def wCv (k : Nat) : Nat := wOpParams + k

/-- The body; `f16` reads the kernel as f16 halves and rounds the input
    to f16 before the product, as ggml-cpu's f16 path. Every params word the
    loops use is read once into a local, and the input and kernel offsets
    advance per loop level (channel, z, y), so the innermost step is two
    loads and one multiply-add. -/
def conv3dBody (f16 : Bool) : List SlangStmt :=
  let s0 := wSrc 0
  let kval : SlangExpr :=
    let idx := add (v "kby") (mul (v "kx") (v "kn0"))
    if f16 then .call "ld_f16_s0" [idx] else .index (v "s0") idx
  let xval : SlangExpr :=
    let x := .index (v "s1") (add (v "iby") (mul (sub (v "x") (v "px")) (v "in0")))
    if f16 then .call "f16tof32" [.call "f32_to_f16" [x]] else x
  let w (n : String) (k : Nat) : SlangStmt := .declInit U n (pwN k)
  [ .decl U "ox", .decl U "oy", .decl U "oz", .decl U "ocn"
  , .expr (.call "unravel4" [v "e", u wDst, v "ox", v "oy", v "oz", v "ocn"])
  , w "c" (wCv 9), w "oc" (wCv 11)
  , w "sx" (wCv 0), w "sy" (wCv 1), w "sz" (wCv 2)
  , w "px" (wCv 3), w "py" (wCv 4), w "pz" (wCv 5)
  , w "dx" (wCv 6), w "dy" (wCv 7), w "dz" (wCv 8)
  , w "kw" (s0 + tNe), w "kh" (s0 + tNe + 1), w "kd" (s0 + tNe + 2)
  , w "iw" (wS1 + tNe), w "ih" (wS1 + tNe + 1), w "idd" (wS1 + tNe + 2)
  , w "kn0" (s0 + tNb), w "kn1" (s0 + tNb + 1), w "kn2" (s0 + tNb + 2), w "kn3" (s0 + tNb + 3)
  , w "in0" (wS1 + tNb), w "in1" (wS1 + tNb + 1), w "in2" (wS1 + tNb + 2), w "in3" (wS1 + tNb + 3)
    -- input channel b*IC + ic, kernel channel oc*IC + ic
  , .declInit U "ib" (add (pwN (wS1 + tOff)) (mul (mul (udiv (v "ocn") (v "oc")) (v "c")) (v "in3")))
  , .declInit U "kb" (add (pwN (s0 + tOff)) (mul (mul (urem (v "ocn") (v "oc")) (v "c")) (v "kn3")))
  , .declInit F "acc" (.litFloat 0.0)
  , .forCount "ic" (u 0) (v "c")
      [ .declInit U "ibc" (add (v "ib") (mul (v "ic") (v "in3")))
      , .declInit U "kbc" (add (v "kb") (mul (v "ic") (v "kn3")))
      , .forCount "kz" (u 0) (v "kd")
          [ .declInit U "z" (add (mul (v "oz") (v "sz")) (mul (v "kz") (v "dz")))
          , .ifNoElse (inside (v "z") (v "pz") (v "idd"))
              [ .declInit U "ibz" (add (v "ibc") (mul (sub (v "z") (v "pz")) (v "in2")))
              , .declInit U "kbz" (add (v "kbc") (mul (v "kz") (v "kn2")))
              , .forCount "ky" (u 0) (v "kh")
                  [ .declInit U "y" (add (mul (v "oy") (v "sy")) (mul (v "ky") (v "dy")))
                  , .ifNoElse (inside (v "y") (v "py") (v "ih"))
                      [ .declInit U "iby" (add (v "ibz") (mul (sub (v "y") (v "py")) (v "in1")))
                      , .declInit U "kby" (add (v "kbz") (mul (v "ky") (v "kn1")))
                      , .forCount "kx" (u 0) (v "kw")
                          [ .declInit U "x" (add (mul (v "ox") (v "sx")) (mul (v "kx") (v "dx")))
                          , .ifNoElse (inside (v "x") (v "px") (v "iw"))
                              [ .assign (v "acc") (add (v "acc") (mul xval kval)) ] ] ] ] ] ] ]
  , .assign (.index (v "dst") (.call "off4" [u wDst, v "ox", v "oy", v "oz", v "ocn"])) (v "acc") ]

def conv3dF32 : SlangShaderModule :=
  kernelModule .float .float .uint .float
    [fnPw, fnUnravel4, fnOff4]
    (entry1D threadgroup (conv3dBody false))

def conv3dF16 : SlangShaderModule :=
  kernelModule .uint .float .uint .float
    [fnPw, fnUnravel4, fnOff4, fnLdF16 "s0", fnF32ToF16]
    (entry1D threadgroup (conv3dBody true))

/-- The kernels this module contributes, by their kernels.txt names. -/
def kernels : List (String × SlangShaderModule) :=
  [ ("im2col_f32", im2colF32)
  , ("im2col_f16", im2colF16)
  , ("conv3d_f32", conv3dF32)
  , ("conv3d_f16", conv3dF16) ]

/-! ## Pins

Every kernel's whole text is pinned: a change to any helper or body shows
up here, next to the emission it moves. -/

def expectedIm2colF32 : String :=
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

float im2col_at(uint i0, uint i1, uint i2, uint i3) {
  bool is2d = (pw(43u) == 1u);
  uint kw = pw(10u);
  uint kh = (is2d ? pw(11u) : 1u);
  uint iw = pw(19u);
  uint ih = (is2d ? pw(20u) : 1u);
  uint kk = (kh * kw);
  uint iic = (i0 / kk);
  uint r = (i0 % kk);
  uint ikh = (r / kw);
  uint ikw = (r % kw);
  uint ioh = (is2d ? i2 : 0u);
  uint inn = (is2d ? i3 : i2);
  uint x = ((i1 * pw(37u)) + (ikw * pw(41u)));
  uint y = ((ioh * pw(38u)) + (ikh * pw(42u)));
  if ((!(((x >= pw(39u)) && ((x - pw(39u)) < iw)) && ((y >= pw(40u)) && ((y - pw(40u)) < ih))))) {
    return 0.000000;
  }
  uint xx = (x - pw(39u));
  uint yy = (y - pw(40u));
  return s1[off4(19u, xx, (is2d ? yy : iic), (is2d ? iic : inn), (is2d ? inn : 0u))];
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
  dst[off4(1u, i0, i1, i2, i3)] = im2col_at(i0, i1, i2, i3);
}"

def expectedIm2colF16 : String :=
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
RWStructuredBuffer<float> s1;
[[vk::binding(3, 0)]]
RWStructuredBuffer<uint> s2;
[[vk::binding(4, 0)]]
RWStructuredBuffer<uint> dst;
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

float im2col_at(uint i0, uint i1, uint i2, uint i3) {
  bool is2d = (pw(43u) == 1u);
  uint kw = pw(10u);
  uint kh = (is2d ? pw(11u) : 1u);
  uint iw = pw(19u);
  uint ih = (is2d ? pw(20u) : 1u);
  uint kk = (kh * kw);
  uint iic = (i0 / kk);
  uint r = (i0 % kk);
  uint ikh = (r / kw);
  uint ikw = (r % kw);
  uint ioh = (is2d ? i2 : 0u);
  uint inn = (is2d ? i3 : i2);
  uint x = ((i1 * pw(37u)) + (ikw * pw(41u)));
  uint y = ((ioh * pw(38u)) + (ikh * pw(42u)));
  if ((!(((x >= pw(39u)) && ((x - pw(39u)) < iw)) && ((y >= pw(40u)) && ((y - pw(40u)) < ih))))) {
    return 0.000000;
  }
  uint xx = (x - pw(39u));
  uint yy = (y - pw(40u));
  return s1[off4(19u, xx, (is2d ? yy : iic), (is2d ? iic : inn), (is2d ? inn : 0u))];
}

uint f32_to_f16(float f) {
  uint u = asuint(f);
  uint s = ((u >> 16u) & 32768u);
  uint a = (u & 2147483647u);
  if ((a > 2139095040u)) {
    return (s | 32256u);
  }
  if ((a >= 1199566848u)) {
    return (s | 31744u);
  }
  if ((a >= 947912704u)) {
    uint h = ((a - 939524096u) >> 13u);
    uint r = (a & 8191u);
    if (((r > 4096u) || ((r == 4096u) && ((h & 1u) != 0u)))) {
      h = (h + 1u);
    }
    return (s | h);
  }
  if ((a < 855638017u)) {
    return s;
  }
  uint m = ((a & 8388607u) | 8388608u);
  uint sh = (126u - (a >> 23u));
  uint h = (m >> sh);
  uint r = (m & ((1u << sh) - 1u));
  uint hw = (1u << (sh - 1u));
  if (((r > hw) || ((r == hw) && ((h & 1u) != 0u)))) {
    h = (h + 1u);
  }
  return (s | h);
}

uint im2col_h(uint h, uint off, uint n, uint old) {
  if ((!((h >= off) && ((h - off) < n)))) {
    return old;
  }
  uint i0;
  uint i1;
  uint i2;
  uint i3;
  unravel4((h - off), 1u, i0, i1, i2, i3);
  return f32_to_f16(im2col_at(i0, i1, i2, i3));
}

[shader(\"compute\")] [numthreads(256, 1, 1)]
void main(uint3 gid : SV_GroupID, uint3 lid : SV_GroupThreadID) {
  uint e = ((((gid.y * pw(54u)) + gid.x) * 256u) + lid.x);
  if ((e >= pw(53u))) {
    return;
  }
  uint off = pw(9u);
  uint n = ((pw(1u) * pw(2u)) * (pw(3u) * pw(4u)));
  uint wd = ((off >> 1u) + e);
  uint h0 = (wd * 2u);
  uint old = 0u;
  if ((!(((h0 >= off) && ((h0 - off) < n)) && (((h0 + 1u) >= off) && (((h0 + 1u) - off) < n))))) {
    old = dst[wd];
  }
  uint lo = im2col_h(h0, off, n, (old & 65535u));
  uint hi = im2col_h((h0 + 1u), off, n, (old >> 16u));
  dst[wd] = (lo | (hi << 16u));
}"

def expectedConv3dF32 : String :=
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
  uint ox;
  uint oy;
  uint oz;
  uint ocn;
  unravel4(e, 1u, ox, oy, oz, ocn);
  uint c = pw(46u);
  uint oc = pw(48u);
  uint sx = pw(37u);
  uint sy = pw(38u);
  uint sz = pw(39u);
  uint px = pw(40u);
  uint py = pw(41u);
  uint pz = pw(42u);
  uint dx = pw(43u);
  uint dy = pw(44u);
  uint dz = pw(45u);
  uint kw = pw(10u);
  uint kh = pw(11u);
  uint kd = pw(12u);
  uint iw = pw(19u);
  uint ih = pw(20u);
  uint idd = pw(21u);
  uint kn0 = pw(14u);
  uint kn1 = pw(15u);
  uint kn2 = pw(16u);
  uint kn3 = pw(17u);
  uint in0 = pw(23u);
  uint in1 = pw(24u);
  uint in2 = pw(25u);
  uint in3 = pw(26u);
  uint ib = (pw(27u) + (((ocn / oc) * c) * in3));
  uint kb = (pw(18u) + (((ocn % oc) * c) * kn3));
  float acc = 0.000000;
  for (uint ic = 0u; ic < c; ++ic) {
    uint ibc = (ib + (ic * in3));
    uint kbc = (kb + (ic * kn3));
    for (uint kz = 0u; kz < kd; ++kz) {
      uint z = ((oz * sz) + (kz * dz));
      if (((z >= pz) && ((z - pz) < idd))) {
        uint ibz = (ibc + ((z - pz) * in2));
        uint kbz = (kbc + (kz * kn2));
        for (uint ky = 0u; ky < kh; ++ky) {
          uint y = ((oy * sy) + (ky * dy));
          if (((y >= py) && ((y - py) < ih))) {
            uint iby = (ibz + ((y - py) * in1));
            uint kby = (kbz + (ky * kn1));
            for (uint kx = 0u; kx < kw; ++kx) {
              uint x = ((ox * sx) + (kx * dx));
              if (((x >= px) && ((x - px) < iw))) {
                acc = (acc + (s1[(iby + ((x - px) * in0))] * s0[(kby + (kx * kn0))]));
              }
            }
          }
        }
      }
    }
  }
  dst[off4(1u, ox, oy, oz, ocn)] = acc;
}"

def expectedConv3dF16 : String :=
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

float ld_f16_s0(uint e) {
  uint w = s0[(e >> 1u)];
  uint h = (((e & 1u) != 0u) ? (w >> 16u) : (w & 65535u));
  return f16tof32(h);
}

uint f32_to_f16(float f) {
  uint u = asuint(f);
  uint s = ((u >> 16u) & 32768u);
  uint a = (u & 2147483647u);
  if ((a > 2139095040u)) {
    return (s | 32256u);
  }
  if ((a >= 1199566848u)) {
    return (s | 31744u);
  }
  if ((a >= 947912704u)) {
    uint h = ((a - 939524096u) >> 13u);
    uint r = (a & 8191u);
    if (((r > 4096u) || ((r == 4096u) && ((h & 1u) != 0u)))) {
      h = (h + 1u);
    }
    return (s | h);
  }
  if ((a < 855638017u)) {
    return s;
  }
  uint m = ((a & 8388607u) | 8388608u);
  uint sh = (126u - (a >> 23u));
  uint h = (m >> sh);
  uint r = (m & ((1u << sh) - 1u));
  uint hw = (1u << (sh - 1u));
  if (((r > hw) || ((r == hw) && ((h & 1u) != 0u)))) {
    h = (h + 1u);
  }
  return (s | h);
}

[shader(\"compute\")] [numthreads(256, 1, 1)]
void main(uint3 gid : SV_GroupID, uint3 lid : SV_GroupThreadID) {
  uint e = ((((gid.y * pw(54u)) + gid.x) * 256u) + lid.x);
  if ((e >= pw(53u))) {
    return;
  }
  uint ox;
  uint oy;
  uint oz;
  uint ocn;
  unravel4(e, 1u, ox, oy, oz, ocn);
  uint c = pw(46u);
  uint oc = pw(48u);
  uint sx = pw(37u);
  uint sy = pw(38u);
  uint sz = pw(39u);
  uint px = pw(40u);
  uint py = pw(41u);
  uint pz = pw(42u);
  uint dx = pw(43u);
  uint dy = pw(44u);
  uint dz = pw(45u);
  uint kw = pw(10u);
  uint kh = pw(11u);
  uint kd = pw(12u);
  uint iw = pw(19u);
  uint ih = pw(20u);
  uint idd = pw(21u);
  uint kn0 = pw(14u);
  uint kn1 = pw(15u);
  uint kn2 = pw(16u);
  uint kn3 = pw(17u);
  uint in0 = pw(23u);
  uint in1 = pw(24u);
  uint in2 = pw(25u);
  uint in3 = pw(26u);
  uint ib = (pw(27u) + (((ocn / oc) * c) * in3));
  uint kb = (pw(18u) + (((ocn % oc) * c) * kn3));
  float acc = 0.000000;
  for (uint ic = 0u; ic < c; ++ic) {
    uint ibc = (ib + (ic * in3));
    uint kbc = (kb + (ic * kn3));
    for (uint kz = 0u; kz < kd; ++kz) {
      uint z = ((oz * sz) + (kz * dz));
      if (((z >= pz) && ((z - pz) < idd))) {
        uint ibz = (ibc + ((z - pz) * in2));
        uint kbz = (kbc + (kz * kn2));
        for (uint ky = 0u; ky < kh; ++ky) {
          uint y = ((oy * sy) + (ky * dy));
          if (((y >= py) && ((y - py) < ih))) {
            uint iby = (ibz + ((y - py) * in1));
            uint kby = (kbz + (ky * kn1));
            for (uint kx = 0u; kx < kw; ++kx) {
              uint x = ((ox * sx) + (kx * dx));
              if (((x >= px) && ((x - px) < iw))) {
                acc = (acc + (f16tof32(f32_to_f16(s1[(iby + ((x - px) * in0))])) * ld_f16_s0((kby + (kx * kn0)))));
              }
            }
          }
        }
      }
    }
  }
  dst[off4(1u, ox, oy, oz, ocn)] = acc;
}"

example : LeanSlang.emit im2colF32 = expectedIm2colF32 := by native_decide
example : LeanSlang.emit im2colF16 = expectedIm2colF16 := by native_decide
example : LeanSlang.emit conv3dF32 = expectedConv3dF32 := by native_decide
example : LeanSlang.emit conv3dF16 = expectedConv3dF16 := by native_decide
example : im2colF32.entryPointName = "main" := by native_decide
example : conv3dF16.entryPointName = "main" := by native_decide

/-- The f16 columns kernel is the f32 one's `im2col_at` plus the rounding
    and the word packing: the shared helper is the same text in both. -/
example : (expectedIm2colF32.splitOn (LeanSlang.emitFunction fnIm2colAt)).length = 2 := by native_decide
example : (expectedIm2colF16.splitOn (LeanSlang.emitFunction fnIm2colAt)).length = 2 := by native_decide

/-- f32_to_f16 is ggml's rounding (round to nearest even), pinned on its
    boundary cases by evaluating the same integer steps in Lean. -/
def f32ToF16Ref (a : Nat) : Nat :=
  let s := (a / 2147483648) % 2 * 32768
  let m := a % 2147483648
  if m > 2139095040 then s + 32256
  else if m >= 1199566848 then s + 31744
  else if m >= 947912704 then
    let h := (m - 939524096) / 8192
    let r := m % 8192
    s + (if r > 4096 || (r == 4096 && h % 2 == 1) then h + 1 else h)
  else if m < 855638017 then s
  else
    let mm := m % 8388608 + 8388608
    let sh := 126 - m / 8388608
    let h := mm / 2 ^ sh
    let r := mm % 2 ^ sh
    let hw := 2 ^ (sh - 1)
    s + (if r > hw || (r == hw && h % 2 == 1) then h + 1 else h)

-- 1.0, 65504 (max half), 65519 (rounds down), 65520 (tie: up to inf), 2^-14
-- (smallest normal), 2^-24 (smallest subnormal), 2^-25 (tie: to 0),
-- 1 + 2^-11 (tie: to even 1.0), 1 + 3*2^-11 (tie: up), -2.0, +inf, NaN.
example : [0x3F800000, 0x477FE000, 0x477FEF00, 0x477FF000, 0x38800000, 0x33800000, 0x33000000,
           0x3F801000, 0x3F803000, 0xC0000000, 0x7F800000, 0x7FC00000].map f32ToF16Ref
        = [0x3C00, 0x7BFF, 0x7BFF, 0x7C00, 0x0400, 0x0001, 0x0000,
           0x3C00, 0x3C02, 0xC000, 0x7C00, 0x7E00] := by native_decide

end Ggml.SlangCodegen.Conv
