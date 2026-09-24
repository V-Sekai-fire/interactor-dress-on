import LeanSlang
import Ggml.SlangCodegen.Common
import Ggml.SlangCodegen.Conv

/-!
# `Ggml.SlangCodegen.Conv2d` — ggml CONV_2D (`ggml_conv_2d_direct`), f32
and f16 kernel

`Conv`'s CONV_3D without the depth (family K7): src0 is the kernel
`[KW, KH, IC, OC]`, src1 the input `[W, H, IC, N]`, dst `[OW, OH, OC, N]`,
op_params `s0 s1 p0 p1 d0 d1` (words 37..42). For dst index
(ox, oy, oc, n):

    dst = sum over ic < IC, ky < KH, kx < KW, (x, y) inside the input, of
          src1[x, y, ic, n] * src0[kx, ky, ic, oc]
    x = ox*s0 + kx*d0 - p0,  y = oy*s1 + ky*d1 - p1

accumulated in f32 in ggml-cpu's im2col order (ic, ky, kx:
`ggml_compute_forward_conv_2d_impl` lays a patch out as
`ic*(KH*KW) + ky*KW + kx` and takes one `ggml_call_mul_mat` dot over it,
whose SIMD blocking is not sequential, so the sum agrees within
test_conv_2d's NMSE 5e-4, not bit for bit). For an f16 kernel ggml-cpu's
patch is f16, so the input is rounded to f16 before the product:
`conv2d_f16` does the same (`f16tof32(f32_to_f16(x))`, `Conv`'s exact
rounding). One thread per output, no barriers, no group memory (so no
Serial sibling); the kernel through its strides (ggml-cpu asserts it
contiguous, the packer too), the input through its.
-/

namespace Ggml.SlangCodegen.Conv2d

open LeanSlang
open Ggml.SlangCodegen.Common
open Ggml.SlangCodegen.Conv

/-- Threads per work group. -/
def threadgroup : Nat := 256

private def U : SlangType := .scalar .uint
private def F : SlangType := .scalar .float

/-- op_params words: s0 s1 p0 p1 d0 d1. -/
def wCv (k : Nat) : Nat := wOpParams + k

/-- The body; `f16` reads the kernel as f16 halves and rounds the input
    to f16 before the product. Every params word the loops use is read
    once into a local; the input and kernel offsets advance per loop level
    (channel, y), so the innermost step is two loads and one multiply-add. -/
def conv2dBody (f16 : Bool) : List SlangStmt :=
  let s0 := wSrc 0
  let kval : SlangExpr :=
    let idx := add (v "kby") (mul (v "kx") (v "kn0"))
    if f16 then .call "ld_f16_s0" [idx] else .index (v "s0") idx
  let xval : SlangExpr :=
    let x := .index (v "s1") (add (v "iby") (mul (sub (v "x") (v "px")) (v "in0")))
    if f16 then .call "f16tof32" [.call "f32_to_f16" [x]] else x
  let w (n : String) (k : Nat) : SlangStmt := .declInit U n (pwN k)
  [ .decl U "ox", .decl U "oy", .decl U "oc", .decl U "n"
  , .expr (.call "unravel4" [v "e", u wDst, v "ox", v "oy", v "oc", v "n"])
  , w "sx" (wCv 0), w "sy" (wCv 1)
  , w "px" (wCv 2), w "py" (wCv 3)
  , w "dx" (wCv 4), w "dy" (wCv 5)
  , w "kw" (s0 + tNe), w "kh" (s0 + tNe + 1), w "c" (s0 + tNe + 2)
  , w "iw" (wS1 + tNe), w "ih" (wS1 + tNe + 1)
  , w "kn0" (s0 + tNb), w "kn1" (s0 + tNb + 1), w "kn2" (s0 + tNb + 2), w "kn3" (s0 + tNb + 3)
  , w "in0" (wS1 + tNb), w "in1" (wS1 + tNb + 1), w "in2" (wS1 + tNb + 2), w "in3" (wS1 + tNb + 3)
  , .declInit U "ib" (add (pwN (wS1 + tOff)) (mul (v "n") (v "in3")))
  , .declInit U "kb" (add (pwN (s0 + tOff)) (mul (v "oc") (v "kn3")))
  , .declInit F "acc" (.litFloat 0.0)
  , .forCount "ic" (u 0) (v "c")
      [ .declInit U "ibc" (add (v "ib") (mul (v "ic") (v "in2")))
      , .declInit U "kbc" (add (v "kb") (mul (v "ic") (v "kn2")))
      , .forCount "ky" (u 0) (v "kh")
          [ .declInit U "y" (add (mul (v "oy") (v "sy")) (mul (v "ky") (v "dy")))
          , .ifNoElse (inside (v "y") (v "py") (v "ih"))
              [ .declInit U "iby" (add (v "ibc") (mul (sub (v "y") (v "py")) (v "in1")))
              , .declInit U "kby" (add (v "kbc") (mul (v "ky") (v "kn1")))
              , .forCount "kx" (u 0) (v "kw")
                  [ .declInit U "x" (add (mul (v "ox") (v "sx")) (mul (v "kx") (v "dx")))
                  , .ifNoElse (inside (v "x") (v "px") (v "iw"))
                      [ .assign (v "acc") (add (v "acc") (mul xval kval)) ] ] ] ] ]
  , .assign (.index (v "dst") (.call "off4" [u wDst, v "ox", v "oy", v "oc", v "n"])) (v "acc") ]

def conv2dF32 : SlangShaderModule :=
  kernelModule .float .float .uint .float
    [fnPw, fnUnravel4, fnOff4]
    (entry1D threadgroup (conv2dBody false))

def conv2dF16 : SlangShaderModule :=
  kernelModule .uint .float .uint .float
    [fnPw, fnUnravel4, fnOff4, fnLdF16 "s0", fnF32ToF16]
    (entry1D threadgroup (conv2dBody true))

/-- The kernels this module contributes, by their kernels.txt names. -/
def kernels : List (String × SlangShaderModule) :=
  [ ("conv2d_f32", conv2dF32)
  , ("conv2d_f16", conv2dF16) ]

/-! ## Pins: the entry in full; the declarations and helpers are CONV_3D's. -/

def expectedEntryF32 : String :=
"[shader(\"compute\")] [numthreads(256, 1, 1)]
void main(uint3 gid : SV_GroupID, uint3 lid : SV_GroupThreadID) {
  uint e = ((((gid.y * pw(54u)) + gid.x) * 256u) + lid.x);
  if ((e >= pw(53u))) {
    return;
  }
  uint ox;
  uint oy;
  uint oc;
  uint n;
  unravel4(e, 1u, ox, oy, oc, n);
  uint sx = pw(37u);
  uint sy = pw(38u);
  uint px = pw(39u);
  uint py = pw(40u);
  uint dx = pw(41u);
  uint dy = pw(42u);
  uint kw = pw(10u);
  uint kh = pw(11u);
  uint c = pw(12u);
  uint iw = pw(19u);
  uint ih = pw(20u);
  uint kn0 = pw(14u);
  uint kn1 = pw(15u);
  uint kn2 = pw(16u);
  uint kn3 = pw(17u);
  uint in0 = pw(23u);
  uint in1 = pw(24u);
  uint in2 = pw(25u);
  uint in3 = pw(26u);
  uint ib = (pw(27u) + (n * in3));
  uint kb = (pw(18u) + (oc * kn3));
  float acc = 0.000000;
  for (uint ic = 0u; ic < c; ++ic) {
    uint ibc = (ib + (ic * in2));
    uint kbc = (kb + (ic * kn2));
    for (uint ky = 0u; ky < kh; ++ky) {
      uint y = ((oy * sy) + (ky * dy));
      if (((y >= py) && ((y - py) < ih))) {
        uint iby = (ibc + ((y - py) * in1));
        uint kby = (kbc + (ky * kn1));
        for (uint kx = 0u; kx < kw; ++kx) {
          uint x = ((ox * sx) + (kx * dx));
          if (((x >= px) && ((x - px) < iw))) {
            acc = (acc + (s1[(iby + ((x - px) * in0))] * s0[(kby + (kx * kn0))]));
          }
        }
      }
    }
  }
  dst[off4(1u, ox, oy, oc, n)] = acc;
}"

def expectedEntryF16 : String :=
  expectedEntryF32.replace "(s1[(iby + ((x - px) * in0))] * s0[(kby + (kx * kn0))])"
    "(f16tof32(f32_to_f16(s1[(iby + ((x - px) * in0))])) * ld_f16_s0((kby + (kx * kn0))))"

/-- The head (declarations and helpers) is exactly the CONV_3D kernel's:
    its text up to the entry attribute. -/
def head (s : String) : String := (s.splitOn "[shader(").headD ""

example : LeanSlang.emit conv2dF32 = head expectedConv3dF32 ++ expectedEntryF32 := by native_decide
example : LeanSlang.emit conv2dF16 = head expectedConv3dF16 ++ expectedEntryF16 := by native_decide
example : (head expectedConv3dF32).endsWith "}\n\n" ∧ (head expectedConv3dF16).endsWith "}\n\n" := by
  native_decide
example : conv2dF32.entryPointName = "main" ∧ conv2dF16.entryPointName = "main" := by native_decide

end Ggml.SlangCodegen.Conv2d
