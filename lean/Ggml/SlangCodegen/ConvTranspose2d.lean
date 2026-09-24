import LeanSlang
import Ggml.SlangCodegen.Common
import Ggml.SlangCodegen.Conv

/-!
# `Ggml.SlangCodegen.ConvTranspose2d` — ggml CONV_TRANSPOSE_2D
(`ggml_conv_transpose_2d_p0`), f16 or f32 kernel

src0 is the kernel `[KW, KH, OC, IC]`, src1 the input `[SW, SH, IC, 1]`,
dst `[(SW-1)*s + KW, (SH-1)*s + KH, OC, 1]`, `s` the stride (op_params
word 0). ggml-cpu (`ggml_compute_forward_conv_transpose_2d_impl`) permutes
both operands so the channels are innermost, then for each output
channel, each input pixel (i11, i10) in that order and each kernel tap
(i01, i00) takes one dot over the channels and adds it to the output
pixel `(i10*s + i00, i11*s + i01)`. This kernel is the gather of the same
sums: one thread per output (ox, oy, oc),

    acc = 0
    for i11 < SH, i10 < SW  with  kx = ox - i10*s in [0, KW), ky = oy - i11*s in [0, KH):
        v = Σ_ic  src1[i10, i11, ic] * src0[kx, ky, oc, ic]     (f32, sequential)
        acc = acc + v

The outer order (i11, then i10) is ggml-cpu's order of `+=` into that
pixel, and it starts from 0 as ggml-cpu's zeroed dst does. The inner dot
differs: ggml-cpu's `ggml_vec_dot_f16` / `_f32` accumulates in double
(its scalar path) or by SIMD lanes, then rounds `v` to f32; this kernel
sums the f32 products sequentially. So results agree within
test_conv_transpose_2d's NMSE 5e-4, not bit for bit
(`tests/ggml_rd_kernels/cases/seethrough.cpp` records the measured
value). For an f16 kernel ggml-cpu converts the input to f16 first;
`conv_transpose_2d_f16` does the same (`Conv`'s `f32_to_f16`). Every
operand through its strides; the packer requires ggml-cpu's own
assumptions (nb00 = one element, nb10 = one element, ne13 = 1: ggml-cpu
ignores an input batch beyond the first).
-/

namespace Ggml.SlangCodegen.ConvTranspose2d

open LeanSlang
open Ggml.SlangCodegen.Common
open Ggml.SlangCodegen.Conv

/-- Threads per work group. -/
def threadgroup : Nat := 256

private def U : SlangType := .scalar .uint
private def F : SlangType := .scalar .float

def body (f16 : Bool) : List SlangStmt :=
  let s0 := wSrc 0
  let kval : SlangExpr :=
    let idx := add (v "kb") (mul (v "ic") (v "kn3"))
    if f16 then .call "ld_f16_s0" [idx] else .index (v "s0") idx
  let xval : SlangExpr :=
    let x := .index (v "s1") (add (v "ib") (mul (v "ic") (v "in2")))
    if f16 then .call "f16tof32" [.call "f32_to_f16" [x]] else x
  let w (n : String) (k : Nat) : SlangStmt := .declInit U n (pwN k)
  [ .decl U "ox", .decl U "oy", .decl U "oc", .decl U "i3"
  , .expr (.call "unravel4" [v "e", u wDst, v "ox", v "oy", v "oc", v "i3"])
  , w "st" wOpParams
  , w "kw" (s0 + tNe), w "kh" (s0 + tNe + 1), w "c" (s0 + tNe + 3)
  , w "sw" (wS1 + tNe), w "sh" (wS1 + tNe + 1)
  , w "kn0" (s0 + tNb), w "kn1" (s0 + tNb + 1), w "kn2" (s0 + tNb + 2), w "kn3" (s0 + tNb + 3)
  , w "in0" (wS1 + tNb), w "in1" (wS1 + tNb + 1), w "in2" (wS1 + tNb + 2)
  , .declInit U "koc" (add (pwN (s0 + tOff)) (mul (v "oc") (v "kn2")))
  , .declInit F "acc" (.litFloat 0.0)
  , .forCount "i11" (u 0) (v "sh")
      [ .declInit U "y" (mul (v "i11") (v "st"))
      , .ifNoElse (land (ge (v "oy") (v "y")) (lt (sub (v "oy") (v "y")) (v "kh")))
          [ .declInit U "ky" (sub (v "oy") (v "y"))
          , .forCount "i10" (u 0) (v "sw")
              [ .declInit U "x" (mul (v "i10") (v "st"))
              , .ifNoElse (land (ge (v "ox") (v "x")) (lt (sub (v "ox") (v "x")) (v "kw")))
                  [ .declInit U "kx" (sub (v "ox") (v "x"))
                  , .declInit U "ib" (add (pwN (wS1 + tOff)) (add (mul (v "i10") (v "in0")) (mul (v "i11") (v "in1"))))
                  , .declInit U "kb" (add (v "koc") (add (mul (v "kx") (v "kn0")) (mul (v "ky") (v "kn1"))))
                  , .declInit F "vdot" (.litFloat 0.0)
                  , .forCount "ic" (u 0) (v "c")
                      [ .assign (v "vdot") (add (v "vdot") (mul xval kval)) ]
                  , .assign (v "acc") (add (v "acc") (v "vdot")) ] ] ] ]
  , .assign (.index (v "dst") (.call "off4" [u wDst, v "ox", v "oy", v "oc", v "i3"])) (v "acc") ]

def convTranspose2dF32 : SlangShaderModule :=
  kernelModule .float .float .uint .float
    [fnPw, fnUnravel4, fnOff4]
    (entry1D threadgroup (body false))

def convTranspose2dF16 : SlangShaderModule :=
  kernelModule .uint .float .uint .float
    [fnPw, fnUnravel4, fnOff4, fnLdF16 "s0", fnF32ToF16]
    (entry1D threadgroup (body true))

/-- The kernels this module contributes, by their kernels.txt names. -/
def kernels : List (String × SlangShaderModule) :=
  [ ("conv_transpose_2d_f32", convTranspose2dF32)
  , ("conv_transpose_2d_f16", convTranspose2dF16) ]

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
  uint i3;
  unravel4(e, 1u, ox, oy, oc, i3);
  uint st = pw(37u);
  uint kw = pw(10u);
  uint kh = pw(11u);
  uint c = pw(13u);
  uint sw = pw(19u);
  uint sh = pw(20u);
  uint kn0 = pw(14u);
  uint kn1 = pw(15u);
  uint kn2 = pw(16u);
  uint kn3 = pw(17u);
  uint in0 = pw(23u);
  uint in1 = pw(24u);
  uint in2 = pw(25u);
  uint koc = (pw(18u) + (oc * kn2));
  float acc = 0.000000;
  for (uint i11 = 0u; i11 < sh; ++i11) {
    uint y = (i11 * st);
    if (((oy >= y) && ((oy - y) < kh))) {
      uint ky = (oy - y);
      for (uint i10 = 0u; i10 < sw; ++i10) {
        uint x = (i10 * st);
        if (((ox >= x) && ((ox - x) < kw))) {
          uint kx = (ox - x);
          uint ib = (pw(27u) + ((i10 * in0) + (i11 * in1)));
          uint kb = (koc + ((kx * kn0) + (ky * kn1)));
          float vdot = 0.000000;
          for (uint ic = 0u; ic < c; ++ic) {
            vdot = (vdot + (s1[(ib + (ic * in2))] * s0[(kb + (ic * kn3))]));
          }
          acc = (acc + vdot);
        }
      }
    }
  }
  dst[off4(1u, ox, oy, oc, i3)] = acc;
}"

def expectedEntryF16 : String :=
  expectedEntryF32.replace "(s1[(ib + (ic * in2))] * s0[(kb + (ic * kn3))])"
    "(f16tof32(f32_to_f16(s1[(ib + (ic * in2))])) * ld_f16_s0((kb + (ic * kn3))))"

def head (s : String) : String := (s.splitOn "[shader(").headD ""

example : LeanSlang.emit convTranspose2dF32 = head expectedConv3dF32 ++ expectedEntryF32 := by native_decide
example : LeanSlang.emit convTranspose2dF16 = head expectedConv3dF16 ++ expectedEntryF16 := by native_decide
example : convTranspose2dF32.entryPointName = "main" ∧ convTranspose2dF16.entryPointName = "main" := by
  native_decide

end Ggml.SlangCodegen.ConvTranspose2d
