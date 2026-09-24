import LeanSlang
import Ggml.SlangCodegen.Common

/-!
# `Ggml.SlangCodegen.ConvDw` — ggml CONV_2D_DW (depthwise), f32

`ggml_conv_2d_dw_direct(kernel, input, s0, s1, p0, p1, d0, d1)`: kernel
`[KW, KH, 1, C]` (src0), input `[W, H, C, N]` (src1), dst `[OW, OH, C, N]`,
channel `c` of the output convolved with channel `c` of the kernel alone
(rf-detr's segmentation head, `segmentation.cpp`: 3x3, stride 1, pad 1).
One thread per destination element `(ox, oy, c, n)`, the shape of `Unary`
with two sources:

    sum = 0
    for ky in 0..KH: y = oy s1 + ky d1 - p1, skipped outside [0, H)
      for kx in 0..KW: x = ox s0 + kx d0 - p0, skipped outside [0, W)
        sum += kernel[kx, ky, 0, c] * input[x, y, c, n]
    dst[ox, oy, c, n] = sum

exactly ggml-cpu's `ggml_compute_forward_conv_2d_dw_whcn` and `_cwhn`
(the same taps in the same order, `sum += k * s`): every operand is read
through its strides (`off4`), so the WHCN and the channel-contiguous CWHN
layouts (ggml gives dst the input's layout) are the one kernel. The
padding test is unsigned, as `Conv`'s: `q = oy s1 + ky d1` is in range
iff `q >= p1` and `q - p1 < H` (the packer refuses negative padding and
non-positive stride or dilation). op_params words 37-42 are `s0 s1 p0 p1
d0 d1`. f32 sums in tap order. The host L2 reference is ggml-cpu built
with `-mfma`, whose compiler fuses `sum += k * s` into one FMA per tap
while the cpp emit is compiled without, so the two land one ULP apart
(NMSE ~4e-15 over test-backend-ops' shapes, not 0); the GPU fuses the
same tap.
-/

namespace Ggml.SlangCodegen.ConvDw

open LeanSlang
open Ggml.SlangCodegen.Common

def threadgroup : Nat := 256

def fl : SlangType := .scalar .float
def ui : SlangType := .scalar .uint
def fLit (x : Float) : SlangExpr := .litFloatExact x
def lt (a b : SlangExpr) : SlangExpr := .bin "<" a b
def ge (a b : SlangExpr) : SlangExpr := .bin ">=" a b
def sub (a b : SlangExpr) : SlangExpr := .bin "-" a b
def land (a b : SlangExpr) : SlangExpr := .bin "&&" a b

def body : List SlangStmt :=
  let k := wSrc 0
  let img := wSrc 1
  [ .decl ui "i0", .decl ui "i1", .decl ui "i2", .decl ui "i3"
  , .expr (.call "unravel4" [v "e", u wDst, v "i0", v "i1", v "i2", v "i3"])
  , .declInit ui "d" (.call "off4" [u wDst, v "i0", v "i1", v "i2", v "i3"])
  , .declInit ui "kw" (pwN (k + tNe))
  , .declInit ui "kh" (pwN (k + tNe + 1))
  , .declInit ui "iw" (pwN (img + tNe))
  , .declInit ui "ih" (pwN (img + tNe + 1))
  , .declInit ui "st0" (pwN wOpParams)
  , .declInit ui "st1" (pwN (wOpParams + 1))
  , .declInit ui "p0" (pwN (wOpParams + 2))
  , .declInit ui "p1" (pwN (wOpParams + 3))
  , .declInit ui "d0" (pwN (wOpParams + 4))
  , .declInit ui "d1" (pwN (wOpParams + 5))
  , .declInit fl "sum" (fLit 0.0)
  , .forCount "ky" (u 0) (v "kh")
      [ .declInit ui "qy" (add (mul (v "i1") (v "st1")) (mul (v "ky") (v "d1")))
      , .ifNoElse (land (ge (v "qy") (v "p1")) (lt (sub (v "qy") (v "p1")) (v "ih")))
          [ .forCount "kx" (u 0) (v "kw")
              [ .declInit ui "qx" (add (mul (v "i0") (v "st0")) (mul (v "kx") (v "d0")))
              , .ifNoElse (land (ge (v "qx") (v "p0")) (lt (sub (v "qx") (v "p0")) (v "iw")))
                  [ .assign (v "sum") (add (v "sum")
                      (mul (.index (v "s0") (.call "off4" [u k, v "kx", v "ky", u 0, v "i2"]))
                           (.index (v "s1") (.call "off4" [u img, sub (v "qx") (v "p0"), sub (v "qy") (v "p1"), v "i2", v "i3"])))) ] ] ] ]
  , .assign (.index (v "dst") (v "d")) (v "sum") ]

def convDwF32 : SlangShaderModule :=
  kernelModule .float .float .uint .float [fnPw, fnUnravel4, fnOff4] (entry1D threadgroup body)

def kernels : List (String × SlangShaderModule) :=
  [ ("conv_2d_dw_f32", convDwF32) ]

example : (LeanSlang.emit convDwF32).endsWith
"  uint i0;
  uint i1;
  uint i2;
  uint i3;
  unravel4(e, 1u, i0, i1, i2, i3);
  uint d = off4(1u, i0, i1, i2, i3);
  uint kw = pw(10u);
  uint kh = pw(11u);
  uint iw = pw(19u);
  uint ih = pw(20u);
  uint st0 = pw(37u);
  uint st1 = pw(38u);
  uint p0 = pw(39u);
  uint p1 = pw(40u);
  uint d0 = pw(41u);
  uint d1 = pw(42u);
  float sum = 0.0f;
  for (uint ky = 0u; ky < kh; ++ky) {
    uint qy = ((i1 * st1) + (ky * d1));
    if (((qy >= p1) && ((qy - p1) < ih))) {
      for (uint kx = 0u; kx < kw; ++kx) {
        uint qx = ((i0 * st0) + (kx * d0));
        if (((qx >= p0) && ((qx - p0) < iw))) {
          sum = (sum + (s0[off4(10u, kx, ky, 0u, i2)] * s1[off4(19u, (qx - p0), (qy - p1), i2, i3)]));
        }
      }
    }
  }
  dst[d] = sum;
}" := by native_decide

end Ggml.SlangCodegen.ConvDw
