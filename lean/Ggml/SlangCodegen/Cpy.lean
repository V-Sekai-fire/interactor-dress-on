import LeanSlang
import Ggml.SlangCodegen.Common
import Ggml.SlangCodegen.Move

/-!
# `Ggml.SlangCodegen.Cpy` — ggml CPY, DUP and CONT among f32, f16, bf16

One kernel per (source, destination) element kind; the packer
(`guest/ggml-rd/ops/cpy.cpp`) serves CPY (into src1's view), DUP and CONT
with them. Both sides may have any strides, and the shapes may differ (a
reshaping copy: ggml only asks for equal element counts): the destination
element at iteration index (i0..i3) has ggml linear index

    l = i0*w56 + i1*w57 + i2*w58 + i3*w59

(words 56-59 are the original linear weights of the permuted iteration
dimensions, `Move.wLinW`), and its source is element `l` of src0 in ggml
order: `unravel(l)` over src0's ne, then src0's strides.

    cpy_b32        f32 -> f32, i32 -> i32   bits
    cpy_b16        f16 -> f16, bf16 -> bf16, i16 -> i16   bits
    cpy_f32_f16    cpy_f32_bf16    cpy_f16_f32    cpy_bf16_f32
    cpy_f16_bf16   cpy_bf16_f16   (through f32, as ggml-cpu)
-/

namespace Ggml.SlangCodegen.Cpy

open LeanSlang
open Ggml.SlangCodegen.Common
open Ggml.SlangCodegen.Move

/-- The source element of destination index (i0..i3), converted. -/
def valCpy (s d : Kind) : SlangFunctionDecl :=
  fnVal
    [ .declInit tU "l" (add (add (add (mul (v "i0") (pwN (wLinW 0))) (mul (v "i1") (pwN (wLinW 1))))
                                  (mul (v "i2") (pwN (wLinW 2))))
                             (mul (v "i3") (pwN (wLinW 3))))
    , .decl tU "j0", .decl tU "j1", .decl tU "j2", .decl tU "j3"
    , .expr (.call "unravel4" [v "l", u (wSrc 0), v "j0", v "j1", v "j2", v "j3"])
    , .ret (some (conv s d (load s "s0" (.call "off4" [u (wSrc 0), v "j0", v "j1", v "j2", v "j3"])))) ]

def dst16 (d : Kind) : Bool := is16 d

def shader (s d : Kind) : SlangShaderModule :=
  moveModule (dst16 d) (convHelpers s d ["s0"]) (valCpy s d)

def cpyB32 : SlangShaderModule := shader .b32 .b32
def cpyB16 : SlangShaderModule := shader .b16 .b16
def cpyF32F16 : SlangShaderModule := shader .f32 .f16
def cpyF32Bf16 : SlangShaderModule := shader .f32 .bf16
def cpyF16F32 : SlangShaderModule := shader .f16 .f32
def cpyBf16F32 : SlangShaderModule := shader .bf16 .f32
def cpyF16Bf16 : SlangShaderModule := shader .f16 .bf16
def cpyBf16F16 : SlangShaderModule := shader .bf16 .f16

def kernels : List (String × SlangShaderModule) :=
  [ ("cpy_b32", cpyB32)
  , ("cpy_b16", cpyB16)
  , ("cpy_f32_f16", cpyF32F16)
  , ("cpy_f32_bf16", cpyF32Bf16)
  , ("cpy_f16_f32", cpyF16F32)
  , ("cpy_bf16_f32", cpyBf16F32)
  , ("cpy_f16_bf16", cpyF16Bf16)
  , ("cpy_bf16_f16", cpyBf16F16) ]

/-! ## Pins: every kernel's text, from pinned pieces -/

/-- The val of cpy_b32; the others differ in the returned expression only. -/
def valB32Text : String :=
"uint val(uint i0, uint i1, uint i2, uint i3) {
  uint l = ((((i0 * pw(56u)) + (i1 * pw(57u))) + (i2 * pw(58u))) + (i3 * pw(59u)));
  uint j0;
  uint j1;
  uint j2;
  uint j3;
  unravel4(l, 10u, j0, j1, j2, j3);
  return s0[off4(10u, j0, j1, j2, j3)];
}"
example : emitFunction (valCpy .b32 .b32) = valB32Text := by native_decide

def ret (x : String) : String :=
  valB32Text.replace "return s0[off4(10u, j0, j1, j2, j3)];" ("return " ++ x ++ ";")

def ld : String := "ld16_s0(off4(10u, j0, j1, j2, j3))"
def w32 : String := "s0[off4(10u, j0, j1, j2, j3)]"

example : emitFunction (valCpy .b16 .b16) = ret ld := by native_decide
example : emitFunction (valCpy .f32 .f16) = ret ("f32_to_f16(" ++ w32 ++ ")") := by native_decide
example : emitFunction (valCpy .f32 .bf16) = ret ("f32_to_bf16(" ++ w32 ++ ")") := by native_decide
example : emitFunction (valCpy .f16 .f32) = ret ("f16_to_f32(" ++ ld ++ ")") := by native_decide
example : emitFunction (valCpy .bf16 .f32) = ret ("(" ++ ld ++ " << 16u)") := by native_decide
example : emitFunction (valCpy .f16 .bf16) = ret ("f32_to_bf16(f16_to_f32(" ++ ld ++ "))") := by native_decide
example : emitFunction (valCpy .bf16 .f16) = ret ("f32_to_f16((" ++ ld ++ " << 16u))") := by native_decide

example : emit cpyB32 = assemble [] valB32Text false := by native_decide
example : emit cpyB16 = assemble [ld16S0Text] (ret ld) true := by native_decide
example : emit cpyF32F16 = assemble [f32ToF16Text] (ret ("f32_to_f16(" ++ w32 ++ ")")) true := by native_decide
example : emit cpyF32Bf16 = assemble [f32ToBf16Text] (ret ("f32_to_bf16(" ++ w32 ++ ")")) true := by native_decide
example : emit cpyF16F32 = assemble [ld16S0Text, f16ToF32Text] (ret ("f16_to_f32(" ++ ld ++ ")")) false := by
  native_decide
example : emit cpyBf16F32 = assemble [ld16S0Text] (ret ("(" ++ ld ++ " << 16u)")) false := by native_decide
example : emit cpyF16Bf16 =
    assemble [ld16S0Text, f16ToF32Text, f32ToBf16Text] (ret ("f32_to_bf16(f16_to_f32(" ++ ld ++ "))")) true := by
  native_decide
example : emit cpyBf16F16 =
    assemble [ld16S0Text, f32ToF16Text] (ret ("f32_to_f16((" ++ ld ++ " << 16u))")) true := by
  native_decide

end Ggml.SlangCodegen.Cpy
