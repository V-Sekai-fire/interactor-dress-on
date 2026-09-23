import LeanSlang
import Ggml.SlangCodegen.Common
import Ggml.SlangCodegen.Move

/-!
# `Ggml.SlangCodegen.GetRows` — ggml GET_ROWS, {f32, i32, f16, bf16} x i32

`dst[i0, i1, i2, i3] = src0[i0, rows[i1, i2, i3], i2, i3]`, rows the i32
src1 (ggml asserts `src0.ne2 = src1.ne1`, `src0.ne3 = src1.ne2`,
`src1.ne3 = 1`), the destination f32 (i32 for an i32 src0). One thread per
destination element (`Move.entry32`), in dst's own order; every operand
strided. The row index is read once per element, from the cache.

    get_rows_b32    f32 -> f32, i32 -> i32   bits
    get_rows_f16    f16 -> f32 (exact)
    get_rows_bf16   bf16 -> f32 (exact)
-/

namespace Ggml.SlangCodegen.GetRows

open LeanSlang
open Ggml.SlangCodegen.Common
open Ggml.SlangCodegen.Move

def valGetRows (s : Kind) : SlangFunctionDecl :=
  fnVal
    [ .declInit tU "r" (.index (v "s1") (.call "off4" [u (wSrc 1), v "i1", v "i2", v "i3", u 0]))
    , .ret (some (conv s .f32 (load s "s0" (.call "off4" [u (wSrc 0), v "i0", v "r", v "i2", v "i3"])))) ]

def shader (s : Kind) : SlangShaderModule :=
  moveModule false (convHelpers s .f32 ["s0"]) (valGetRows s)

def getRowsB32 : SlangShaderModule := shader .b32
def getRowsF16 : SlangShaderModule := shader .f16
def getRowsBf16 : SlangShaderModule := shader .bf16

def kernels : List (String × SlangShaderModule) :=
  [ ("get_rows_b32", getRowsB32)
  , ("get_rows_f16", getRowsF16)
  , ("get_rows_bf16", getRowsBf16) ]

def valB32Text : String :=
"uint val(uint i0, uint i1, uint i2, uint i3) {
  uint r = s1[off4(19u, i1, i2, i3, 0u)];
  return s0[off4(10u, i0, r, i2, i3)];
}"
example : emitFunction (valGetRows .b32) = valB32Text := by native_decide

def ret (x : String) : String :=
  valB32Text.replace "return s0[off4(10u, i0, r, i2, i3)];" ("return " ++ x ++ ";")
def ld : String := "ld16_s0(off4(10u, i0, r, i2, i3))"

example : emit getRowsB32 = assemble [] valB32Text false := by native_decide
example : emit getRowsF16 = assemble [ld16S0Text, f16ToF32Text] (ret ("f16_to_f32(" ++ ld ++ ")")) false := by
  native_decide
example : emit getRowsBf16 = assemble [ld16S0Text] (ret ("(" ++ ld ++ " << 16u)")) false := by native_decide

end Ggml.SlangCodegen.GetRows
