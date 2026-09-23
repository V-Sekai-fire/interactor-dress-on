import LeanSlang
import Ggml.SlangCodegen.Common
import Ggml.SlangCodegen.Move

/-!
# `Ggml.SlangCodegen.Concat` — ggml CONCAT along any dimension

`dst = concat(src0, src1, dim)`: src0 and src1 agree on every dimension
but `dim`, and dst's `dim` is the sum. One thread per destination element;
no word says which dimension: an index is past src0 in at most one
dimension (every other i_k < src0.ne_k = dst.ne_k), so

    i inside src0 (i_k < src0.ne_k for all k)  ->  src0[i]
    else                                       ->  src1[i - src0.ne on the dim that is past]

which also makes the packer's permutation of the blocks (iteration order,
`Move`) free. Same type on all three; the skin-tokens KV cache is the hot
case (f32, dims 2 and 3, about 400k per run).

    concat_b32   f32, i32        concat_b16   f16, bf16, i16
-/

namespace Ggml.SlangCodegen.Concat

open LeanSlang
open Ggml.SlangCodegen.Common
open Ggml.SlangCodegen.Move

def valConcat (k : Kind) : SlangFunctionDecl :=
  let a (d : Nat) := v ("a" ++ toString d)
  let i (d : Nat) := v ("i" ++ toString d)
  let back (d : Nat) := .ternary (ge (i d) (a d)) (sub (i d) (a d)) (i d)
  fnVal
    [ .declInit tU "a0" (pwN (wSrc 0 + tNe))
    , .declInit tU "a1" (pwN (wSrc 0 + tNe + 1))
    , .declInit tU "a2" (pwN (wSrc 0 + tNe + 2))
    , .declInit tU "a3" (pwN (wSrc 0 + tNe + 3))
    , .ifNoElse (land (land (land (lt (i 0) (a 0)) (lt (i 1) (a 1))) (lt (i 2) (a 2))) (lt (i 3) (a 3)))
        [ .ret (some (load k "s0" (.call "off4" ([u (wSrc 0)] ++ iv)))) ]
    , .ret (some (load k "s1" (.call "off4" [u (wSrc 1), back 0, back 1, back 2, back 3]))) ]

def shader (k : Kind) : SlangShaderModule :=
  moveModule (is16 k) (if is16 k then [fnLd16 "s0", fnLd16 "s1"] else []) (valConcat k)

def concatB32 : SlangShaderModule := shader .b32
def concatB16 : SlangShaderModule := shader .b16

def kernels : List (String × SlangShaderModule) :=
  [ ("concat_b32", concatB32)
  , ("concat_b16", concatB16) ]

def valB32Text : String :=
"uint val(uint i0, uint i1, uint i2, uint i3) {
  uint a0 = pw(10u);
  uint a1 = pw(11u);
  uint a2 = pw(12u);
  uint a3 = pw(13u);
  if (((((i0 < a0) && (i1 < a1)) && (i2 < a2)) && (i3 < a3))) {
    return s0[off4(10u, i0, i1, i2, i3)];
  }
  return s1[off4(19u, ((i0 >= a0) ? (i0 - a0) : i0), ((i1 >= a1) ? (i1 - a1) : i1), ((i2 >= a2) ? (i2 - a2) : i2), ((i3 >= a3) ? (i3 - a3) : i3))];
}"
example : emitFunction (valConcat .b32) = valB32Text := by native_decide

def valB16Text : String :=
  (valB32Text.replace "return s0[off4(10u, i0, i1, i2, i3)];" "return ld16_s0(off4(10u, i0, i1, i2, i3));").replace
    "return s1[off4(19u, ((i0 >= a0) ? (i0 - a0) : i0), ((i1 >= a1) ? (i1 - a1) : i1), ((i2 >= a2) ? (i2 - a2) : i2), ((i3 >= a3) ? (i3 - a3) : i3))];"
    "return ld16_s1(off4(19u, ((i0 >= a0) ? (i0 - a0) : i0), ((i1 >= a1) ? (i1 - a1) : i1), ((i2 >= a2) ? (i2 - a2) : i2), ((i3 >= a3) ? (i3 - a3) : i3)));"
example : emitFunction (valConcat .b16) = valB16Text := by native_decide
example : valB16Text ≠ valB32Text := by native_decide

example : emit concatB32 = assemble [] valB32Text false := by native_decide
example : emit concatB16 = assemble [ld16S0Text, ld16S0Text.replace "s0" "s1"] valB16Text true := by native_decide

end Ggml.SlangCodegen.Concat
