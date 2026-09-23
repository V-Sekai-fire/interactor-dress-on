import LeanSlang
import Ggml.SlangCodegen.Common
import Ggml.SlangCodegen.Move

/-!
# `Ggml.SlangCodegen.Repeat` — ggml REPEAT

`dst[i] = src0[i mod src0.ne]` (`ggml_can_repeat(src0, dst)`: each of
src0's dims divides dst's). One thread per destination element, any
strides; same type on both sides.

    repeat_b32   f32, i32        repeat_b16   f16, bf16, i16
-/

namespace Ggml.SlangCodegen.Repeat

open LeanSlang
open Ggml.SlangCodegen.Common
open Ggml.SlangCodegen.Move

def valRepeat (k : Kind) : SlangFunctionDecl :=
  let s0 := wSrc 0
  fnVal
    [ .ret (some (load k "s0" (.call "off4"
        [ u s0
        , urem (v "i0") (pwN (s0 + tNe))
        , urem (v "i1") (pwN (s0 + tNe + 1))
        , urem (v "i2") (pwN (s0 + tNe + 2))
        , urem (v "i3") (pwN (s0 + tNe + 3)) ]))) ]

def shader (k : Kind) : SlangShaderModule :=
  moveModule (is16 k) (if is16 k then [fnLd16 "s0"] else []) (valRepeat k)

def repeatB32 : SlangShaderModule := shader .b32
def repeatB16 : SlangShaderModule := shader .b16

def kernels : List (String × SlangShaderModule) :=
  [ ("repeat_b32", repeatB32)
  , ("repeat_b16", repeatB16) ]

def valB32Text : String :=
"uint val(uint i0, uint i1, uint i2, uint i3) {
  return s0[off4(10u, (i0 % pw(10u)), (i1 % pw(11u)), (i2 % pw(12u)), (i3 % pw(13u)))];
}"
example : emitFunction (valRepeat .b32) = valB32Text := by native_decide

def valB16Text : String :=
"uint val(uint i0, uint i1, uint i2, uint i3) {
  return ld16_s0(off4(10u, (i0 % pw(10u)), (i1 % pw(11u)), (i2 % pw(12u)), (i3 % pw(13u))));
}"
example : emitFunction (valRepeat .b16) = valB16Text := by native_decide

example : emit repeatB32 = assemble [] valB32Text false := by native_decide
example : emit repeatB16 = assemble [ld16S0Text] valB16Text true := by native_decide

end Ggml.SlangCodegen.Repeat
