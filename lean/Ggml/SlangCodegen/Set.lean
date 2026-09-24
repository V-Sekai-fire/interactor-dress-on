import LeanSlang
import Ggml.SlangCodegen.Common

/-!
# `Ggml.SlangCodegen.Set` — ggml SET, f32

`ggml_set(a, b, nb1, nb2, nb3, offset)` (and `ggml_set_1d`, `ggml_set_2d`,
the in-place forms): dst is `a` with `b` written into the window of it
that starts `offset` bytes in and has `b`'s shape with row strides `nb1,
nb2, nb3` (bytes; `nb0` is the element size). ggml-cpu
(`ggml_compute_forward_set_f32`) copies `a` into dst, then copies `b`'s
rows in through those strides. rf-detr sets box columns into a `[4, n]`
tensor (`decoder.cpp`) and each attention head's output into its
`[d_model, n]` slice (`deform_attn.cpp`).

One dispatch, one thread per destination element, a select rather than
two copies: `dst[i] = b[j]` when `i` is the window element `j`, else
`a[i]`. dst and `a` are contiguous (ggml-cpu asserts it), so an element's
position in the window is its linear index `l = i0 + i1 ne0 + i2 ne0 ne1 +
i3 ne0 ne1 ne2` less the offset, and the window index is recovered by
division: `j3 = r / nb3, r %= nb3, j2 = r / nb2, r %= nb2, j1 = r / nb1,
j0 = r % nb1` (strides and offset in elements, derived words 55-58 from
the packer), then `j` is in the window iff each `j_k < b.ne[k]`. That
inverse is exact only when the window is nested (`nb1 >= ne10, nb2 >=
ne11 nb1, nb3 >= ne12 nb2`), which `guest/ggml-rd/ops/set.cpp` requires:
elsewhere two window elements could share an address and ggml-cpu's
answer would be its own write order. In place (dst is `a`'s view) each
thread reads its own element of `a` before writing it. A bit copy, so
bit-exact.
-/

namespace Ggml.SlangCodegen.Set

open LeanSlang
open Ggml.SlangCodegen.Common

def threadgroup : Nat := 256

/-- Derived words: the window's strides and offset, in elements. -/
def wNb1 : Nat := 55
def wNb2 : Nat := 56
def wNb3 : Nat := 57
def wOffset : Nat := 58

example : wDerived + 2 = wNb1 ∧ wOffset < wordsPerSlot := by native_decide

def fl : SlangType := .scalar .float
def ui : SlangType := .scalar .uint
def lt (a b : SlangExpr) : SlangExpr := .bin "<" a b
def ge (a b : SlangExpr) : SlangExpr := .bin ">=" a b
def sub (a b : SlangExpr) : SlangExpr := .bin "-" a b
def land (a b : SlangExpr) : SlangExpr := .bin "&&" a b

def body : List SlangStmt :=
  let s0 := wSrc 0
  let s1 := wSrc 1
  let n (k : Nat) : SlangExpr := pwN (wDst + tNe + k)
  let inWindow :=
    land (land (lt (v "j0") (pwN (s1 + tNe))) (lt (v "j1") (pwN (s1 + tNe + 1))))
         (land (lt (v "j2") (pwN (s1 + tNe + 2))) (lt (v "j3") (pwN (s1 + tNe + 3))))
  [ .decl ui "i0", .decl ui "i1", .decl ui "i2", .decl ui "i3"
  , .expr (.call "unravel4" [v "e", u wDst, v "i0", v "i1", v "i2", v "i3"])
  , .declInit ui "d" (.call "off4" [u wDst, v "i0", v "i1", v "i2", v "i3"])
  , .declInit ui "l" (add (add (add (v "i0") (mul (v "i1") (n 0)))
                                (mul (v "i2") (mul (n 0) (n 1))))
                           (mul (v "i3") (mul (mul (n 0) (n 1)) (n 2))))
  , .declInit fl "x" (.index (v "s0") (.call "off4" [u s0, v "i0", v "i1", v "i2", v "i3"]))
  , .declInit ui "o" (pwN wOffset)
  , .ifNoElse (ge (v "l") (v "o"))
      [ .declInit ui "r" (sub (v "l") (v "o"))
      , .declInit ui "j3" (udiv (v "r") (pwN wNb3))
      , .assign (v "r") (urem (v "r") (pwN wNb3))
      , .declInit ui "j2" (udiv (v "r") (pwN wNb2))
      , .assign (v "r") (urem (v "r") (pwN wNb2))
      , .declInit ui "j1" (udiv (v "r") (pwN wNb1))
      , .declInit ui "j0" (urem (v "r") (pwN wNb1))
      , .ifNoElse inWindow
          [ .assign (v "x") (.index (v "s1") (.call "off4" [u s1, v "j0", v "j1", v "j2", v "j3"])) ] ]
  , .assign (.index (v "dst") (v "d")) (v "x") ]

def setF32 : SlangShaderModule :=
  kernelModule .float .float .uint .float [fnPw, fnUnravel4, fnOff4] (entry1D threadgroup body)

def kernels : List (String × SlangShaderModule) :=
  [ ("set_f32", setF32) ]

example : (LeanSlang.emit setF32).endsWith
"  uint i0;
  uint i1;
  uint i2;
  uint i3;
  unravel4(e, 1u, i0, i1, i2, i3);
  uint d = off4(1u, i0, i1, i2, i3);
  uint l = (((i0 + (i1 * pw(1u))) + (i2 * (pw(1u) * pw(2u)))) + (i3 * ((pw(1u) * pw(2u)) * pw(3u))));
  float x = s0[off4(10u, i0, i1, i2, i3)];
  uint o = pw(58u);
  if ((l >= o)) {
    uint r = (l - o);
    uint j3 = (r / pw(57u));
    r = (r % pw(57u));
    uint j2 = (r / pw(56u));
    r = (r % pw(56u));
    uint j1 = (r / pw(55u));
    uint j0 = (r % pw(55u));
    if ((((j0 < pw(19u)) && (j1 < pw(20u))) && ((j2 < pw(21u)) && (j3 < pw(22u))))) {
      x = s1[off4(19u, j0, j1, j2, j3)];
    }
  }
  dst[d] = x;
}" := by native_decide

example : ((LeanSlang.emit setF32).splitOn "RWStructuredBuffer<float> s1;").length = 2 := by native_decide

end Ggml.SlangCodegen.Set
