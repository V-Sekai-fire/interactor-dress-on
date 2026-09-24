import LeanSlang
import Ggml.SlangCodegen.Common

/-!
# `Ggml.SlangCodegen.Argsort` — ggml ARGSORT, f32 rows to i32 indices

`ggml_argsort(a, order)`: dst i32 of src0's shape; row `(i1, i2, i3)` of
dst holds the indices `0 .. ne0-1` of that row of src0 sorted by value,
ascending (`GGML_SORT_ORDER_ASC`, op_params word 37 = 0) or descending
(1). `ggml_argsort_top_k(a, k)` is the descending sort with a `[k, ..]`
view on it (rf-detr's `decoder.cpp`, the top queries by score).

A rank kernel, not a sort: one thread per destination element `(j, i1,
i2, i3)` counts how many elements `k` of the row sort before element `j`,

    before(k, j) = (x[k] < x[j]) || (x[k] == x[j] && k < j)     (ASC; `>` for DESC)

and writes `dst[rank, i1, i2, i3] = j`. Ties break by index, so the ranks
of a row are a permutation of `0 .. ne0-1` and every destination element
is written by exactly one thread; no group memory, so the cpp target
runs the same module. O(ne0^2) reads per row, spread over ne0 threads:
the packer (`guest/ggml-rd/ops/argsort.cpp`) refuses rows longer than its
cap so a graph cannot stall the GPU on a sort it never meant.

ggml-cpu (`ggml_compute_forward_argsort_f32`) is `std::sort` with
`data[a] < data[b]` (or `>`): the same order wherever the row has no ties,
and the tie order of an unstable sort otherwise (test-backend-ops fills
argsort rows with unique values). A NaN in the row is not supported by
either (`std::sort` on a NaN is not a strict weak order; here the NaN's
comparisons are all false, so its rank is its index among equal keys and
other ranks may collide).
-/

namespace Ggml.SlangCodegen.Argsort

open LeanSlang
open Ggml.SlangCodegen.Common

def threadgroup : Nat := 256

def fl : SlangType := .scalar .float
def ui : SlangType := .scalar .uint
def lt (a b : SlangExpr) : SlangExpr := .bin "<" a b
def gt (a b : SlangExpr) : SlangExpr := .bin ">" a b
def eq (a b : SlangExpr) : SlangExpr := .bin "==" a b
def neq (a b : SlangExpr) : SlangExpr := .bin "!=" a b
def land (a b : SlangExpr) : SlangExpr := .bin "&&" a b
def lor (a b : SlangExpr) : SlangExpr := .bin "||" a b

def body : List SlangStmt :=
  let s0 := wSrc 0
  let ld (i : SlangExpr) : SlangExpr := .index (v "s0") (add (v "xb") (mul i (v "xs")))
  [ .decl ui "i0", .decl ui "i1", .decl ui "i2", .decl ui "i3"
  , .expr (.call "unravel4" [v "e", u wDst, v "i0", v "i1", v "i2", v "i3"])
  , .declInit ui "xb" (.call "off4" [u s0, u 0, v "i1", v "i2", v "i3"])
  , .declInit ui "xs" (pwN (s0 + tNb))
  , .declInit ui "n" (pwN (s0 + tNe))
  , .declInit (.scalar .bool) "desc" (neq (pwN wOpParams) (u 0))
  , .declInit fl "xj" (ld (v "i0"))
  , .declInit ui "rank" (u 0)
  , .forCount "k" (u 0) (v "n")
      [ .declInit fl "xk" (ld (v "k"))
      , .declInit (.scalar .bool) "before"
          (.ternary (v "desc") (gt (v "xk") (v "xj")) (lt (v "xk") (v "xj")))
      , .ifNoElse (lor (v "before") (land (eq (v "xk") (v "xj")) (lt (v "k") (v "i0"))))
          [ .assign (v "rank") (add (v "rank") (u 1)) ] ]
  , .assign (.index (v "dst") (.call "off4" [u wDst, v "rank", v "i1", v "i2", v "i3"])) (v "i0") ]

/-- dst is a `uint` word buffer: the i32 index bits. -/
def argsortF32 : SlangShaderModule :=
  kernelModule .float .uint .uint .uint [fnPw, fnUnravel4, fnOff4] (entry1D threadgroup body)

def kernels : List (String × SlangShaderModule) :=
  [ ("argsort_f32", argsortF32) ]

example : (LeanSlang.emit argsortF32).endsWith
"  uint i0;
  uint i1;
  uint i2;
  uint i3;
  unravel4(e, 1u, i0, i1, i2, i3);
  uint xb = off4(10u, 0u, i1, i2, i3);
  uint xs = pw(14u);
  uint n = pw(10u);
  bool desc = (pw(37u) != 0u);
  float xj = s0[(xb + (i0 * xs))];
  uint rank = 0u;
  for (uint k = 0u; k < n; ++k) {
    float xk = s0[(xb + (k * xs))];
    bool before = (desc ? (xk > xj) : (xk < xj));
    if ((before || ((xk == xj) && (k < i0)))) {
      rank = (rank + 1u);
    }
  }
  dst[off4(1u, rank, i1, i2, i3)] = i0;
}" := by native_decide

example : ((LeanSlang.emit argsortF32).splitOn "RWStructuredBuffer<uint> dst;").length = 2 := by
  native_decide

end Ggml.SlangCodegen.Argsort
