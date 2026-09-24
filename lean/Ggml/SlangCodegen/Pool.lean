import LeanSlang
import Ggml.SlangCodegen.Common

/-!
# `Ggml.SlangCodegen.Pool` — ggml POOL_1D, max and avg, f32

`ggml_pool_1d(a, op, k0, s0, p0)`: dst `[OW, ne1, ne2, ne3]` with
`OW = (ne0 + 2 p0 - k0) / s0 + 1`, each element the max (or mean) of the
window `[ow * s0 - p0, ow * s0 - p0 + k0)` of its row of src0, taps
outside the row skipped. One thread per destination element, the shape of
`Unary` (unravel over dst, `off4` on both sides, any strides), with the
window loop; the packer is `guest/ggml-rd/ops/pool.cpp`. interactor-nx-ggml's
`add_reduce_max_last_axis` and rf-detr's top-k score (`decoder.cpp`) are
the max pool with `k0 = s0 = ne0`, a max over the row.

ggml-cpu (`ggml_compute_forward_pool_1d_ksp`), per tap in window order:

    MAX  res = -FLT_MAX;  res = std::max(v, res)  = (v < res) ? res : v
    AVG  res = 0;         res += v;  count++;      res = count > 0 ? res / count : 0

written here with the same ternary (a NaN tap makes the max NaN on both)
and the same f32 sum in the same order, so the host emit is bit-exact. The
padding test is unsigned, as `Conv`'s: tap `q = ow * s0 + ki` (the source
index plus `p0`) is in the row iff `q >= p0` and `q - p0 < ne0`; op_params
words 38-40 are `k0, s0, p0` (word 37 is the op, which picks the kernel).
-/

namespace Ggml.SlangCodegen.Pool

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

/-- `-FLT_MAX`, ggml-cpu's initial max. -/
def negFltMax : Float := -3.4028234663852886e38

inductive Mode
  | max
  | avg
deriving BEq

/-- The body after the 1-D prologue. -/
def body (m : Mode) : List SlangStmt :=
  let s0 := wSrc 0
  let tap := .index (v "s0") (add (v "xb") (mul (sub (v "q") (v "p")) (v "xs")))
  let fold : List SlangStmt := match m with
    | .max => [ .assign (v "res") (.ternary (lt (v "val") (v "res")) (v "res") (v "val")) ]
    | .avg => [ .assign (v "res") (add (v "res") (v "val"))
              , .assign (v "count") (add (v "count") (u 1)) ]
  let result : SlangExpr := match m with
    | .max => v "res"
    | .avg => .ternary (.bin ">" (v "count") (u 0)) (.bin "/" (v "res") (.cast fl (v "count"))) (fLit 0.0)
  [ .decl ui "i0", .decl ui "i1", .decl ui "i2", .decl ui "i3"
  , .expr (.call "unravel4" [v "e", u wDst, v "i0", v "i1", v "i2", v "i3"])
  , .declInit ui "d" (.call "off4" [u wDst, v "i0", v "i1", v "i2", v "i3"])
  , .declInit ui "xb" (.call "off4" [u s0, u 0, v "i1", v "i2", v "i3"])
  , .declInit ui "xs" (pwN (s0 + tNb))
  , .declInit ui "n" (pwN (s0 + tNe))
  , .declInit ui "k" (pwN (wOpParams + 1))
  , .declInit ui "s" (pwN (wOpParams + 2))
  , .declInit ui "p" (pwN (wOpParams + 3))
  , .declInit ui "b" (mul (v "i0") (v "s"))
  , .declInit fl "res" (match m with | .max => fLit negFltMax | .avg => fLit 0.0) ] ++
  (match m with | .max => [] | .avg => [ .declInit ui "count" (u 0) ]) ++
  [ .forCount "ki" (u 0) (v "k")
      [ .declInit ui "q" (add (v "b") (v "ki"))
      , .ifNoElse (land (ge (v "q") (v "p")) (lt (sub (v "q") (v "p")) (v "n")))
          ([ .declInit fl "val" tap ] ++ fold) ]
  , .assign (.index (v "dst") (v "d")) result ]

def shader (m : Mode) : SlangShaderModule :=
  kernelModule .float .uint .uint .float [fnPw, fnUnravel4, fnOff4] (entry1D threadgroup (body m))

def poolMaxF32 : SlangShaderModule := shader .max
def poolAvgF32 : SlangShaderModule := shader .avg

/-- The kernels this module contributes, by their kernels.txt names. -/
def kernels : List (String × SlangShaderModule) :=
  [ ("pool_1d_max_f32", poolMaxF32)
  , ("pool_1d_avg_f32", poolAvgF32) ]

/-! ## Pins -/

example : (LeanSlang.emit poolMaxF32).endsWith
"  uint i0;
  uint i1;
  uint i2;
  uint i3;
  unravel4(e, 1u, i0, i1, i2, i3);
  uint d = off4(1u, i0, i1, i2, i3);
  uint xb = off4(10u, 0u, i1, i2, i3);
  uint xs = pw(14u);
  uint n = pw(10u);
  uint k = pw(38u);
  uint s = pw(39u);
  uint p = pw(40u);
  uint b = (i0 * s);
  float res = (-3.4028235e38f);
  for (uint ki = 0u; ki < k; ++ki) {
    uint q = (b + ki);
    if (((q >= p) && ((q - p) < n))) {
      float val = s0[(xb + ((q - p) * xs))];
      res = ((val < res) ? res : val);
    }
  }
  dst[d] = res;
}" := by native_decide

example : (LeanSlang.emit poolAvgF32).endsWith
"  float res = 0.0f;
  uint count = 0u;
  for (uint ki = 0u; ki < k; ++ki) {
    uint q = (b + ki);
    if (((q >= p) && ((q - p) < n))) {
      float val = s0[(xb + ((q - p) * xs))];
      res = (res + val);
      count = (count + 1u);
    }
  }
  dst[d] = ((count > 0u) ? (res / float(count)) : 0.0f);
}" := by native_decide

/-- The two share everything up to the accumulator. -/
example : ((LeanSlang.emit poolMaxF32).splitOn "float res =").head! =
    ((LeanSlang.emit poolAvgF32).splitOn "float res =").head! := by native_decide

end Ggml.SlangCodegen.Pool
