import LeanSlang
import Ggml.SlangCodegen.Common

/-!
# `Ggml.SlangCodegen.Rows` — row kernels: one work group per row, tree-reduced

Not a kernel. The pieces the row-wise ops (K3 `Norm`: NORM, RMS_NORM, MEAN;
K4 `SoftMax`) are built from. A row is dim 0 of src0; there are
`ne1 * ne2 * ne3` of them, one work group each:

    row          = gid.y * gx + gid.x          (word 54 = gx; rows >= word 55 return,
                                                the whole group at once)
    (r1, r2, r3) = unravel(row) over src0's ne1, ne2
    xb, xs       = src0's row start and element stride (in elements)
    yb, ys       = dst's row start and element stride

A **reduction** is grid-strided over the row by the 256 threads of the
group (64 in the `_t64` kernels for short rows, with 64 partials and a
six-step tree), then tree-reduced in `groupshared float sh[256]`:

    acc          = init
    i = t; while (i < n) { x = s0[xb + i*xs]; acc = acc ⊕ f(x); i += 256 }
    sh[t] = acc;                                       barrier
    for s in 128, 64, ..., 1: if (t < s) sh[t] = sh[t] ⊕ sh[t + s];   barrier
    r = sh[0];                                         barrier (sh is reused)

A **map** writes `dst[yb + j*ys] = g(x_j)`, grid-strided the same way. Thread
`t` reads (in every reduction) and writes (in the map) only the elements
`j ≡ t (mod 256)`, so a map in place (dst over src0) never overwrites an
element another thread still reads; the barriers are for `sh` alone.

**The Serial sibling** of each kernel is the same program for the cpp target,
which rejects `groupshared` and `GroupMemoryBarrierWithGroupSync` (E36107,
`Cloth.SlangCodegen.DotReduce`): one thread per work group (still one group
per row, so the packer and the grid are the same), `sh` a local array, and
each phase a loop over `t`. It adds the same partial sums in the same order
as its kernel (the strided partials, then the same tree), so the
host test (tests/ggml_rd_kernels, which runs the sibling in place of the
kernel) checks the GPU kernel's arithmetic, not a different summation.

Derived words: 55 = rows (the packer's `ne1 * ne2 * ne3`); 53/54 are the
grid's, from `grid_1d(p, rows * tg)`.
-/

namespace Ggml.SlangCodegen.Rows

open LeanSlang
open Ggml.SlangCodegen.Common

/-- Threads per work group of the parallel kernels (and the partial count
    both variants reduce over): 256, or 64 for short rows (the `_t64`
    kernels; the packer picks one by row length). -/
def threadgroup : Nat := 256
def threadgroupShort : Nat := 64

/-- Derived word 55: the number of rows (= work groups doing work). -/
def wRows : Nat := 55

example : wDerived ≤ wRows ∧ wRows < wDerived + nDerived ∧ wRows ≠ wThreads ∧ wRows ≠ wGroupsX := by
  native_decide

/-- The 256-thread kernel or its one-thread Serial sibling. -/
inductive Variant
  | par
  | ser
deriving BEq, Inhabited

def fl : SlangType := .scalar .float
def ui : SlangType := .scalar .uint

/-- How a reduction combines two partials. -/
inductive Combine
  | sum
  | max
deriving BEq, Inhabited

def combine : Combine → SlangExpr → SlangExpr → SlangExpr
  | .sum, a, b => add a b
  | .max, a, b => .call "max" [a, b]

def sub (a b : SlangExpr) : SlangExpr := .bin "-" a b
def fdiv (a b : SlangExpr) : SlangExpr := .bin "/" a b
def lt (a b : SlangExpr) : SlangExpr := .bin "<" a b
def fLit (x : Float) : SlangExpr := .litFloatExact x
def floatOf (e : SlangExpr) : SlangExpr := .cast fl e
def barrier : SlangStmt := .expr (.call "GroupMemoryBarrierWithGroupSync" [])
def sh (i : SlangExpr) : SlangExpr := .index (v "sh") i

/-- Element `i` of this row of s0. -/
def ldX (i : SlangExpr) : SlangExpr := .index (v "s0") (add (v "xb") (mul i (v "xs")))
/-- `dst[yb + i*ys] = val;` -/
def stY (i val : SlangExpr) : SlangStmt := .assign (.index (v "dst") (add (v "yb") (mul i (v "ys")))) val

/-- The row's geometry: n, the row index split over dims 1..3, the bases
    and element strides of src0 and dst. -/
def prologue : List SlangStmt :=
  [ .declInit ui "n" (pwN (wSrc 0 + tNe))
  , .declInit ui "r1" (urem (v "row") (pwN (wSrc 0 + tNe + 1)))
  , .declInit ui "rr" (udiv (v "row") (pwN (wSrc 0 + tNe + 1)))
  , .declInit ui "r2" (urem (v "rr") (pwN (wSrc 0 + tNe + 2)))
  , .declInit ui "r3" (udiv (v "rr") (pwN (wSrc 0 + tNe + 2)))
  , .declInit ui "xb" (.call "off4" [u (wSrc 0), u 0, v "r1", v "r2", v "r3"])
  , .declInit ui "xs" (pwN (wSrc 0 + tNb))
  , .declInit ui "yb" (.call "off4" [u wDst, u 0, v "r1", v "r2", v "r3"])
  , .declInit ui "ys" (pwN (wDst + tNb)) ]

/-- The tree steps for `tg` partials (a power of two): tg/2, ..., 2, 1. -/
def treeSteps (tg : Nat) : List Nat :=
  ((List.range (Nat.log2 tg)).map (fun k => 2 ^ k)).reverse

example : treeSteps 256 = [128, 64, 32, 16, 8, 4, 2, 1] := by native_decide
example : treeSteps 64 = [32, 16, 8, 4, 2, 1] := by native_decide
example : (treeSteps threadgroup).foldl (· + ·) 0 = threadgroup - 1 ∧
    (treeSteps threadgroupShort).foldl (· + ·) 0 = threadgroupShort - 1 := by native_decide

/-- The strided partial of thread `t` (a variable in scope), into `sh[t]`. -/
def partialSum (tg : Nat) (name : String) (c : Combine) (init : SlangExpr) (f : SlangExpr) : List SlangStmt :=
  let acc := "acc_" ++ name
  let i := "i_" ++ name
  [ .declInit fl acc init
  , .declInit ui i (v "t")
  , .whileLoop (lt (v i) (v "n"))
      [ .declInit fl "x" (ldX (v i))
      , .assign (v acc) (combine c (v acc) f)
      , .assign (v i) (add (v i) (u tg)) ]
  , .assign (sh (v "t")) (v acc) ]

/-- A reduction of `f(x)` (an expression over `x`) over the row into the
    float `name`, which every thread (or the one thread) then holds; `tg`
    threads (partials). -/
def reduce (tg : Nat) (var : Variant) (name : String) (c : Combine) (init : SlangExpr) (f : SlangExpr) :
    List SlangStmt :=
  let step (s : Nat) : SlangStmt := .assign (sh (v "t")) (combine c (sh (v "t")) (sh (add (v "t") (u s))))
  match var with
  | .par =>
      partialSum tg name c init f ++ [barrier] ++
      ((treeSteps tg).map (fun s => [ .ifNoElse (lt (v "t") (u s)) [step s], barrier ])).flatten ++
      [ .declInit fl name (sh (u 0)), barrier ]
  | .ser =>
      [ .forCount "t" (u 0) (u tg) (partialSum tg name c init f) ] ++
      (treeSteps tg).map (fun s => .forCount "t" (u 0) (u s) [step s]) ++
      [ .declInit fl name (sh (u 0)) ]

/-- `dst[j] = g(x_j)` over the row (`g` an expression over `x`). -/
def mapRow (tg : Nat) (var : Variant) (g : SlangExpr) : List SlangStmt :=
  let body := [ .declInit fl "x" (ldX (v "j")), stY (v "j") g ]
  match var with
  | .par =>
      [ .declInit ui "j" (v "t")
      , .whileLoop (lt (v "j") (v "n")) (body ++ [ .assign (v "j") (add (v "j") (u tg)) ]) ]
  | .ser => [ .forCount "j" (u 0) (v "n") body ]

/-- `dst[yb] = val`, by thread 0 (the parallel kernel) or the one thread. -/
def storeFirst (var : Variant) (val : SlangExpr) : List SlangStmt :=
  let st := .assign (.index (v "dst") (v "yb")) val
  match var with
  | .par => [ .ifNoElse (.bin "==" (v "t") (u 0)) [st] ]
  | .ser => [ st ]

/-- The entry: one group per row (both variants); the parallel kernel names
    its thread `t` and shares `sh`, the Serial sibling declares `sh` locally. -/
def entryRows (tg : Nat) (var : Variant) (body : List SlangStmt) : SlangFunctionDecl :=
  { attrs := [.shaderCompute, .numthreads (if var == .par then tg else 1) 1 1]
  , name := "main"
  , params := [ ⟨"gid", .vec .uint 3, .svGroupId, none, none, .qIn⟩
              , ⟨"lid", .vec .uint 3, .svGroupThreadId, none, none, .qIn⟩ ]
  , body :=
      [ .declInit ui "row" (add (mul (.member (v "gid") "y") (pwN wGroupsX)) (.member (v "gid") "x"))
      , .ifNoElse (.bin ">=" (v "row") (pwN wRows)) [ .ret none ] ] ++
      (match var with
       | .par => [ .declInit ui "t" (.member (v "lid") "x") ]
       | .ser => [ .declareArray fl "sh" tg ]) ++
      prologue ++ body }

/-- A row kernel: src0 and dst f32 (s1, s2 unused and kept), the helpers it
    calls, and `sh` in group-shared memory for the parallel variant. -/
def rowModule (tg : Nat) (var : Variant) (helpers : List SlangFunctionDecl) (body : List SlangStmt) :
    SlangShaderModule :=
  let m := kernelModule .float .uint .uint .float helpers (entryRows tg var body)
  match var with
  | .par => { m with groupShared := [ { name := "sh", elemType := fl, dims := [tg] } ] }
  | .ser => m

/-! ## Pinned pieces -/

example : String.intercalate "\n" ((reduce 256 .par "s" .sum (fLit 0.0) (v "x")).map (emitStmt 1)) =
"  float acc_s = 0.0f;
  uint i_s = t;
  while ((i_s < n)) {
    float x = s0[(xb + (i_s * xs))];
    acc_s = (acc_s + x);
    i_s = (i_s + 256u);
  }
  sh[t] = acc_s;
  GroupMemoryBarrierWithGroupSync();
  if ((t < 128u)) {
    sh[t] = (sh[t] + sh[(t + 128u)]);
  }
  GroupMemoryBarrierWithGroupSync();
  if ((t < 64u)) {
    sh[t] = (sh[t] + sh[(t + 64u)]);
  }
  GroupMemoryBarrierWithGroupSync();
  if ((t < 32u)) {
    sh[t] = (sh[t] + sh[(t + 32u)]);
  }
  GroupMemoryBarrierWithGroupSync();
  if ((t < 16u)) {
    sh[t] = (sh[t] + sh[(t + 16u)]);
  }
  GroupMemoryBarrierWithGroupSync();
  if ((t < 8u)) {
    sh[t] = (sh[t] + sh[(t + 8u)]);
  }
  GroupMemoryBarrierWithGroupSync();
  if ((t < 4u)) {
    sh[t] = (sh[t] + sh[(t + 4u)]);
  }
  GroupMemoryBarrierWithGroupSync();
  if ((t < 2u)) {
    sh[t] = (sh[t] + sh[(t + 2u)]);
  }
  GroupMemoryBarrierWithGroupSync();
  if ((t < 1u)) {
    sh[t] = (sh[t] + sh[(t + 1u)]);
  }
  GroupMemoryBarrierWithGroupSync();
  float s = sh[0u];
  GroupMemoryBarrierWithGroupSync();" := by native_decide

/-- The Serial reduction: the same partials (each a loop body over `t`), the
    same tree, in the same order. -/
example : String.intercalate "\n" ((reduce 256 .ser "s" .sum (fLit 0.0) (v "x")).map (emitStmt 1)) =
"  for (uint t = 0u; t < 256u; ++t) {
    float acc_s = 0.0f;
    uint i_s = t;
    while ((i_s < n)) {
      float x = s0[(xb + (i_s * xs))];
      acc_s = (acc_s + x);
      i_s = (i_s + 256u);
    }
    sh[t] = acc_s;
  }
  for (uint t = 0u; t < 128u; ++t) {
    sh[t] = (sh[t] + sh[(t + 128u)]);
  }
  for (uint t = 0u; t < 64u; ++t) {
    sh[t] = (sh[t] + sh[(t + 64u)]);
  }
  for (uint t = 0u; t < 32u; ++t) {
    sh[t] = (sh[t] + sh[(t + 32u)]);
  }
  for (uint t = 0u; t < 16u; ++t) {
    sh[t] = (sh[t] + sh[(t + 16u)]);
  }
  for (uint t = 0u; t < 8u; ++t) {
    sh[t] = (sh[t] + sh[(t + 8u)]);
  }
  for (uint t = 0u; t < 4u; ++t) {
    sh[t] = (sh[t] + sh[(t + 4u)]);
  }
  for (uint t = 0u; t < 2u; ++t) {
    sh[t] = (sh[t] + sh[(t + 2u)]);
  }
  for (uint t = 0u; t < 1u; ++t) {
    sh[t] = (sh[t] + sh[(t + 1u)]);
  }
  float s = sh[0u];" := by native_decide

example : String.intercalate "\n" (prologue.map (emitStmt 1)) =
"  uint n = pw(10u);
  uint r1 = (row % pw(11u));
  uint rr = (row / pw(11u));
  uint r2 = (rr % pw(12u));
  uint r3 = (rr / pw(12u));
  uint xb = off4(10u, 0u, r1, r2, r3);
  uint xs = pw(14u);
  uint yb = off4(1u, 0u, r1, r2, r3);
  uint ys = pw(5u);" := by native_decide

end Ggml.SlangCodegen.Rows
