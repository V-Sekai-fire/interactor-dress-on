import LeanSlang
import Fit.Df32

/-!
# `Fit.SlangCodegen.Hess12` — a second-order forward-mode tape over 12 variables

cloth-fit's per-element Hessians (`vendor/cloth-fit/src/polyfem/autogen/`)
are sympy output: the energy differentiated symbolically, common
subexpressions named, and the result printed as C++. Rule 2 (AGENTS.md)
wants the GPU kernel emitted from Lean, not transcribed from that C++,
so this module differentiates at emit time instead. A `Node` carries
the value of an expression, its 12 first derivatives and its 78 upper-
triangular second derivatives, each as a `Term`: structurally zero,
`±1`, or the name of a `df` variable the tape has declared. Every
non-trivial intermediate becomes one `df tN = df_op(...)` statement, so
the emitted kernel is straight-line df32 code whose size is the number
of distinct intermediate values, not the size of an expanded expression
tree; the zeros and units are folded away and never emitted.

`mul` is the product rule at second order,

    (ab)_i  = a_i b + a b_i
    (ab)_ij = a_ij b + a_i b_j + a_j b_i + a b_ij,

and `add`, `neg` and `sub` are termwise. A kernel writes its energy with
these and reads the Hessian off the final node (`Fit.SlangCodegen.
SimilarityHessianBlock`). The tape is a `StateM` over the statement
list; `run` returns the statements to splice into the kernel body.
-/

namespace Fit.SlangCodegen.Hess12

open LeanSlang Fit.SlangCodegen.Df32

/-- A derivative slot: structurally zero, a unit, or a named `df` local. -/
inductive Term
  | zero
  | one
  | negOne
  | e (name : String)
deriving Repr, BEq, Inhabited

def Term.expr : Term → E
  | .zero   => call "df_make" [fl 0, fl 0]
  | .one    => call "df_make" [fl 1, fl 0]
  | .negOne => call "df_make" [fl (-1), fl 0]
  | .e n    => v n

structure Tape where
  stmts : Array St := #[]
  next  : Nat := 0

abbrev M := StateM Tape

/-- Declare `df tN = rhs;` and name it. -/
def fresh (rhs : E) : M Term := do
  let t ← get
  let n := s!"t{t.next}"
  set ({ stmts := t.stmts.push (ld n rhs), next := t.next + 1 } : Tape)
  return .e n

def negT : Term → M Term
  | .zero   => pure .zero
  | .one    => pure .negOne
  | .negOne => pure .one
  | .e n    => fresh (call "df_neg" [v n])

def addT (a b : Term) : M Term :=
  match a, b with
  | .zero, x => pure x
  | x, .zero => pure x
  | x, y     => fresh (call "df_add" [x.expr, y.expr])

def mulT (a b : Term) : M Term :=
  match a, b with
  | .zero, _   => pure .zero
  | _, .zero   => pure .zero
  | .one, x    => pure x
  | x, .one    => pure x
  | .negOne, x => negT x
  | x, .negOne => negT x
  | x, y       => fresh (call "df_mul" [x.expr, y.expr])

/-- Variables per element and upper-triangular Hessian slots. -/
def nVar : Nat := 12
def nHess : Nat := 78

/-- Slot of `(i, j)` with `i <= j` in the packed upper triangle:
    rows before `i` hold `12 + 11 + ... + (12 - i + 1)` entries. -/
def hIdx (i j : Nat) : Nat := i * nVar - i * (i - 1) / 2 + (j - i)

def hSlot (i j : Nat) : Nat := if i ≤ j then hIdx i j else hIdx j i

structure Node where
  val : Term
  g   : Array Term
  h   : Array Term
deriving Inhabited

def zeros (n : Nat) : Array Term := Array.replicate n .zero

/-- A per-element constant (a coefficient read from a buffer): no derivatives. -/
def Node.const (t : Term) : Node := ⟨t, zeros nVar, zeros nHess⟩

/-- Variable `i` of the 12, held in the `df` local `name`. -/
def Node.var (name : String) (i : Nat) : Node :=
  ⟨.e name, (zeros nVar).set! i .one, zeros nHess⟩

def addN (a b : Node) : M Node := do
  let val ← addT a.val b.val
  let mut g : Array Term := #[]
  for i in [0:nVar] do
    g := g.push (← addT a.g[i]! b.g[i]!)
  let mut h : Array Term := #[]
  for k in [0:nHess] do
    h := h.push (← addT a.h[k]! b.h[k]!)
  return ⟨val, g, h⟩

def negN (a : Node) : M Node := do
  let val ← negT a.val
  let mut g : Array Term := #[]
  for i in [0:nVar] do
    g := g.push (← negT a.g[i]!)
  let mut h : Array Term := #[]
  for k in [0:nHess] do
    h := h.push (← negT a.h[k]!)
  return ⟨val, g, h⟩

def subN (a b : Node) : M Node := do
  addN a (← negN b)

def mulN (a b : Node) : M Node := do
  let val ← mulT a.val b.val
  let mut g : Array Term := #[]
  for i in [0:nVar] do
    let x ← mulT a.g[i]! b.val
    let y ← mulT a.val b.g[i]!
    g := g.push (← addT x y)
  let mut h : Array Term := #[]
  for i in [0:nVar] do
    for j in [i:nVar] do
      let k := hIdx i j
      let t1 ← mulT a.h[k]! b.val
      let t2 ← mulT a.g[i]! b.g[j]!
      let t3 ← mulT a.g[j]! b.g[i]!
      let t4 ← mulT a.val b.h[k]!
      let s1 ← addT t1 t2
      let s2 ← addT s1 t3
      h := h.push (← addT s2 t4)
  return ⟨val, g, h⟩

/-- 3-vectors of nodes. -/
abbrev V3 := Array Node

def subV (a b : V3) : M V3 := do
  return #[← subN a[0]! b[0]!, ← subN a[1]! b[1]!, ← subN a[2]! b[2]!]

def addV (a b : V3) : M V3 := do
  return #[← addN a[0]! b[0]!, ← addN a[1]! b[1]!, ← addN a[2]! b[2]!]

def scaleV (c : Node) (a : V3) : M V3 := do
  return #[← mulN c a[0]!, ← mulN c a[1]!, ← mulN c a[2]!]

def crossV (a b : V3) : M V3 := do
  let x ← subN (← mulN a[1]! b[2]!) (← mulN a[2]! b[1]!)
  let y ← subN (← mulN a[2]! b[0]!) (← mulN a[0]! b[2]!)
  let z ← subN (← mulN a[0]! b[1]!) (← mulN a[1]! b[0]!)
  return #[x, y, z]

def dotV (a b : V3) : M Node := do
  let s0 ← mulN a[0]! b[0]!
  let s1 ← mulN a[1]! b[1]!
  let s2 ← mulN a[2]! b[2]!
  addN (← addN s0 s1) s2

/-- Run a tape: the final node and the statements that define its locals. -/
def run (m : M Node) : Node × Array St :=
  let (n, t) := Id.run (m.run {})
  (n, t.stmts)

end Fit.SlangCodegen.Hess12
