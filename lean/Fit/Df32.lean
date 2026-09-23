import LeanSlang

/-!
# `Fit.SlangCodegen.Df32` — float-float (df32) arithmetic for fit.elf's kernels

Gate 6g.0 (`gates/6g-polyfem-gpu/g0-df32`) showed that PolyFEM's forms,
its SDF sampler, its line-search energies and its CCD keep the fit
converging inside the Gate 6 guard when every operation rounds to a 48-bit
significand with float32 range: two float32s, `hi + lo`, with
`|lo| <= ulp(hi) / 2`. This module is that arithmetic as Slang, shared by
every `Fit.SlangCodegen.*` kernel of cut 6g-C: a `df` struct and the
error-free transformations (Dekker's `two_sum` and `quick_two_sum`,
the FMA `two_prod`) that `df_add`, `df_mul`, `df_div` and `df_sqrt`
are built from.

Every intermediate of an error-free transformation is declared `precise`
(SPIR-V `NoContraction`; slangc's cpp emit under `-ffp-contract=off`):
an optimiser that rewrote `(a + b) - a` as `b` would silently turn the
pair back into a float. `two_prod` is the one place an FMA is meant, and
it is written as `fma(a, b, -p)`. Everything is a function returning a
`df` value, so a kernel's tape (`Fit.SlangCodegen.Hess12`) can declare
each intermediate as `df tN = df_mul(ta, tb);`.

Accuracy: `df_add` is the "IEEE" variant (two `two_sum`s, two
renormalisations), so relative error is O(2^-96) barring cancellation;
`df_mul` and `df_mulf` are O(2^-94); `df_div` takes three quotient
corrections and `df_sqrt` one Newton correction, both O(2^-90). The GPU
and the cpp emit differ only where the driver forms an FMA the cpp
build does not, outside the `precise` intermediates (AGENTS: rd vs cpu
FMA rounding), which the C1 gate measures.
-/

namespace Fit.SlangCodegen.Df32

open LeanSlang

abbrev E := SlangExpr
abbrev St := SlangStmt

def fT : SlangType := .scalar .float
def uT : SlangType := .scalar .uint
def bT : SlangType := .scalar .bool
/-- The pair type. -/
def dfT : SlangType := .named "df"

def v (s : String) : E := .var s
def fl (x : Float) : E := .litFloatExact x
def u (n : Nat) : E := .litUint n
def add (a b : E) : E := .bin "+" a b
def sub (a b : E) : E := .bin "-" a b
def mul (a b : E) : E := .bin "*" a b
def div (a b : E) : E := .bin "/" a b
def neg (a : E) : E := .un "-" a
def hi (a : E) : E := .member a "hi"
def lo (a : E) : E := .member a "lo"
def call (f : String) (args : List E) : E := .call f args
def pf (n : String) (e : E) : St := .declarePrecise fT n (some e)
def lf (n : String) (e : E) : St := .declare fT n (some e)
def ld (n : String) (e : E) : St := .declare dfT n (some e)
def ret (e : E) : St := .ret (some e)

def pIn (n : String) (t : SlangType) : SlangBinding :=
  ⟨n, t, Semantic.none, none, none, .qIn⟩

/-- `struct df { float hi; float lo; };` -/
def structDecl : SlangStructDecl :=
  { name := "df", fields := [pIn "hi" fT, pIn "lo" fT] }

/-- `df df_make(float h, float l)`. -/
def fnMake : SlangFunctionDecl :=
  { retType := dfT, name := "df_make", params := [pIn "h" fT, pIn "l" fT]
  , body := [ .declare dfT "r" none
            , .assign (hi (v "r")) (v "h")
            , .assign (lo (v "r")) (v "l")
            , ret (v "r") ] }

/-- `df df_f(float a)`: a float widened to a pair. -/
def fnF : SlangFunctionDecl :=
  { retType := dfT, name := "df_f", params := [pIn "a" fT]
  , body := [ ret (call "df_make" [v "a", fl 0]) ] }

/-- Dekker/Knuth `two_sum`: `a + b` as an exact pair. -/
def fnTwoSum : SlangFunctionDecl :=
  { retType := dfT, name := "two_sum", params := [pIn "a" fT, pIn "b" fT]
  , body := [ pf "s" (add (v "a") (v "b"))
            , pf "bb" (sub (v "s") (v "a"))
            , pf "ea" (sub (v "a") (sub (v "s") (v "bb")))
            , pf "eb" (sub (v "b") (v "bb"))
            , pf "e" (add (v "ea") (v "eb"))
            , ret (call "df_make" [v "s", v "e"]) ] }

/-- `quick_two_sum`: `a + b` as an exact pair when `|a| >= |b|`. -/
def fnQuickTwoSum : SlangFunctionDecl :=
  { retType := dfT, name := "quick_two_sum", params := [pIn "a" fT, pIn "b" fT]
  , body := [ pf "s" (add (v "a") (v "b"))
            , pf "e" (sub (v "b") (sub (v "s") (v "a")))
            , ret (call "df_make" [v "s", v "e"]) ] }

/-- `two_prod`: `a * b` as an exact pair, the error term by FMA. -/
def fnTwoProd : SlangFunctionDecl :=
  { retType := dfT, name := "two_prod", params := [pIn "a" fT, pIn "b" fT]
  , body := [ pf "p" (mul (v "a") (v "b"))
            , pf "e" (call "fma" [v "a", v "b", neg (v "p")])
            , ret (call "df_make" [v "p", v "e"]) ] }

/-- `df df_add(df x, df y)`. -/
def fnAdd : SlangFunctionDecl :=
  { retType := dfT, name := "df_add", params := [pIn "x" dfT, pIn "y" dfT]
  , body := [ ld "s" (call "two_sum" [hi (v "x"), hi (v "y")])
            , ld "t" (call "two_sum" [lo (v "x"), lo (v "y")])
            , pf "sl" (add (lo (v "s")) (hi (v "t")))
            , ld "u" (call "quick_two_sum" [hi (v "s"), v "sl"])
            , pf "ul" (add (lo (v "u")) (lo (v "t")))
            , ret (call "quick_two_sum" [hi (v "u"), v "ul"]) ] }

/-- `df df_neg(df x)`. -/
def fnNeg : SlangFunctionDecl :=
  { retType := dfT, name := "df_neg", params := [pIn "x" dfT]
  , body := [ ret (call "df_make" [neg (hi (v "x")), neg (lo (v "x"))]) ] }

/-- `df df_sub(df x, df y)`. -/
def fnSub : SlangFunctionDecl :=
  { retType := dfT, name := "df_sub", params := [pIn "x" dfT, pIn "y" dfT]
  , body := [ ret (call "df_add" [v "x", call "df_neg" [v "y"]]) ] }

/-- `df df_mul(df x, df y)`. -/
def fnMul : SlangFunctionDecl :=
  { retType := dfT, name := "df_mul", params := [pIn "x" dfT, pIn "y" dfT]
  , body := [ ld "p" (call "two_prod" [hi (v "x"), hi (v "y")])
            , pf "c" (mul (hi (v "x")) (lo (v "y")))
            , pf "d" (mul (lo (v "x")) (hi (v "y")))
            , pf "cd" (add (v "c") (v "d"))
            , pf "pl" (add (lo (v "p")) (v "cd"))
            , ret (call "quick_two_sum" [hi (v "p"), v "pl"]) ] }

/-- `df df_mulf(df x, float c)`. -/
def fnMulF : SlangFunctionDecl :=
  { retType := dfT, name := "df_mulf", params := [pIn "x" dfT, pIn "c" fT]
  , body := [ ld "p" (call "two_prod" [hi (v "x"), v "c"])
            , pf "d" (mul (lo (v "x")) (v "c"))
            , pf "pl" (add (lo (v "p")) (v "d"))
            , ret (call "quick_two_sum" [hi (v "p"), v "pl"]) ] }

/-- `df df_div(df x, df y)`: three quotient corrections. -/
def fnDiv : SlangFunctionDecl :=
  { retType := dfT, name := "df_div", params := [pIn "x" dfT, pIn "y" dfT]
  , body := [ lf "q1" (div (hi (v "x")) (hi (v "y")))
            , ld "r1" (call "df_sub" [v "x", call "df_mulf" [v "y", v "q1"]])
            , lf "q2" (div (hi (v "r1")) (hi (v "y")))
            , ld "r2" (call "df_sub" [v "r1", call "df_mulf" [v "y", v "q2"]])
            , lf "q3" (div (hi (v "r2")) (hi (v "y")))
            , ld "q" (call "quick_two_sum" [v "q1", v "q2"])
            , ret (call "df_add" [v "q", call "df_f" [v "q3"]]) ] }

/-- `df df_sqrt(df x)`: `sqrt(hi)` plus one Newton correction; 0 for
    `x <= 0` (a norm at float noise can be exactly 0 on rd and not on cpu,
    AGENTS: guard every such root and division in the kernel). -/
def fnSqrt : SlangFunctionDecl :=
  { retType := dfT, name := "df_sqrt", params := [pIn "x" dfT]
  , body := [ .ifThen (.bin "<=" (hi (v "x")) (fl 0)) [ret (call "df_make" [fl 0, fl 0])] []
            , lf "r" (call "sqrt" [hi (v "x")])
            , ld "rr" (call "two_prod" [v "r", v "r"])
            , ld "d" (call "df_sub" [v "x", v "rr"])
            , pf "dd" (add (hi (v "d")) (lo (v "d")))
            , lf "corr" (div (v "dd") (mul (fl 2) (v "r")))
            , ret (call "quick_two_sum" [v "r", v "corr"]) ] }

/-- `float df_to_f(df x)`. -/
def fnToF : SlangFunctionDecl :=
  { retType := fT, name := "df_to_f", params := [pIn "x" dfT]
  , body := [ ret (add (hi (v "x")) (lo (v "x"))) ] }

/-- `bool df_lt(df x, df y)`: `x < y` on normalised pairs. -/
def fnLt : SlangFunctionDecl :=
  { retType := bT, name := "df_lt", params := [pIn "x" dfT, pIn "y" dfT]
  , body := [ ret (.bin "||" (.bin "<" (hi (v "x")) (hi (v "y")))
                             (.bin "&&" (.bin "==" (hi (v "x")) (hi (v "y")))
                                        (.bin "<" (lo (v "x")) (lo (v "y"))))) ] }

/-- `df df_abs(df x)`. -/
def fnAbs : SlangFunctionDecl :=
  { retType := dfT, name := "df_abs", params := [pIn "x" dfT]
  , body := [ .ifThen (call "df_lt" [v "x", call "df_make" [fl 0, fl 0]]) [ret (call "df_neg" [v "x"])] []
            , ret (v "x") ] }

/-- The helper functions, in dependency order. -/
def decls : List SlangFunctionDecl :=
  [ fnMake, fnF, fnTwoSum, fnQuickTwoSum, fnTwoProd, fnAdd, fnNeg, fnSub
  , fnMul, fnMulF, fnDiv, fnSqrt, fnToF, fnLt, fnAbs ]

/-! ### Lean model: the same arithmetic on `Float` pairs, for the
    `native_decide` pins of the kernels that use it. Lean's `Float` is
    binary64, so this models the *shape* (which terms are formed, in which
    order), not the binary32 rounding; the C1 gate holds the emitted
    kernels to the CPU double path at 1e-9. -/

structure Df where
  hi : Float
  lo : Float
deriving Repr

def Df.ofFloat (a : Float) : Df := ⟨a, 0⟩
def Df.toFloat (x : Df) : Float := x.hi + x.lo
def Df.zero : Df := ⟨0, 0⟩

def twoSum (a b : Float) : Df :=
  let s := a + b
  let bb := s - a
  ⟨s, (a - (s - bb)) + (b - bb)⟩

def quickTwoSum (a b : Float) : Df :=
  let s := a + b
  ⟨s, b - (s - a)⟩

/-- Exact on Lean's binary64 only up to its own rounding: no FMA in Lean's
    `Float`, so the model's `two_prod` error term is 0. -/
def twoProd (a b : Float) : Df := ⟨a * b, 0⟩

def Df.add (x y : Df) : Df :=
  let s := twoSum x.hi y.hi
  let t := twoSum x.lo y.lo
  let u := quickTwoSum s.hi (s.lo + t.hi)
  quickTwoSum u.hi (u.lo + t.lo)

def Df.neg (x : Df) : Df := ⟨-x.hi, -x.lo⟩
def Df.sub (x y : Df) : Df := x.add y.neg

def Df.mul (x y : Df) : Df :=
  let p := twoProd x.hi y.hi
  quickTwoSum p.hi (p.lo + (x.hi * y.lo + x.lo * y.hi))

end Fit.SlangCodegen.Df32
