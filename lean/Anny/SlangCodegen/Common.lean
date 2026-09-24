import Drape.SlangCodegen.Dsl

/-!
# `Anny.SlangCodegen.Common` — straight-line 3×3 algebra for the ANNY kernels

Not a kernel. Statement generators that the forward-kinematics kernels
inline: a 3×3 matrix is nine scalar locals `<pfx>0 … <pfx>8` (row-major,
`<pfx>(3i+j)` is row i, column j) and a 3-vector three locals
`<pfx>0 … <pfx>2`. Every loop here runs in Lean, so the emitted Slang is
unrolled straight-line code with no local arrays (a local array in a
single-thread GPU kernel spills; scalars stay in registers).

Rotations are 3×3 matrices, parameterised by their 6D truncation, the
first two rows re-orthonormalised (AGENTS.md rule 11):

    r1 = a1 / |a1|,  u = a2 − (r1·a2) r1,  r2 = u / |u|,  r3 = r1 × r2.

`rot6` emits that map and `rot6Backward` its vector-Jacobian product.
Both divide by `max(|·|, 1e-20)`: the GPU and the guest CPU round the
same Slang differently, and a norm at float noise must not become a NaN
on one of them only (AGENTS.md, Gate 5 G10).

Affine transforms `[R | t]` are 12 floats, row r at `4r … 4r+3` with the
translation in column 3 (the layout of `Sinew.SlangCodegen.Lbs`'s
`bone`).
-/

namespace Anny.SlangCodegen.Common

open LeanSlang
open Drape.SlangCodegen.Dsl

/-- The scalar `<pfx><k>`. -/
def s (pfx : String) (k : Nat) : E := v (pfx ++ toString k)
/-- Row i, column j of the 3×3 matrix `pfx`. -/
def m (pfx : String) (i j : Nat) : E := s pfx (3 * i + j)

def r3 : List Nat := [0, 1, 2]
def r9 : List Nat := List.range 9

/-- Declare the 3-vector `pfx` with components `f k`. -/
def vec (pfx : String) (f : Nat → E) : List St := r3.map fun k => let_ fT (pfx ++ toString k) (f k)
/-- Declare the 3×3 matrix `pfx` with entries `f i j`. -/
def mat (pfx : String) (f : Nat → Nat → E) : List St :=
  r9.map fun k => let_ fT (pfx ++ toString k) (f (k / 3) (k % 3))

/-- `c = a · b`. -/
def matMul (c a b : String) : List St :=
  mat c fun i j => m a i 0 * m b 0 j + m a i 1 * m b 1 j + m a i 2 * m b 2 j
/-- `c = aᵀ · b`. -/
def matTMul (c a b : String) : List St :=
  mat c fun i j => m a 0 i * m b 0 j + m a 1 i * m b 1 j + m a 2 i * m b 2 j
/-- `c = a · bᵀ`. -/
def matMulT (c a b : String) : List St :=
  mat c fun i j => m a i 0 * m b j 0 + m a i 1 * m b j 1 + m a i 2 * m b j 2
/-- `y = a · x`. -/
def matVec (y a x : String) : List St :=
  vec y fun i => m a i 0 * s x 0 + m a i 1 * s x 1 + m a i 2 * s x 2
/-- `y = aᵀ · x`. -/
def matTVec (y a x : String) : List St :=
  vec y fun i => m a 0 i * s x 0 + m a 1 i * s x 1 + m a 2 i * s x 2
/-- `a · b` of two 3-vectors. -/
def dot3 (a b : String) : E := s a 0 * s b 0 + s a 1 * s b 1 + s a 2 * s b 2
/-- `c = a × b`. -/
def cross (c a b : String) : List St :=
  vec c fun i =>
    let j := (i + 1) % 3
    let k := (i + 2) % 3
    s a j * s b k - s a k * s b j

/-- The guard every norm is divided through. -/
def tiny : E := .litFloatExact 1e-20

/-- The 3×3 rotation of the affine `buf[base … base+11]` into `pfx`. -/
def loadRot12 (pfx buf : String) (base : E) : List St :=
  mat pfx fun i j => at_ buf (base + u (4 * i + j))
/-- The translation of the affine `buf[base … base+11]` into `pfx`. -/
def loadT12 (pfx buf : String) (base : E) : List St :=
  vec pfx fun i => at_ buf (base + u (4 * i + 3))
/-- Write `[r | t]` to `buf[base … base+11]`. -/
def store12 (buf : String) (base : E) (r t : String) : List St :=
  (r9.map fun k => setAt buf (base + u (4 * (k / 3) + k % 3)) (s r k)) ++
  (r3.map fun i => setAt buf (base + u (4 * i + 3)) (s t i))

/-- The rotation of the 6D pair at `buf[base … base+5]` into the matrix
    `pfx` (rows r1, r2, r3). Leaves the forward intermediates
    `<pfx>_a1*`, `<pfx>_a2*`, `<pfx>_n1`, `<pfx>_d`, `<pfx>_nu` in scope
    for `rot6Backward`. -/
def rot6 (pfx buf : String) (base : E) : List St :=
  let a1 := pfx ++ "_a1"
  let a2 := pfx ++ "_a2"
  let r1 := pfx ++ "_r1"
  let uu := pfx ++ "_u"
  let r2 := pfx ++ "_r2"
  let r3v := pfx ++ "_r3"
  vec a1 (fun k => at_ buf (base + u k)) ++
  vec a2 (fun k => at_ buf (base + u (3 + k))) ++
  [ let_ fT (pfx ++ "_n1") (fmax (call "sqrt" [dot3 a1 a1]) tiny) ] ++
  vec r1 (fun k => s a1 k / v (pfx ++ "_n1")) ++
  [ let_ fT (pfx ++ "_d") (dot3 r1 a2) ] ++
  vec uu (fun k => s a2 k - v (pfx ++ "_d") * s r1 k) ++
  [ let_ fT (pfx ++ "_nu") (fmax (call "sqrt" [dot3 uu uu]) tiny) ] ++
  vec r2 (fun k => s uu k / v (pfx ++ "_nu")) ++
  cross r3v r1 r2 ++
  mat pfx (fun i j => s (if i == 0 then r1 else if i == 1 then r2 else r3v) j)

/-- The vector-Jacobian product of `rot6 pfx`: given the cotangent `dR`
    of the matrix `pfx`, write `d(a1, a2)` to `out[base … base+5]`.
    Must follow `rot6 pfx` in the same scope. -/
def rot6Backward (pfx dR out : String) (base : E) : List St :=
  let a2 := pfx ++ "_a2"
  let r1 := pfx ++ "_r1"
  let uu := pfx ++ "_u"
  let r2 := pfx ++ "_r2"
  let g1 := pfx ++ "_g1"
  let g2 := pfx ++ "_g2"
  let g3 := pfx ++ "_g3"
  let c1 := pfx ++ "_c1"
  let c2 := pfx ++ "_c2"
  let h1 := pfx ++ "_h1"
  let h2 := pfx ++ "_h2"
  let du := pfx ++ "_du"
  let da2 := pfx ++ "_da2"
  let h1b := pfx ++ "_h1b"
  let da1 := pfx ++ "_da1"
  -- the rows of dR
  vec g1 (fun k => m dR 0 k) ++ vec g2 (fun k => m dR 1 k) ++ vec g3 (fun k => m dR 2 k) ++
  -- r3 = r1 × r2: d r1 += r2 × g3, d r2 += g3 × r1
  cross c1 r2 g3 ++ cross c2 g3 r1 ++
  vec h1 (fun k => s g1 k + s c1 k) ++
  vec h2 (fun k => s g2 k + s c2 k) ++
  -- r2 = u / |u|
  [ let_ fT (pfx ++ "_p2") (dot3 r2 h2) ] ++
  vec du (fun k => (s h2 k - v (pfx ++ "_p2") * s r2 k) / v (pfx ++ "_nu")) ++
  -- u = a2 − (r1·a2) r1
  [ let_ fT (pfx ++ "_q") (dot3 r1 du) ] ++
  vec da2 (fun k => s du k - v (pfx ++ "_q") * s r1 k) ++
  vec h1b (fun k => s h1 k - (v (pfx ++ "_d") * s du k + v (pfx ++ "_q") * s a2 k)) ++
  -- r1 = a1 / |a1|
  [ let_ fT (pfx ++ "_p1") (dot3 r1 h1b) ] ++
  vec da1 (fun k => (s h1b k - v (pfx ++ "_p1") * s r1 k) / v (pfx ++ "_n1")) ++
  (r3.map fun k => setAt out (base + u k) (s da1 k)) ++
  (r3.map fun k => setAt out (base + u (3 + k)) (s da2 k))

end Anny.SlangCodegen.Common
