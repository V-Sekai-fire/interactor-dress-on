import LeanSlang
import Fit.Df32
import Fit.Hess12

/-!
# `Fit.SlangCodegen.SimilarityHessianBlock` — SimilarityForm's 12x12 Hessian per hinge, in df32

cloth-fit's `SimilarityForm` (`garment_forms/GarmentForm.cpp`) penalises,
per hinge `k` (an edge `ve0 ve1` shared by a face with apex `vf0` and a
face with apex `vf1`), the change of a fixed direction expressed in the
two faces' affine frames:

    e  = ve1 - ve0,  e0 = vf0 - ve0,  e1 = vf1 - ve0
    n0 = e x e0,     n1 = e1 x e
    v0 = c0 e + c1 e0 + c2 n0
    v1 = c3 e1 + c4 e + c5 n1
    err = |v0 - v1|^2

with `c0..c5` the hinge's `orig_coeffs` (computed once from the rest
shape). The CPU path evaluates `similarity_hessian` (sympy autogen,
`autogen/auto_derivatives2.cpp:3320`) on the 12 coordinates
`ve0, ve1, vf0, vf1`, projects it to PSD, scales it by
`0.5 (area_i + area_j)` and scatters it. This kernel is the first of
those steps for every hinge at once: `err` is written with the tape of
`Fit.SlangCodegen.Hess12`, which differentiates it twice at emit time,
so the Hessian below is derived from the energy in Lean, not copied
from sympy's C++; the C1 gate holds it to that C++ at 1e-9 relative.

One thread per hinge. Bindings:

  0  StructuredBuffer<df>   pos      3 per compact vertex slot (x, y, z pairs)
  1  StructuredBuffer<uint> hinge_v  4 slots per hinge: ve0, ve1, vf0, vf1
  2  StructuredBuffer<df>   coef     6 per hinge: c0..c5
  3  RWStructuredBuffer<df> blocks   144 per hinge, row-major H[i][j]
  4  ConstantBuffer<SimHessParams> params   { uint count; float sign; }

`sign` scales the block (1 in the solver). The gate's control sets -1: a
kernel with the wrong sign has to FAIL the comparison, and this is the
same kernel producing it.
-/

namespace Fit.SlangCodegen.SimilarityHessianBlock

open LeanSlang Fit.SlangCodegen.Df32 Fit.SlangCodegen.Hess12

private def ix (buf : String) (i : E) : E := .index (v buf) i

/-- The energy of one hinge on the tape. Variables 0..11 are the `df`
    locals `p0..p11`; the coefficients are the locals `c0..c5`. -/
def energy : M Node := do
  let p (i : Nat) : Node := Node.var s!"p{i}" i
  let c (i : Nat) : Node := Node.const (.e s!"c{i}")
  let vec (b : Nat) : V3 := #[p b, p (b + 1), p (b + 2)]
  let ve0 := vec 0
  let ve1 := vec 3
  let vf0 := vec 6
  let vf1 := vec 9
  let e  ← subV ve1 ve0
  let e0 ← subV vf0 ve0
  let e1 ← subV vf1 ve0
  let n0 ← crossV e e0
  let n1 ← crossV e1 e
  let v0 ← addV (← addV (← scaleV (c 0) e) (← scaleV (c 1) e0)) (← scaleV (c 2) n0)
  let v1 ← addV (← addV (← scaleV (c 3) e1) (← scaleV (c 4) e)) (← scaleV (c 5) n1)
  let d ← subV v0 v1
  dotV d d

/-- The tape, run once at emit time. -/
def tape : Node × Array St := Hess12.run energy

/-- Loads of the 12 coordinates and 6 coefficients. -/
private def loads : List St :=
  (List.range 4).flatMap (fun s =>
    (List.range 3).map (fun d =>
      let slot := ix "hinge_v" (add (v "hb") (u s))
      ld s!"p{3 * s + d}" (ix "pos" (add (mul (u 3) slot) (u d)))))
  ++ (List.range 6).map (fun i => ld s!"c{i}" (ix "coef" (add (v "cb") (u i))))

/-- The 144 stores, `blocks[ob + 12 i + j] = df_mulf(H_ij, sg)`. -/
private def stores (n : Node) : List St :=
  (List.range 12).flatMap (fun i =>
    (List.range 12).map (fun j =>
      let t := n.h[hSlot i j]!
      .assign (ix "blocks" (add (v "ob") (u (12 * i + j))))
              (call "df_mulf" [t.expr, v "sg"])))

def body : List St :=
  let (n, stmts) := tape
  [ .declare uT "k" (some (.member (v "tid") "x"))
  , .ifThen (.bin ">=" (v "k") (.member (v "params") "count")) [.ret none] []
  , .declare uT "hb" (some (mul (u 4) (v "k")))
  , .declare uT "cb" (some (mul (u 6) (v "k")))
  , .declare uT "ob" (some (mul (u 144) (v "k")))
  ] ++ loads ++ stmts.toList ++
  [ lf "sg" (.member (v "params") "sign") ] ++ stores n

private def glob (n : String) (ty : SlangType) (b : Nat) : SlangBinding :=
  ⟨n, ty, Semantic.none, some b, some 0, .qIn⟩

def shader : SlangShaderModule :=
  { structs :=
      [ Df32.structDecl
      , { name := "SimHessParams"
        , fields := [pIn "count" uT, pIn "sign" fT] } ]
  , globals :=
      [ glob "pos"     (.roBuf dfT) 0
      , glob "hinge_v" (.roBuf uT) 1
      , glob "coef"    (.roBuf dfT) 2
      , glob "blocks"  (.rwBuf dfT) 3
      , glob "params"  (.const "SimHessParams") 4 ]
  , functions := Df32.decls ++
      [ { attrs  := [.shaderCompute, .numthreads 64 1 1]
          name   := "main"
          params := [{ name := "tid", type := .vec .uint 3
                     , semantic := Semantic.svDispatchThreadId }]
          body   := body } ] }

/-- The tape's size is the kernel's: pinned so a change of the energy or
    of the tape's folding shows up here before it shows up in a gate. -/
def tapeLength : Nat := tape.2.size

example : shader.entryPointName = "main" := by native_decide
example : tapeLength = 1424 := by native_decide

end Fit.SlangCodegen.SimilarityHessianBlock
