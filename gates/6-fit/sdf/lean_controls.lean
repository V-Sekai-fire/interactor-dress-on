import Fit

/-!
Negative controls for the `native_decide` pins of
`lean/Fit/SdfSplineHessian.lean`: would the fixture pins notice the two
mistakes the kernel is most exposed to?

    cd lean && lake env lean ../gates/6-fit/sdf/lean_controls.lean

1. **Summation order.** The value accumulated as `v * (a * b * c)` instead of
   `((v * a) * b) * c`. Expected: `false` on both fixtures (the bits move).
2. **A float 2/3.** `spline` with 2/3 rounded to binary32 and widened, which
   is what an unsuffixed Slang literal gives. Expected: `false`, and all 64
   points of a 1/64 grid on [0, 1) differ.
-/

open Fit.SlangCodegen.SdfSplineHessian

def modelReordered (st : Array Float) (w : Float × Float × Float) : Float := Id.run do
  let (b0, _, _) := tables w.1
  let (b1, _, _) := tables w.2.1
  let (b2, _, _) := tables w.2.2
  let mut acc : Float := 0
  for i in [0:4] do
    for j in [0:4] do
      for k in [0:4] do
        acc := acc + st[i * 16 + j * 4 + k]! * (b0[i]! * b1[j]! * b2[k]!)
  return acc

def splineF32 (x : Float) : Float :=
  let a := x.abs
  if a >= 2 then 0 else if a >= 1 then let t := 2 - a; t * t * t / 6
  else (Float32.toFloat ((2 : Float32) / 3)) + (0.5 * a - 1) * x * x

def grid : List Float := (List.range 64).map fun n => Float.ofNat n / 64

#eval s!"reordered value == pinned value, fixture0: {(modelReordered fixtureStencil fixture0).toBits == (bits fixtureStencil fixture0).head!}"
#eval s!"reordered value == pinned value, fixture1: {(modelReordered fixtureStencil fixture1).toBits == (bits fixtureStencil fixture1).head!}"
#eval s!"float 2/3 spline == spline on all 64 grid points: {grid.all fun x => (splineF32 x).toBits == (spline x).toBits}"
#eval s!"grid points where they differ: {(grid.filter fun x => (splineF32 x).toBits != (spline x).toBits).length}/64"
