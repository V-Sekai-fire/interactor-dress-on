import LeanSlang

/-!
# `Fit.SlangCodegen.SdfSplineHessian` — tricubic B-spline value, gradient, Hessian

The sampler cloth-fit's `FitForm` evaluates on its SDF: value, gradient
and symmetric Hessian of the uniform cubic B-spline over the 4³ stencil
around `floor(p/h)`. It replaces `openvdb::tools::SplineSampler::sampleHessian`
of the Huangzizhou/openvdb fork (c96cb069, `openvdb/tools/Interpolation.h`,
`SplineSampler::spline*` and `trilinearInterpolationHessian`), and it must
round exactly as that does (Gate 6a, check a.kernel: 0 ULP on all 13
outputs against OpenVDB's own stencils).

Call-site contract (`vendor/cloth-fit/.../garment_forms/SdfSpline.hpp`),
one thread per sample:

  0  StructuredBuffer<double>   stencil   64 per sample, data[i][j][k] at
                                          i*16 + j*4 + k, offsets -1..2
  1  StructuredBuffer<double>   uvw       3 per sample, index-space fraction
  2  RWStructuredBuffer<double> result    10 per sample: x, gx, gy, gz,
                                          hxx, hxy, hxz, hyy, hyz, hzz
  3  ConstantBuffer<SdfSplineParams> params   count = samples

(`out` is a Slang keyword, hence `result`.)

Arithmetic, operation for operation as the fork:

```
spline(x)    |x| >= 2: 0;  |x| >= 1: t = 2 - |x|, ((t*t)*t)/6
             else 2/3 + (((0.5*|x|) - 1)*x)*x
spline'(x)   |x| >= 2: 0;  |x| >= 1: t = 2 - |x|, ((-0.5*t)*t)*sign(x)
             else x*((1.5*|x|) - 2)
spline''(x)  |x| >= 2: 0;  |x| >= 1: 2 - |x|;  else (3*|x|) - 2
tables per axis d, u = uvw[d]:
  basis  = spline(u+1),   spline(u),   spline(1-u),    spline(2-u)
  deriv1 = spline'(u+1),  spline'(u),  -spline'(1-u),  -spline'(2-u)
  deriv2 = spline''(u+1), spline''(u), spline''(1-u),  spline''(2-u)
then for i, j, k ascending, each of the ten terms acc = acc + ((v*a)*b)*c.
```

Every constant is a `double`. An unsuffixed Slang literal is `float`, so
`2.0 / 3.0` would fold to 0.66666668653488159 and break the kernel (the
Gate 6a stand-in measured 8.7e18 ULP); here every literal goes through a
`double(...)` cast of an integer or of a dyadic float (0.5, 1.5, -0.5,
exact in float), and 2/3 is the double division `double(2) / double(3)`.

`model` below is the same arithmetic on Lean's `Float` (IEEE binary64,
correctly rounded), pinned with `native_decide` to the bits the native
reference sampler (`gates/6-fit/sdf/spline_ref.h`) prints for two fixtures.
-/

namespace Fit.SlangCodegen.SdfSplineHessian

open LeanSlang

private abbrev E := SlangExpr
private abbrev St := SlangStmt

private def dT : SlangType := .scalar .double
private def uT : SlangType := .scalar .uint

/-- `double(n)`: an integer-valued double constant. -/
private def dI (n : Int) : E := .cast dT (.litInt n)
/-- `double(x)` for a float literal that is exact in float (0.5, 1.5, -0.5). -/
private def dF (x : Float) : E := .cast dT (.litFloat x)
private def v (s : String) : E := .var s
private def u (n : Nat) : E := .litUint n
private def add (a b : E) : E := .bin "+" a b
private def sub (a b : E) : E := .bin "-" a b
private def mul (a b : E) : E := .bin "*" a b
private def div (a b : E) : E := .bin "/" a b
private def ix (buf : String) (i : E) : E := .index (.var buf) i
private def let_ (ty : SlangType) (n : String) (e : E) : St := .declare ty n (some e)
private def setAt (buf : String) (i r : E) : St := .assign (.index (.var buf) i) r
private def ret (e : E) : St := .ret (some e)
private def ge (a b : E) : E := .bin ">=" a b

/-- The three piecewise cubics share one shape: `|x| >= 2`, `|x| >= 1`, inner. -/
private def piecewise (name : String) (outer inner : List St) : SlangFunctionDecl :=
  { retType := dT, name := name
  , params := [{ name := "x", type := dT }]
  , body :=
      [ let_ dT "absx" (.call "abs" [v "x"])
      , .ifThen (ge (v "absx") (dI 2)) [ret (dI 0)] []
      , .ifThen (ge (v "absx") (dI 1)) outer []
      ] ++ inner }

private def tmp : St := let_ dT "tmp" (sub (dI 2) (v "absx"))

/-- `spline(x)`. -/
def fnSpline : SlangFunctionDecl :=
  piecewise "bspline"
    [ tmp, ret (div (mul (mul (v "tmp") (v "tmp")) (v "tmp")) (dI 6)) ]
    [ ret (add (div (dI 2) (dI 3))
               (mul (mul (sub (mul (dF 0.5) (v "absx")) (dI 1)) (v "x")) (v "x"))) ]

/-- `spline'(x)`. -/
def fnSplineD1 : SlangFunctionDecl :=
  piecewise "bspline_d1"
    [ tmp
    , ret (mul (mul (mul (dF (-0.5)) (v "tmp")) (v "tmp"))
               (.ternary (.bin ">" (v "x") (dI 0)) (dI 1) (dI (-1)))) ]
    [ ret (mul (v "x") (sub (mul (dF 1.5) (v "absx")) (dI 2))) ]

/-- `spline''(x)`. -/
def fnSplineD2 : SlangFunctionDecl :=
  piecewise "bspline_d2"
    [ ret (sub (dI 2) (v "absx")) ]
    [ ret (sub (mul (dI 3) (v "absx")) (dI 2)) ]

/-- Table slot `4d + n` of one of `basis`, `deriv1`, `deriv2`. -/
private def slot (n : Nat) : E := if n == 0 then v "t" else add (v "t") (u n)

/-- One table row: `basis[4d+n] = spline(a)`, `deriv1[..] = ±spline'(a)`,
    `deriv2[..] = spline''(a)`. -/
private def row (n : Nat) (a : E) (negD1 : Bool) : List St :=
  let d1 : E := .call "bspline_d1" [a]
  [ setAt "basis"  (slot n) (.call "bspline" [a])
  , setAt "deriv1" (slot n) (if negD1 then .un "-" d1 else d1)
  , setAt "deriv2" (slot n) (.call "bspline_d2" [a]) ]

/-- The ten accumulators, in output order, with their three factors
    (x-, y-, z-axis table) of `((v*a)*b)*c`. -/
private def terms : List (String × String × String × String) :=
  [ ("x",   "basis",  "basis",  "basis")
  , ("g0",  "deriv1", "basis",  "basis")
  , ("g1",  "basis",  "deriv1", "basis")
  , ("g2",  "basis",  "basis",  "deriv1")
  , ("h00", "deriv2", "basis",  "basis")
  , ("h01", "deriv1", "deriv1", "basis")
  , ("h02", "deriv1", "basis",  "deriv1")
  , ("h11", "basis",  "deriv2", "basis")
  , ("h12", "basis",  "deriv1", "deriv1")
  , ("h22", "basis",  "basis",  "deriv2") ]

private def body : List St :=
  [ let_ uT "lane" (.member (v "tid") "x")
  , .ifThen (ge (v "lane") (.member (v "params") "count")) [.ret none] []
  , let_ uT "sb" (mul (u 64) (v "lane"))
  , let_ uT "ub" (mul (u 3) (v "lane"))
  , let_ uT "rb" (mul (u 10) (v "lane"))
  , .declareArray dT "basis" 12
  , .declareArray dT "deriv1" 12
  , .declareArray dT "deriv2" 12
  , .forCount "d" (u 0) (u 3)
      ([ let_ dT "w" (ix "uvw" (add (v "ub") (v "d")))
       , let_ uT "t" (mul (u 4) (v "d")) ]
       ++ row 0 (add (v "w") (dI 1)) false
       ++ row 1 (v "w") false
       ++ row 2 (sub (dI 1) (v "w")) true
       ++ row 3 (sub (dI 2) (v "w")) true)
  ] ++ terms.map (fun (acc, _, _, _) => let_ dT acc (dI 0)) ++
  [ .forCount "i" (u 0) (u 4)
    [ .forCount "j" (u 0) (u 4)
      [ .forCount "k" (u 0) (u 4)
        ([ let_ dT "s"
             (ix "stencil" (add (add (add (v "sb") (mul (v "i") (u 16)))
                                     (mul (v "j") (u 4))) (v "k"))) ]
         ++ terms.map (fun (acc, a, b, c) =>
              .assign (v acc)
                (add (v acc)
                  (mul (mul (mul (v "s") (ix a (v "i")))
                            (ix b (add (u 4) (v "j"))))
                       (ix c (add (u 8) (v "k")))))) ) ] ] ] ++
  (terms.zipIdx.map fun ((acc, _, _, _), n) =>
    setAt "result" (if n == 0 then v "rb" else add (v "rb") (u n)) (v acc))

private def glob (n : String) (ty : SlangType) (b : Nat) : SlangBinding :=
  ⟨n, ty, Semantic.none, some b, some 0, .qIn⟩

def shader : SlangShaderModule :=
  { structs :=
      [ { name := "SdfSplineParams"
        , fields := [ ⟨"count", uT, Semantic.none, none, none, .qIn⟩ ] } ]
  , globals :=
      [ glob "stencil" (.roBuf dT) 0
      , glob "uvw"     (.roBuf dT) 1
      , glob "result"  (.rwBuf dT) 2
      , glob "params"  (.const "SdfSplineParams") 3 ]
  , functions :=
      [ fnSpline, fnSplineD1, fnSplineD2
      , { attrs  := [.shaderCompute, .numthreads 64 1 1]
          name   := "main"
          params := [{ name := "tid", type := .vec .uint 3
                     , semantic := Semantic.svDispatchThreadId }]
          body   := body } ] }

def expected : String :=
"struct SdfSplineParams {
  uint count;
};

[[vk::binding(0, 0)]]
StructuredBuffer<double> stencil;
[[vk::binding(1, 0)]]
StructuredBuffer<double> uvw;
[[vk::binding(2, 0)]]
RWStructuredBuffer<double> result;
[[vk::binding(3, 0)]]
ConstantBuffer<SdfSplineParams> params;

double bspline(double x) {
  double absx = abs(x);
  if ((absx >= double(2))) {
    return double(0);
  }
  if ((absx >= double(1))) {
    double tmp = (double(2) - absx);
    return (((tmp * tmp) * tmp) / double(6));
  }
  return ((double(2) / double(3)) + ((((double(0.500000) * absx) - double(1)) * x) * x));
}

double bspline_d1(double x) {
  double absx = abs(x);
  if ((absx >= double(2))) {
    return double(0);
  }
  if ((absx >= double(1))) {
    double tmp = (double(2) - absx);
    return (((double(-0.500000) * tmp) * tmp) * ((x > double(0)) ? double(1) : double((-1))));
  }
  return (x * ((double(1.500000) * absx) - double(2)));
}

double bspline_d2(double x) {
  double absx = abs(x);
  if ((absx >= double(2))) {
    return double(0);
  }
  if ((absx >= double(1))) {
    return (double(2) - absx);
  }
  return ((double(3) * absx) - double(2));
}

[shader(\"compute\")] [numthreads(64, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
  uint lane = tid.x;
  if ((lane >= params.count)) {
    return;
  }
  uint sb = (64u * lane);
  uint ub = (3u * lane);
  uint rb = (10u * lane);
  double basis[12];
  double deriv1[12];
  double deriv2[12];
  for (uint d = 0u; d < 3u; ++d) {
    double w = uvw[(ub + d)];
    uint t = (4u * d);
    basis[t] = bspline((w + double(1)));
    deriv1[t] = bspline_d1((w + double(1)));
    deriv2[t] = bspline_d2((w + double(1)));
    basis[(t + 1u)] = bspline(w);
    deriv1[(t + 1u)] = bspline_d1(w);
    deriv2[(t + 1u)] = bspline_d2(w);
    basis[(t + 2u)] = bspline((double(1) - w));
    deriv1[(t + 2u)] = (-bspline_d1((double(1) - w)));
    deriv2[(t + 2u)] = bspline_d2((double(1) - w));
    basis[(t + 3u)] = bspline((double(2) - w));
    deriv1[(t + 3u)] = (-bspline_d1((double(2) - w)));
    deriv2[(t + 3u)] = bspline_d2((double(2) - w));
  }
  double x = double(0);
  double g0 = double(0);
  double g1 = double(0);
  double g2 = double(0);
  double h00 = double(0);
  double h01 = double(0);
  double h02 = double(0);
  double h11 = double(0);
  double h12 = double(0);
  double h22 = double(0);
  for (uint i = 0u; i < 4u; ++i) {
    for (uint j = 0u; j < 4u; ++j) {
      for (uint k = 0u; k < 4u; ++k) {
        double s = stencil[(((sb + (i * 16u)) + (j * 4u)) + k)];
        x = (x + (((s * basis[i]) * basis[(4u + j)]) * basis[(8u + k)]));
        g0 = (g0 + (((s * deriv1[i]) * basis[(4u + j)]) * basis[(8u + k)]));
        g1 = (g1 + (((s * basis[i]) * deriv1[(4u + j)]) * basis[(8u + k)]));
        g2 = (g2 + (((s * basis[i]) * basis[(4u + j)]) * deriv1[(8u + k)]));
        h00 = (h00 + (((s * deriv2[i]) * basis[(4u + j)]) * basis[(8u + k)]));
        h01 = (h01 + (((s * deriv1[i]) * deriv1[(4u + j)]) * basis[(8u + k)]));
        h02 = (h02 + (((s * deriv1[i]) * basis[(4u + j)]) * deriv1[(8u + k)]));
        h11 = (h11 + (((s * basis[i]) * deriv2[(4u + j)]) * basis[(8u + k)]));
        h12 = (h12 + (((s * basis[i]) * deriv1[(4u + j)]) * deriv1[(8u + k)]));
        h22 = (h22 + (((s * basis[i]) * basis[(4u + j)]) * deriv2[(8u + k)]));
      }
    }
  }
  result[rb] = x;
  result[(rb + 1u)] = g0;
  result[(rb + 2u)] = g1;
  result[(rb + 3u)] = g2;
  result[(rb + 4u)] = h00;
  result[(rb + 5u)] = h01;
  result[(rb + 6u)] = h02;
  result[(rb + 7u)] = h11;
  result[(rb + 8u)] = h12;
  result[(rb + 9u)] = h22;
}"

example : LeanSlang.emit shader = expected := by native_decide
example : shader.entryPointName = "main" := by native_decide

/-! ## The arithmetic on `Float`, pinned to the native reference

`model` is the kernel's arithmetic on binary64: the same operations, in
the same order, as the emitted Slang and as `spline_ref.h`. The pins
below hold it to the bits `spline_ref.h` prints for two fixtures
(`gates/6-fit/sdf/lean_fixtures.cpp`); fixture 1 has u = 0 on the x axis, so the
|x| = 1 and |x| = 2 branch edges are taken. -/

def spline (x : Float) : Float :=
  let a := x.abs
  if a >= 2 then 0
  else if a >= 1 then let t := 2 - a; t * t * t / 6
  else 2 / 3 + (0.5 * a - 1) * x * x

def splineD1 (x : Float) : Float :=
  let a := x.abs
  if a >= 2 then 0
  else if a >= 1 then let t := 2 - a; -0.5 * t * t * (if x > 0 then 1 else -1)
  else x * (1.5 * a - 2)

def splineD2 (x : Float) : Float :=
  let a := x.abs
  if a >= 2 then 0
  else if a >= 1 then 2 - a
  else 3 * a - 2

/-- Per-axis tables `(basis, deriv1, deriv2)` for one fraction `u`. -/
def tables (w : Float) : Array Float × Array Float × Array Float :=
  ( #[spline (w + 1), spline w, spline (1 - w), spline (2 - w)]
  , #[splineD1 (w + 1), splineD1 w, -splineD1 (1 - w), -splineD1 (2 - w)]
  , #[splineD2 (w + 1), splineD2 w, splineD2 (1 - w), splineD2 (2 - w)] )

/-- The ten outputs for one 64-value stencil and `uvw`. -/
def model (st : Array Float) (w : Float × Float × Float) : Array Float := Id.run do
  let (b0, d0, e0) := tables w.1
  let (b1, d1, e1) := tables w.2.1
  let (b2, d2, e2) := tables w.2.2
  let mut acc : Array Float := Array.replicate 10 0
  for i in [0:4] do
    for j in [0:4] do
      for k in [0:4] do
        let s := st[i * 16 + j * 4 + k]!
        acc := acc.modify 0 (· + s * b0[i]! * b1[j]! * b2[k]!)
        acc := acc.modify 1 (· + s * d0[i]! * b1[j]! * b2[k]!)
        acc := acc.modify 2 (· + s * b0[i]! * d1[j]! * b2[k]!)
        acc := acc.modify 3 (· + s * b0[i]! * b1[j]! * d2[k]!)
        acc := acc.modify 4 (· + s * e0[i]! * b1[j]! * b2[k]!)
        acc := acc.modify 5 (· + s * d0[i]! * d1[j]! * b2[k]!)
        acc := acc.modify 6 (· + s * d0[i]! * b1[j]! * d2[k]!)
        acc := acc.modify 7 (· + s * b0[i]! * e1[j]! * b2[k]!)
        acc := acc.modify 8 (· + s * b0[i]! * d1[j]! * d2[k]!)
        acc := acc.modify 9 (· + s * b0[i]! * b1[j]! * e2[k]!)
  return acc

/-- Fixture stencil: value n is `((37 n) mod 64) / 7 - 4.5`. -/
def fixtureStencil : Array Float :=
  (Array.range 64).map fun n => Float.ofNat ((37 * n) % 64) / 7 - 4.5

def fixture0 : Float × Float × Float := (1 / 3, 2 / 7, 0.96875)
def fixture1 : Float × Float × Float := (0, 0.5, 5 / 9)

def bits (st : Array Float) (w : Float × Float × Float) : List UInt64 :=
  (model st w).toList.map Float.toBits

example : (2 / 3 : Float).toBits = 0x3FE5555555555555 := by native_decide

example : bits fixtureStencil fixture0 = [0x3FD01273BF9B23E9, 0x3FF3C4E2E1B062D7, 0xBFFC9AD24B4E1D8C, 0x3FE4599A8C959322, 0x3FE3653EEF4CBF8E, 0xBFC40C2543331C9F, 0x3FE0E68369F47200, 0xBFF95FC5C73275D6, 0xBFC7BBE8146B37A1, 0xC000231768B8956F] := by native_decide
example : bits fixtureStencil fixture1 = [0xBFE27204F23077E4, 0x3FB10482692907DF, 0xBFF2999675EF9F9A, 0x3FCFA7B4F3687F38, 0x400F9A6B13E7ACC0, 0xBFCFC16462203504, 0x3FDC419E2CD95252, 0x3FE54F44CCF82DF7, 0xC00108E125C68255, 0x3FEDE96CCFB08941] := by native_decide

end Fit.SlangCodegen.SdfSplineHessian
