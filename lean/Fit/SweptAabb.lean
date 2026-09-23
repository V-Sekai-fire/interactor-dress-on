import LeanSlang

/-!
# `Fit.SlangCodegen.SweptAabb` — the broad phase's boxes, one per primitive

ipc-toolkit's `BroadPhase::build(V0, V1, E, F, r)` (`broad_phase/aabb.cpp`)
gives every vertex the box of its positions at both ends of the line
search, inflated by `r` (= dhat / 2), and every edge and face the union
of its vertices' boxes. This kernel is the same, one thread per
primitive, straight from the positions (an edge's box is the union over
its two vertices' four positions), so it needs no barrier between the
vertex and the edge boxes.

The CPU inflates with directed rounding (`fesetround`); the guest ignores
`fesetround` (AGENTS), and float has no room for it anyway, so the box is
widened by an explicit margin `m (1 + max |coordinate|)` on every side
with `m = 2^-20`: about 2^4 float ulps at the coordinates' scale, far
above the rounding of any float operation here, and 0.1 % of `r` at the
solve's scale (r = 1e-3). Every box holds the CPU's double box, so every
pair the CPU finds is found here (the C2 gate counts the misses, which
must be 0, and the extras, which the margin keeps under 5 %).

Bindings:

  0  StructuredBuffer<float>   x0      3 per vertex, the line search's start
  1  StructuredBuffer<float>   x1      3 per vertex, its end
  2  StructuredBuffer<uint>    edges   2 per edge
  3  StructuredBuffer<uint>    faces   3 per face
  4  RWStructuredBuffer<float> boxes   6 per primitive: min xyz, max xyz;
                                       primitive i is vertex i, edge i - n_v,
                                       or face i - n_v - n_e
  5  ConstantBuffer<AabbParams> params { uint n_v, n_e, n_f; float inflation, margin; }
-/

namespace Fit.SlangCodegen.SweptAabb

open LeanSlang

private abbrev E := SlangExpr
private abbrev St := SlangStmt

private def fT : SlangType := .scalar .float
private def uT : SlangType := .scalar .uint
private def v (s : String) : E := .var s
private def u (n : Nat) : E := .litUint n
private def fl (x : Float) : E := .litFloatExact x
private def add (a b : E) : E := .bin "+" a b
private def sub (a b : E) : E := .bin "-" a b
private def mul (a b : E) : E := .bin "*" a b
private def ix (buf : String) (i : E) : E := .index (v buf) i
private def prm (f : String) : E := .member (v "params") f
private def pIn (n : String) (t : SlangType) : SlangBinding := ⟨n, t, Semantic.none, none, none, .qIn⟩

/-- The three-vertex slot table: `vv[k]` for `k < nv`. -/
private def classify : List St :=
  [ .declareArray uT "vv" 3
  , .declare uT "nv" (some (u 1))
  , .assign (ix "vv" (u 0)) (v "i")
  , .assign (ix "vv" (u 1)) (u 0)
  , .assign (ix "vv" (u 2)) (u 0)
  , .ifThen (.bin ">=" (v "i") (prm "n_v"))
      [ .ifThen (.bin "<" (v "i") (add (prm "n_v") (prm "n_e")))
          [ .declare uT "e" (some (sub (v "i") (prm "n_v")))
          , .assign (v "nv") (u 2)
          , .assign (ix "vv" (u 0)) (ix "edges" (mul (u 2) (v "e")))
          , .assign (ix "vv" (u 1)) (ix "edges" (add (mul (u 2) (v "e")) (u 1))) ]
          [ .declare uT "f" (some (sub (sub (v "i") (prm "n_v")) (prm "n_e")))
          , .assign (v "nv") (u 3)
          , .assign (ix "vv" (u 0)) (ix "faces" (mul (u 3) (v "f")))
          , .assign (ix "vv" (u 1)) (ix "faces" (add (mul (u 3) (v "f")) (u 1)))
          , .assign (ix "vv" (u 2)) (ix "faces" (add (mul (u 3) (v "f")) (u 2))) ] ]
      [] ]

def body : List St :=
  [ .declare uT "i" (some (.member (v "tid") "x"))
  , .ifThen (.bin ">=" (v "i") (add (add (prm "n_v") (prm "n_e")) (prm "n_f"))) [.ret none] []
  ] ++ classify ++
  [ .declareArray fT "mn" 3
  , .declareArray fT "mx" 3
  , .forCount "d" (u 0) (u 3)
      [ .assign (ix "mn" (v "d")) (fl 3.0e38)
      , .assign (ix "mx" (v "d")) (fl (-3.0e38)) ]
  , .forCount "k" (u 0) (v "nv")
      [ .declare uT "b" (some (mul (u 3) (ix "vv" (v "k"))))
      , .forCount "d" (u 0) (u 3)
          [ .declare fT "a0" (some (ix "x0" (add (v "b") (v "d"))))
          , .declare fT "a1" (some (ix "x1" (add (v "b") (v "d"))))
          , .assign (ix "mn" (v "d")) (.call "min" [ix "mn" (v "d"), .call "min" [v "a0", v "a1"]])
          , .assign (ix "mx" (v "d")) (.call "max" [ix "mx" (v "d"), .call "max" [v "a0", v "a1"]]) ] ]
  , .declare uT "ob" (some (mul (u 6) (v "i")))
  , .forCount "d" (u 0) (u 3)
      [ .declare fT "m" (some (mul (prm "margin")
            (add (fl 1) (.call "max" [.call "abs" [ix "mn" (v "d")], .call "abs" [ix "mx" (v "d")]]))))
      , .assign (ix "boxes" (add (v "ob") (v "d"))) (sub (sub (ix "mn" (v "d")) (prm "inflation")) (v "m"))
      , .assign (ix "boxes" (add (add (v "ob") (u 3)) (v "d"))) (add (add (ix "mx" (v "d")) (prm "inflation")) (v "m")) ] ]

private def glob (n : String) (ty : SlangType) (b : Nat) : SlangBinding :=
  ⟨n, ty, Semantic.none, some b, some 0, .qIn⟩

def shader : SlangShaderModule :=
  { structs :=
      [ { name := "AabbParams"
        , fields := [pIn "n_v" uT, pIn "n_e" uT, pIn "n_f" uT, pIn "inflation" fT, pIn "margin" fT] } ]
  , globals :=
      [ glob "x0"     (.roBuf fT) 0
      , glob "x1"     (.roBuf fT) 1
      , glob "edges"  (.roBuf uT) 2
      , glob "faces"  (.roBuf uT) 3
      , glob "boxes"  (.rwBuf fT) 4
      , glob "params" (.const "AabbParams") 5 ]
  , functions :=
      [ { attrs  := [.shaderCompute, .numthreads 64 1 1]
          name   := "main"
          params := [{ name := "tid", type := .vec .uint 3
                     , semantic := Semantic.svDispatchThreadId }]
          body   := body } ] }

example : shader.entryPointName = "main" := by native_decide

end Fit.SlangCodegen.SweptAabb
