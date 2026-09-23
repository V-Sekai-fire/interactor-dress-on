import LeanSlang

/-!
# `Fit.SlangCodegen.BoxPair` — the broad phase's pairs: count, then write

ipc-toolkit's brute force (`broad_phase/brute_force.cpp`) tests every
edge box against every later edge box and every face box against every
vertex box, keeping a pair when `can_edges_collide` /
`can_face_vertex_collide` allow it (no shared vertex, and some vertex
pair passes the mesh's `can_collide`) and the boxes overlap. cloth-fit's
`can_collide` (guest/fit/fit_driver.cpp) admits a pair only when a
garment vertex is involved (with self-collision on: at least one of the
two is garment; off: one garment and one avatar), so the avatar's own
pairs, 66 % of foxgirl's boxes squared, are never wanted. This kernel
walks only the queries that can produce a wanted pair, one thread each,
against every target of its kind:

  kind 0  an edge with a garment vertex, against every edge
  kind 1  a garment vertex, against every face without a garment vertex
  kind 2  a face with a garment vertex, against every vertex

Each wanted pair is produced exactly once: an edge pair by the lower-
numbered edge when both are queries, else by the query; a face-vertex
pair by the face when the face is a query, else by the vertex. The
targets are walked in index order, so a query's pairs come out ordered
and the list is the same on every run (no atomics).

Two dispatches of the same kernel make one build: `write = 0` counts a
query's pairs into `counts[q]`; the host scans the counts into offsets,
uploads them into the same buffer, and `write = 1` writes each query's
pairs at its offset. Two round trips per build.

Bindings:

  0  StructuredBuffer<uint>    queries  kind << 30 | id
  1  StructuredBuffer<float>   boxes    6 per primitive (Fit.SlangCodegen.SweptAabb)
  2  StructuredBuffer<uint>    edges    2 per edge
  3  StructuredBuffer<uint>    faces    3 per face
  4  StructuredBuffer<uint>    garment  1 per vertex: 1 when a garment vertex
  5  RWStructuredBuffer<uint>  counts   per query: the count (write = 0), the offset (write = 1)
  6  RWStructuredBuffer<uint>  pairs    2 per pair: (edge, edge) with edge0 < edge1, or (face, vertex)
  7  ConstantBuffer<PairParams> params  { uint n_queries, n_v, n_e, n_f, self_collision, write; }
-/

namespace Fit.SlangCodegen.BoxPair

open LeanSlang

private abbrev E := SlangExpr
private abbrev St := SlangStmt

private def fT : SlangType := .scalar .float
private def uT : SlangType := .scalar .uint
private def bT : SlangType := .scalar .bool
private def v (s : String) : E := .var s
private def u (n : Nat) : E := .litUint n
private def add (a b : E) : E := .bin "+" a b
private def sub (a b : E) : E := .bin "-" a b
private def mul (a b : E) : E := .bin "*" a b
private def ix (buf : String) (i : E) : E := .index (v buf) i
private def prm (f : String) : E := .member (v "params") f
private def pIn (n : String) (t : SlangType) : SlangBinding := ⟨n, t, Semantic.none, none, none, .qIn⟩
private def band (a b : E) : E := .bin "&&" a b
private def bor (a b : E) : E := .bin "||" a b
private def bnot (a : E) : E := .un "!" a
private def ne (a b : E) : E := .bin "!=" a b
private def eq (a b : E) : E := .bin "==" a b

/-- `bool can_collide(uint a, uint b)`: cloth-fit's predicate on the garment flags. -/
def fnCanCollide : SlangFunctionDecl :=
  { retType := bT, name := "can_collide", params := [pIn "a" uT, pIn "b" uT]
  , body :=
      [ .declare bT "ga" (some (ne (ix "garment" (v "a")) (u 0)))
      , .declare bT "gb" (some (ne (ix "garment" (v "b")) (u 0)))
      , .ifThen (ne (prm "self_collision") (u 0)) [.ret (some (bor (v "ga") (v "gb")))] []
      , .ret (some (ne (v "ga") (v "gb"))) ] }

/-- `bool overlaps(uint a, uint b)`: the boxes of primitives `a` and `b`
    (`min <= other.max` on every axis, both ways, as `AABB::intersects`). -/
def fnOverlaps : SlangFunctionDecl :=
  { retType := bT, name := "overlaps", params := [pIn "a" uT, pIn "b" uT]
  , body :=
      [ .declare uT "oa" (some (mul (u 6) (v "a")))
      , .declare uT "ob" (some (mul (u 6) (v "b")))
      , .forCount "d" (u 0) (u 3)
          [ .ifThen (.bin ">" (ix "boxes" (add (v "oa") (v "d"))) (ix "boxes" (add (add (v "ob") (u 3)) (v "d")))) [.ret (some (.litBool false))] []
          , .ifThen (.bin ">" (ix "boxes" (add (v "ob") (v "d"))) (ix "boxes" (add (add (v "oa") (u 3)) (v "d")))) [.ret (some (.litBool false))] [] ]
      , .ret (some (.litBool true)) ] }

/-- `bool garment_face(uint f)`: a face with a garment vertex. -/
def fnGarmentFace : SlangFunctionDecl :=
  { retType := bT, name := "garment_face", params := [pIn "f" uT]
  , body :=
      [ .declare uT "b" (some (mul (u 3) (v "f")))
      , .ret (some (bor (bor (ne (ix "garment" (ix "faces" (v "b"))) (u 0))
                             (ne (ix "garment" (ix "faces" (add (v "b") (u 1)))) (u 0)))
                        (ne (ix "garment" (ix "faces" (add (v "b") (u 2)))) (u 0)))) ] }

/-- Emit one pair `(a, b)` (write mode) and count it. -/
private def emitPair (a b : E) : List St :=
  [ .ifThen (ne (prm "write") (u 0))
      [ .assign (ix "pairs" (mul (u 2) (add (v "o") (v "c")))) a
      , .assign (ix "pairs" (add (mul (u 2) (add (v "o") (v "c"))) (u 1))) b ]
      []
  , .assign (v "c") (add (v "c") (u 1)) ]

/-- kind 0: edge `id` against every edge `j`. -/
private def edgeLoop : List St :=
  [ .declare uT "a0" (some (ix "edges" (mul (u 2) (v "id"))))
  , .declare uT "a1" (some (ix "edges" (add (mul (u 2) (v "id")) (u 1))))
  , .forCount "j" (u 0) (prm "n_e")
      [ .ifThen (ne (v "j") (v "id"))
          [ .declare uT "b0" (some (ix "edges" (mul (u 2) (v "j"))))
          , .declare uT "b1" (some (ix "edges" (add (mul (u 2) (v "j")) (u 1))))
          , .declare bT "share" (some (bor (bor (eq (v "a0") (v "b0")) (eq (v "a0") (v "b1")))
                                            (bor (eq (v "a1") (v "b0")) (eq (v "a1") (v "b1")))))
          -- a target that is itself a query is left to the lower-numbered edge
          , .declare bT "target_is_query" (some (bor (ne (ix "garment" (v "b0")) (u 0)) (ne (ix "garment" (v "b1")) (u 0))))
          , .declare bT "mine" (some (bor (bnot (v "target_is_query")) (.bin ">" (v "j") (v "id"))))
          , .ifThen (band (band (bnot (v "share")) (v "mine"))
                          (bor (bor (.call "can_collide" [v "a0", v "b0"]) (.call "can_collide" [v "a0", v "b1"]))
                               (bor (.call "can_collide" [v "a1", v "b0"]) (.call "can_collide" [v "a1", v "b1"]))))
              [ .ifThen (.call "overlaps" [add (prm "n_v") (v "id"), add (prm "n_v") (v "j")])
                  (emitPair (.call "min" [v "id", v "j"]) (.call "max" [v "id", v "j"])) [] ]
              [] ]
          [] ] ]

/-- kind 1: garment vertex `id` against every face without a garment vertex. -/
private def vertexLoop : List St :=
  [ .forCount "f" (u 0) (prm "n_f")
      [ .declare uT "b" (some (mul (u 3) (v "f")))
      , .declare uT "f0" (some (ix "faces" (v "b")))
      , .declare uT "f1" (some (ix "faces" (add (v "b") (u 1))))
      , .declare uT "f2" (some (ix "faces" (add (v "b") (u 2))))
      , .declare bT "inface" (some (bor (bor (eq (v "id") (v "f0")) (eq (v "id") (v "f1"))) (eq (v "id") (v "f2"))))
      , .ifThen (band (band (bnot (v "inface")) (bnot (.call "garment_face" [v "f"])))
                      (bor (bor (.call "can_collide" [v "id", v "f0"]) (.call "can_collide" [v "id", v "f1"]))
                           (.call "can_collide" [v "id", v "f2"])))
          [ .ifThen (.call "overlaps" [v "id", add (add (prm "n_v") (prm "n_e")) (v "f")])
              (emitPair (v "f") (v "id")) [] ]
          [] ] ]

/-- kind 2: face `id` (with a garment vertex) against every vertex. -/
private def faceLoop : List St :=
  [ .declare uT "b" (some (mul (u 3) (v "id")))
  , .declare uT "f0" (some (ix "faces" (v "b")))
  , .declare uT "f1" (some (ix "faces" (add (v "b") (u 1))))
  , .declare uT "f2" (some (ix "faces" (add (v "b") (u 2))))
  , .forCount "w" (u 0) (prm "n_v")
      [ .declare bT "inface" (some (bor (bor (eq (v "w") (v "f0")) (eq (v "w") (v "f1"))) (eq (v "w") (v "f2"))))
      , .ifThen (band (bnot (v "inface"))
                      (bor (bor (.call "can_collide" [v "w", v "f0"]) (.call "can_collide" [v "w", v "f1"]))
                           (.call "can_collide" [v "w", v "f2"])))
          [ .ifThen (.call "overlaps" [v "w", add (add (prm "n_v") (prm "n_e")) (v "id")])
              (emitPair (v "id") (v "w")) [] ]
          [] ] ]

def body : List St :=
  [ .declare uT "q" (some (.member (v "tid") "x"))
  , .ifThen (.bin ">=" (v "q") (prm "n_queries")) [.ret none] []
  , .declare uT "code" (some (ix "queries" (v "q")))
  , .declare uT "kind" (some (.bin ">>" (v "code") (u 30)))
  , .declare uT "id" (some (.bin "&" (v "code") (u 0x3fffffff)))
  , .declare uT "c" (some (u 0))
  , .declare uT "o" (some (.ternary (ne (prm "write") (u 0)) (ix "counts" (v "q")) (u 0)))
  , .ifThen (eq (v "kind") (u 0)) edgeLoop
      [ .ifThen (eq (v "kind") (u 1)) vertexLoop faceLoop ]
  , .ifThen (eq (prm "write") (u 0)) [ .assign (ix "counts" (v "q")) (v "c") ] [] ]

private def glob (n : String) (ty : SlangType) (b : Nat) : SlangBinding :=
  ⟨n, ty, Semantic.none, some b, some 0, .qIn⟩

def shader : SlangShaderModule :=
  { structs :=
      [ { name := "PairParams"
        , fields := [pIn "n_queries" uT, pIn "n_v" uT, pIn "n_e" uT, pIn "n_f" uT, pIn "self_collision" uT, pIn "write" uT] } ]
  , globals :=
      [ glob "queries" (.roBuf uT) 0
      , glob "boxes"   (.roBuf fT) 1
      , glob "edges"   (.roBuf uT) 2
      , glob "faces"   (.roBuf uT) 3
      , glob "garment" (.roBuf uT) 4
      , glob "counts"  (.rwBuf uT) 5
      , glob "pairs"   (.rwBuf uT) 6
      , glob "params"  (.const "PairParams") 7 ]
  , functions :=
      [ fnCanCollide, fnOverlaps, fnGarmentFace
      , { attrs  := [.shaderCompute, .numthreads 64 1 1]
          name   := "main"
          params := [{ name := "tid", type := .vec .uint 3
                     , semantic := Semantic.svDispatchThreadId }]
          body   := body } ] }

example : shader.entryPointName = "main" := by native_decide

end Fit.SlangCodegen.BoxPair
