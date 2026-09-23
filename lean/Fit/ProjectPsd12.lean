import LeanSlang
import Fit.Df32

/-!
# `Fit.SlangCodegen.ProjectPsd12` — project a 12x12 symmetric block onto the PSD cone, in df32

`ipc::project_to_psd` (ipc-toolkit `utils/eigen_ext.tpp`): eigen-decompose
the symmetric block, return it unchanged when its smallest eigenvalue is
`>= 0`, else rebuild it from the eigenvectors with every negative
eigenvalue set to 0. The CPU does that with Eigen's
`SelfAdjointEigenSolver` (tridiagonalisation plus implicit QR); one
thread here does it with cyclic Jacobi rotations, the eigensolver that
needs no pivoting, no square roots of differences and no branching on
convergence history, so the same block gives the same bits on every GPU.

Per block: `A` (row-major, 144 pairs) and `V = I`; up to 16 sweeps over
the pairs `p < q`; a sweep stops the loop when the off-diagonal energy
`sum a_pq^2` has fallen under `1e-30` of the diagonal energy (the pair's
48 bits squared; the loop is a counted `for` with a `go` flag because the
emitted Slang has no `break`). Each rotation is Numerical Recipes'
`jacobi`: `theta = (a_qq - a_pp) / (2 a_pq)`,
`t = sgn(theta) / (|theta| + sqrt(theta^2 + 1))`, `c = 1 / sqrt(t^2 + 1)`,
`s = t c`, applied to the columns then the rows of `A` and to the columns
of `V`. A `theta` beyond float range makes `t = 0`: a rotation that does
nothing to an entry already 19 orders below its diagonal.

Then `min_m a_mm >= 0` leaves the block untouched (the caller's
unchanged bits, as the CPU returns `A` itself), else
`blocks[i][j] = sum_{m : a_mm > 0} a_mm V[i][m] V[j][m]`.

Bindings:

  0  RWStructuredBuffer<df> blocks   144 per block, in place
  1  ConstantBuffer<PsdParams> params { uint count; uint enabled; }

`enabled = 0` returns at once: the solver's descent strategies turn the
projection on and off per iteration (`force_psd_projection`,
`SparseProjectedNewton` versus `SparseRegularizedNewton`), and the
kernel is dispatched either way so the compute list has one shape.
-/

namespace Fit.SlangCodegen.ProjectPsd12

open LeanSlang Fit.SlangCodegen.Df32

private def ix (buf : String) (i : E) : E := .index (v buf) i
private def at12 (buf : String) (r c : E) : E := ix buf (add (mul r (u 12)) c)
private def diag (buf : String) (r : E) : E := ix buf (mul r (u 13))
private def zero : E := call "df_make" [fl 0, fl 0]
private def one : E := call "df_f" [fl 1]

/-- `(x_p, x_q) <- (c x_p - s x_q, s x_p + c x_q)` on two addressed entries. -/
private def rotate (buf : String) (ip iq : E) (tag : String) : List St :=
  [ ld s!"{tag}p" (ix buf ip)
  , ld s!"{tag}q" (ix buf iq)
  , .assign (ix buf ip) (call "df_sub" [call "df_mul" [v "c", v s!"{tag}p"], call "df_mul" [v "s", v s!"{tag}q"]])
  , .assign (ix buf iq) (call "df_add" [call "df_mul" [v "s", v s!"{tag}p"], call "df_mul" [v "c", v s!"{tag}q"]]) ]

private def rotation : List St :=
  [ ld "apq" (at12 "A" (v "p") (v "q"))
  , .ifThen (.bin "||" (.bin "!=" (hi (v "apq")) (fl 0)) (.bin "!=" (lo (v "apq")) (fl 0)))
      [ ld "theta" (call "df_div" [call "df_sub" [diag "A" (v "q"), diag "A" (v "p")], call "df_mulf" [v "apq", fl 2]])
      , ld "at" (call "df_abs" [v "theta"])
      , ld "t" (call "df_div" [one, call "df_add" [v "at", call "df_sqrt" [call "df_add" [call "df_mul" [v "at", v "at"], one]]]])
      , .ifThen (.bin "<" (hi (v "theta")) (fl 0)) [.assign (v "t") (call "df_neg" [v "t"])] []
      , ld "c" (call "df_div" [one, call "df_sqrt" [call "df_add" [call "df_mul" [v "t", v "t"], one]]])
      , ld "s" (call "df_mul" [v "t", v "c"])
      , .forCount "r" (u 0) (u 12) (rotate "A" (add (mul (v "r") (u 12)) (v "p")) (add (mul (v "r") (u 12)) (v "q")) "ac")
      , .forCount "r" (u 0) (u 12) (rotate "A" (add (mul (v "p") (u 12)) (v "r")) (add (mul (v "q") (u 12)) (v "r")) "ar")
      , .forCount "r" (u 0) (u 12) (rotate "V" (add (mul (v "r") (u 12)) (v "p")) (add (mul (v "r") (u 12)) (v "q")) "vc") ]
      [] ]

private def sweep : List St :=
  [ .ifThen (v "go")
      [ lf "off" (fl 0)
      , lf "dg" (fl 0)
      , .forCount "p" (u 0) (u 12)
          [ .assign (v "dg") (add (v "dg") (mul (hi (diag "A" (v "p"))) (hi (diag "A" (v "p")))))
          , .forCount "q" (add (v "p") (u 1)) (u 12)
              [ .assign (v "off") (add (v "off") (mul (hi (at12 "A" (v "p") (v "q"))) (hi (at12 "A" (v "p") (v "q"))))) ] ]
      , .ifThen (.bin "<=" (v "off") (mul (v "dg") (fl 1e-30)))
          [ .assign (v "go") (.litBool false) ]
          [ .forCount "p" (u 0) (u 12)
              [ .forCount "q" (add (v "p") (u 1)) (u 12) rotation ] ] ]
      [] ]

def body : List St :=
  [ .declare uT "k" (some (.member (v "tid") "x"))
  , .ifThen (.bin ">=" (v "k") (.member (v "params") "count")) [.ret none] []
  , .ifThen (.bin "==" (.member (v "params") "enabled") (u 0)) [.ret none] []
  , .declare uT "ob" (some (mul (u 144) (v "k")))
  , .declareArray dfT "A" 144
  , .declareArray dfT "V" 144
  , .forCount "i" (u 0) (u 144)
      [ .assign (ix "A" (v "i")) (ix "blocks" (add (v "ob") (v "i")))
      , .assign (ix "V" (v "i")) zero ]
  , .forCount "i" (u 0) (u 12) [ .assign (diag "V" (v "i")) one ]
  , .declare bT "go" (some (.litBool true))
  , .forCount "sweep" (u 0) (u 16) sweep
  , .declare bT "negative" (some (.litBool false))
  , .forCount "i" (u 0) (u 12)
      [ .ifThen (call "df_lt" [diag "A" (v "i"), zero]) [.assign (v "negative") (.litBool true)] [] ]
  , .ifThen (.un "!" (v "negative")) [.ret none] []
  , .forCount "i" (u 0) (u 12)
      [ .forCount "j" (u 0) (u 12)
          [ ld "acc" zero
          , .forCount "m" (u 0) (u 12)
              [ ld "lam" (diag "A" (v "m"))
              , .ifThen (call "df_lt" [zero, v "lam"])
                  [ .assign (v "acc") (call "df_add" [v "acc", call "df_mul" [v "lam", call "df_mul" [at12 "V" (v "i") (v "m"), at12 "V" (v "j") (v "m")]]]) ]
                  [] ]
          , .assign (ix "blocks" (add (v "ob") (add (mul (v "i") (u 12)) (v "j")))) (v "acc") ] ] ]

private def glob (n : String) (ty : SlangType) (b : Nat) : SlangBinding :=
  ⟨n, ty, Semantic.none, some b, some 0, .qIn⟩

def shader : SlangShaderModule :=
  { structs :=
      [ Df32.structDecl
      , { name := "PsdParams", fields := [pIn "count" uT, pIn "enabled" uT] } ]
  , globals :=
      [ glob "blocks" (.rwBuf dfT) 0
      , glob "params" (.const "PsdParams") 1 ]
  , functions := Df32.decls ++
      [ { attrs  := [.shaderCompute, .numthreads 64 1 1]
          name   := "main"
          params := [{ name := "tid", type := .vec .uint 3
                     , semantic := Semantic.svDispatchThreadId }]
          body   := body } ] }

example : shader.entryPointName = "main" := by native_decide

end Fit.SlangCodegen.ProjectPsd12
