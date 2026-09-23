import LeanSlang
import Ggml.SlangCodegen.Common

/-!
# `Ggml.SlangCodegen.MulMat` — what the MUL_MAT kernels share (family K6)

Not a kernel. ggml's MUL_MAT is `dst = src1 · src0ᵀ` per batch:

    src0  a  [K, M, ne02, ne03]      src1  b  [K, N, ne12, ne13]
    dst      [M, N, ne12, ne13]      (always f32)
    dst[i, j, i2, i3] = Σ_k a[k, i, i2 / r2, i3 / r3] · b[k, j, i2, i3]
    r2 = ne12 / ne02,  r3 = ne13 / ne03   (ggml's batch broadcast)

Every operand is read through its strides (`off + Σ i_k nb_k`, in
elements), so a permuted or strided src1, non-contiguous src0 rows (the
census's RC and RR classes) and a strided dst all go through the same
code. f16 and bf16 operands stay in `uint` word buffers and are widened
exactly on load (`Common.fnLdF16`, `Common.fnLdBf16`: bf16 is the top half
of an f32); the products are summed in f32 whatever the inputs, which is
what `GGML_PREC_F32` asks for.

The kernels (`MulMatTiled`, `MulMatVec`, and their single-thread-per-output
siblings in `MulMatSerial`) differ only in how threads share the work; this
module holds the operand types, the load helpers, the words and the
preamble they all start with.

## Derived words (55-57; 53 and 54 are not used)

    55  r2 = ne12 / ne02           56  r3 = ne13 / ne03
    57  row-block groups along x (MulMatVec's 2-D split of its row blocks)
-/

namespace Ggml.SlangCodegen.MulMat

open LeanSlang
open Ggml.SlangCodegen.Common

/-- The operand types the kernels take. -/
inductive Ty
  | f32
  | f16
  | bf16
deriving Repr, BEq, DecidableEq, Inhabited

def Ty.name : Ty → String
  | .f32 => "f32"
  | .f16 => "f16"
  | .bf16 => "bf16"

/-- The element type a binding of this operand type is declared with. -/
def Ty.scalar : Ty → Scalar
  | .f32 => .float
  | _ => .uint

/-- The (src0, src1) pairs that have kernels: every census row
    (`{f32, f16, bf16} × f32`, and `f16 × f16` from DINO's im2col) and
    nothing ggml-cpu cannot check (it refuses `f32 × f16` and
    `bf16 × f16`). -/
def pairs : List (Ty × Ty) :=
  [(.f32, .f32), (.f16, .f32), (.bf16, .f32), (.f16, .f16)]

/-- Kernel name suffix, `<src0>_<src1>`. -/
def suffix (a b : Ty) : String := a.name ++ "_" ++ b.name

/-! ## Words -/

def wR2 : Nat := 55
def wR3 : Nat := 56
def wRowGroupsX : Nat := 57

example : wDerived + 2 = wR2 ∧ wR3 + 1 = wRowGroupsX ∧ wRowGroupsX < wordsPerSlot := by native_decide

/-! ## Load helpers: `float lda(uint e)` (s0) and `float ldb(uint e)` (s1) -/

private def uintParam (n : String) : SlangBinding :=
  ⟨n, .scalar .uint, Semantic.none, none, none, .qIn⟩

/-- `float <name>(uint e)`: element `e` of `buf` as a float. -/
def fnLoad (name buf : String) : Ty → SlangFunctionDecl
  | .f32 =>
    { retType := .scalar .float
    , name := name
    , params := [uintParam "e"]
    , body := [ .ret (some (.index (v buf) (v "e"))) ] }
  | .f16 => { fnLdF16 buf with name := name }
  | .bf16 => { fnLdBf16 buf with name := name }

def fnLda (a : Ty) : SlangFunctionDecl := fnLoad "lda" "s0" a
def fnLdb (b : Ty) : SlangFunctionDecl := fnLoad "ldb" "s1" b

def lda (e : SlangExpr) : SlangExpr := .call "lda" [e]
def ldb (e : SlangExpr) : SlangExpr := .call "ldb" [e]

example : LeanSlang.emitFunction (fnLda .f32) =
"float lda(uint e) {
  return s0[e];
}" := by native_decide

example : LeanSlang.emitFunction (fnLda .f16) =
"float lda(uint e) {
  uint w = s0[(e >> 1u)];
  uint h = (((e & 1u) != 0u) ? (w >> 16u) : (w & 65535u));
  return f16tof32(h);
}" := by native_decide

example : LeanSlang.emitFunction (fnLda .bf16) =
"float lda(uint e) {
  uint w = s0[(e >> 1u)];
  uint h = (((e & 1u) != 0u) ? (w >> 16u) : (w & 65535u));
  return asfloat((h << 16u));
}" := by native_decide

example : LeanSlang.emitFunction (fnLdb .f16) =
"float ldb(uint e) {
  uint w = s1[(e >> 1u)];
  uint h = (((e & 1u) != 0u) ? (w >> 16u) : (w & 65535u));
  return f16tof32(h);
}" := by native_decide

/-! ## Expression shorthands -/

def sub (a b : SlangExpr) : SlangExpr := .bin "-" a b
def ltE (a b : SlangExpr) : SlangExpr := .bin "<" a b
def and_ (a b : SlangExpr) : SlangExpr := .bin "&&" a b
def gid (c : String) : SlangExpr := .member (v "gid") c
def lid (c : String) : SlangExpr := .member (v "lid") c
def f0 : SlangExpr := .litFloatExact 0.0
def uintTy : SlangType := .scalar .uint
def floatTy : SlangType := .scalar .float
def barrier : SlangStmt := .expr (.call "GroupMemoryBarrierWithGroupSync" [])

/-- The words every MUL_MAT kernel reads first: the sizes, this group's
    batch `(b2, b3)` = `gid.z` over dst's dims 2 and 3, and the element
    offsets of that batch's matrices in s0 (broadcast: `b2 / r2`,
    `b3 / r3`), s1 and dst; then the strides the loops use. -/
def preamble : List SlangStmt :=
  let s0 := wSrc 0
  let s1 := wSrc 1
  [ .declInit uintTy "K" (pwN (s0 + tNe))
  , .declInit uintTy "M" (pwN (s0 + tNe + 1))
  , .declInit uintTy "N" (pwN (s1 + tNe + 1))
  , .declInit uintTy "ne12" (pwN (s1 + tNe + 2))
  , .declInit uintTy "b2" (urem (gid "z") (v "ne12"))
  , .declInit uintTy "b3" (udiv (gid "z") (v "ne12"))
  , .declInit uintTy "aBase"
      (add (add (pwN (s0 + tOff))
        (mul (udiv (v "b2") (pwN wR2)) (pwN (s0 + tNb + 2))))
        (mul (udiv (v "b3") (pwN wR3)) (pwN (s0 + tNb + 3))))
  , .declInit uintTy "bBase"
      (add (add (pwN (s1 + tOff)) (mul (v "b2") (pwN (s1 + tNb + 2)))) (mul (v "b3") (pwN (s1 + tNb + 3))))
  , .declInit uintTy "dBase"
      (add (add (pwN (wDst + tOff)) (mul (v "b2") (pwN (wDst + tNb + 2)))) (mul (v "b3") (pwN (wDst + tNb + 3))))
  , .declInit uintTy "ak" (pwN (s0 + tNb))
  , .declInit uintTy "am" (pwN (s0 + tNb + 1))
  , .declInit uintTy "bk" (pwN (s1 + tNb))
  , .declInit uintTy "bn" (pwN (s1 + tNb + 1))
  , .declInit uintTy "dm" (pwN (wDst + tNb))
  , .declInit uintTy "dn" (pwN (wDst + tNb + 1)) ]

/-- `dst[dBase + i * dm + j * dn] = x;` -/
def store (i j x : SlangExpr) : SlangStmt :=
  .assign (.index (v "dst") (add (add (v "dBase") (mul i (v "dm"))) (mul j (v "dn")))) x

/-- `s0 = src0's words, s1 = src1's, dst f32`; s2 unused (`uint`, kept by
    `-preserve-params`). -/
def module (a b : Ty) (groupShared : List SlangGroupSharedDecl) (main : SlangFunctionDecl) :
    SlangShaderModule :=
  { kernelModule a.scalar b.scalar .uint .float [fnPw, fnLda a, fnLdb b] main with groupShared := groupShared }

/-- The compute entry: `numthreads(x, y, 1)`, group id and thread id. -/
def entry (x y : Nat) (body : List SlangStmt) : SlangFunctionDecl :=
  { attrs := [.shaderCompute, .numthreads x y 1]
  , name := "main"
  , params := [ ⟨"gid", .vec .uint 3, .svGroupId, none, none, .qIn⟩
              , ⟨"lid", .vec .uint 3, .svGroupThreadId, none, none, .qIn⟩ ]
  , body := body }

/-- The declarations of a module over (a, b), as emitted; the pins of the
    other pairs replace f16 × f32's with these. -/
def declText (a b : Ty) : String :=
  "RWStructuredBuffer<" ++ emitScalar a.scalar ++ "> s0;\n[[vk::binding(2, 0)]]\nRWStructuredBuffer<"
    ++ emitScalar b.scalar ++ "> s1;"

/-- A module's text for (a, b), from its f16 × f32 text: the two buffer
    declarations and the two load helpers are all that differ. -/
def retype (t : String) (a b : Ty) : String :=
  ((t.replace (declText .f16 .f32) (declText a b)).replace
      (emitFunction (fnLda .f16)) (emitFunction (fnLda a))).replace
    (emitFunction (fnLdb .f32)) (emitFunction (fnLdb b))

end Ggml.SlangCodegen.MulMat
