import LeanSlang
import Ggml.SlangCodegen.Common
import Ggml.SlangCodegen.MulMat
import Ggml.SlangCodegen.MulMatTiled

/-!
# `Ggml.SlangCodegen.MulMatVec` — MUL_MAT for ne11 ≤ 4 (decode, beams)

A group of 32 lanes × 8 rows: row `(gid.y * gx + gid.x) * 8 + lid.y` of
src0 (gx = word 57, the 2-D split of the row blocks), batch `gid.z`. Lane
`l` sums `a[k, row] · b[k, j]` over k = l, l + 32, l + 64, … for every
column j < N ≤ 4 (four f32 partials), writes them to group memory
`red[4 · lid.y + j][l]`, and the 32 partials of each (row, j) are added in
a tree, s = 16, 8, 4, 2, 1: `red[·][l] += red[·][l + s]` for l < s, a
barrier after each step. Lane j < N stores `red[4 · lid.y + j][0]`.

`MulMatSerial.vec` is the same sum in the same order with no group memory
(lane 0 of each row does all 32 lanes' partials and the tree), the cpp
target's sibling.

Grid (`guest/ggml-rd/ops/mul_mat.cpp`): (gx, ⌈⌈M/8⌉/gx⌉, ne12·ne13).
-/

namespace Ggml.SlangCodegen.MulMatVec

open LeanSlang
open Ggml.SlangCodegen.Common
open Ggml.SlangCodegen.MulMat

def lanes : Nat := 32
def rows : Nat := 8
def maxCols : Nat := 4

/-- The tree's steps: 16, 8, 4, 2, 1. -/
def steps : List Nat := [16, 8, 4, 2, 1]

example : steps.foldl (· + ·) 0 + 1 = lanes := by native_decide

def pName (j : Nat) : String := "p" ++ toString j

private def idx2 (arr : String) (i j : SlangExpr) : SlangExpr := .index (.index (v arr) i) j

/-- `ry + j`, the group-memory row of column j (ry = 4 · lid.y). -/
def redRow (j : Nat) : SlangExpr := if j == 0 then v "ry" else add (v "ry") (u j)

/-- The row index and the four partials' declarations. -/
def rowDecls : List SlangStmt :=
  [ .declInit uintTy "row"
      (add (mul (add (mul (gid "y") (pwN wRowGroupsX)) (gid "x")) (u rows)) (lid "y")) ] ++
  (List.range maxCols).map (fun j => .declInit floatTy (pName j) f0)

/-- `p_j += a · b[k, j]` for j < N, over k = k, k + 32, … (`k` declared by
    the caller). Shared with `MulMatSerial.vec`. -/
def dotLoop : SlangStmt :=
  .whileLoop (ltE (v "k") (v "K")) (
    [ .declInit floatTy "a" (lda (add (add (v "aBase") (mul (v "k") (v "ak"))) (mul (v "row") (v "am"))))
    , .declInit uintTy "bo" (add (v "bBase") (mul (v "k") (v "bk")))
    , .assign (v (pName 0)) (add (v (pName 0)) (mul (v "a") (ldb (v "bo")))) ] ++
    ((List.range maxCols).drop 1).map (fun j =>
      .ifNoElse (.bin ">" (v "N") (u j))
        [ .assign (v (pName j))
            (add (v (pName j)) (mul (v "a") (ldb (add (v "bo") (if j == 1 then v "bn" else mul (u j) (v "bn")))))) ]) ++
    [ .assign (v "k") (add (v "k") (u lanes)) ])

def body : List SlangStmt :=
  preamble ++ rowDecls ++
  [ .declInit uintTy "lane" (lid "x")
  , .declInit uintTy "ry" (mul (lid "y") (u maxCols))
  , .ifNoElse (ltE (v "row") (v "M"))
      [ .declInit uintTy "k" (v "lane"), dotLoop ] ] ++
  (List.range maxCols).map (fun j => .assign (idx2 "red" (redRow j) (v "lane")) (v (pName j))) ++
  [ barrier ] ++
  steps.flatMap (fun s =>
    [ .ifNoElse (ltE (v "lane") (u s))
        ((List.range maxCols).map (fun j =>
          .assign (idx2 "red" (redRow j) (v "lane"))
            (add (idx2 "red" (redRow j) (v "lane")) (idx2 "red" (redRow j) (add (v "lane") (u s))))))
    , barrier ]) ++
  [ .ifNoElse (and_ (ltE (v "lane") (v "N")) (ltE (v "row") (v "M")))
      [ store (v "row") (v "lane") (idx2 "red" (add (v "ry") (v "lane")) (u 0)) ] ]

def groupShared : List SlangGroupSharedDecl :=
  [ { name := "red", elemType := floatTy, dims := [rows * maxCols, lanes] } ]

def shader (a b : Ty) : SlangShaderModule :=
  module a b groupShared (entry lanes rows body)

def kernels : List (String × SlangShaderModule) :=
  pairs.map (fun (a, b) => ("mul_mat_vec_" ++ suffix a b, shader a b))

def expectedMain : String :=
"[shader(\"compute\")] [numthreads(32, 8, 1)]
void main(uint3 gid : SV_GroupID, uint3 lid : SV_GroupThreadID) {
  uint K = pw(10u);
  uint M = pw(11u);
  uint N = pw(20u);
  uint ne12 = pw(21u);
  uint b2 = (gid.z % ne12);
  uint b3 = (gid.z / ne12);
  uint aBase = ((pw(18u) + ((b2 / pw(55u)) * pw(16u))) + ((b3 / pw(56u)) * pw(17u)));
  uint bBase = ((pw(27u) + (b2 * pw(25u))) + (b3 * pw(26u)));
  uint dBase = ((pw(9u) + (b2 * pw(7u))) + (b3 * pw(8u)));
  uint ak = pw(14u);
  uint am = pw(15u);
  uint bk = pw(23u);
  uint bn = pw(24u);
  uint dm = pw(5u);
  uint dn = pw(6u);
  uint row = ((((gid.y * pw(57u)) + gid.x) * 8u) + lid.y);
  float p0 = 0.0f;
  float p1 = 0.0f;
  float p2 = 0.0f;
  float p3 = 0.0f;
  uint lane = lid.x;
  uint ry = (lid.y * 4u);
  if ((row < M)) {
    uint k = lane;
    while ((k < K)) {
      float a = lda(((aBase + (k * ak)) + (row * am)));
      uint bo = (bBase + (k * bk));
      p0 = (p0 + (a * ldb(bo)));
      if ((N > 1u)) {
        p1 = (p1 + (a * ldb((bo + bn))));
      }
      if ((N > 2u)) {
        p2 = (p2 + (a * ldb((bo + (2u * bn)))));
      }
      if ((N > 3u)) {
        p3 = (p3 + (a * ldb((bo + (3u * bn)))));
      }
      k = (k + 32u);
    }
  }
  red[ry][lane] = p0;
  red[(ry + 1u)][lane] = p1;
  red[(ry + 2u)][lane] = p2;
  red[(ry + 3u)][lane] = p3;
  GroupMemoryBarrierWithGroupSync();
  if ((lane < 16u)) {
    red[ry][lane] = (red[ry][lane] + red[ry][(lane + 16u)]);
    red[(ry + 1u)][lane] = (red[(ry + 1u)][lane] + red[(ry + 1u)][(lane + 16u)]);
    red[(ry + 2u)][lane] = (red[(ry + 2u)][lane] + red[(ry + 2u)][(lane + 16u)]);
    red[(ry + 3u)][lane] = (red[(ry + 3u)][lane] + red[(ry + 3u)][(lane + 16u)]);
  }
  GroupMemoryBarrierWithGroupSync();
  if ((lane < 8u)) {
    red[ry][lane] = (red[ry][lane] + red[ry][(lane + 8u)]);
    red[(ry + 1u)][lane] = (red[(ry + 1u)][lane] + red[(ry + 1u)][(lane + 8u)]);
    red[(ry + 2u)][lane] = (red[(ry + 2u)][lane] + red[(ry + 2u)][(lane + 8u)]);
    red[(ry + 3u)][lane] = (red[(ry + 3u)][lane] + red[(ry + 3u)][(lane + 8u)]);
  }
  GroupMemoryBarrierWithGroupSync();
  if ((lane < 4u)) {
    red[ry][lane] = (red[ry][lane] + red[ry][(lane + 4u)]);
    red[(ry + 1u)][lane] = (red[(ry + 1u)][lane] + red[(ry + 1u)][(lane + 4u)]);
    red[(ry + 2u)][lane] = (red[(ry + 2u)][lane] + red[(ry + 2u)][(lane + 4u)]);
    red[(ry + 3u)][lane] = (red[(ry + 3u)][lane] + red[(ry + 3u)][(lane + 4u)]);
  }
  GroupMemoryBarrierWithGroupSync();
  if ((lane < 2u)) {
    red[ry][lane] = (red[ry][lane] + red[ry][(lane + 2u)]);
    red[(ry + 1u)][lane] = (red[(ry + 1u)][lane] + red[(ry + 1u)][(lane + 2u)]);
    red[(ry + 2u)][lane] = (red[(ry + 2u)][lane] + red[(ry + 2u)][(lane + 2u)]);
    red[(ry + 3u)][lane] = (red[(ry + 3u)][lane] + red[(ry + 3u)][(lane + 2u)]);
  }
  GroupMemoryBarrierWithGroupSync();
  if ((lane < 1u)) {
    red[ry][lane] = (red[ry][lane] + red[ry][(lane + 1u)]);
    red[(ry + 1u)][lane] = (red[(ry + 1u)][lane] + red[(ry + 1u)][(lane + 1u)]);
    red[(ry + 2u)][lane] = (red[(ry + 2u)][lane] + red[(ry + 2u)][(lane + 1u)]);
    red[(ry + 3u)][lane] = (red[(ry + 3u)][lane] + red[(ry + 3u)][(lane + 1u)]);
  }
  GroupMemoryBarrierWithGroupSync();
  if (((lane < N) && (row < M))) {
    dst[((dBase + (row * dm)) + (lane * dn))] = red[(ry + lane)][0u];
  }
}"

example : LeanSlang.emitFunction (entry lanes rows body) = expectedMain := by native_decide

/-- f16 × f32 is the tiled kernel's module with the vec's group memory and
    entry; the other pairs differ in the declarations and loads only. -/
example : LeanSlang.emit (shader .f16 .f32) =
    ((((Ggml.SlangCodegen.MulMatTiled.expected.replace
      "groupshared float As[16][65];\ngroupshared float Bs[16][65];" "groupshared float red[32][32];").splitOn
      "[shader(\"compute\")]").head!) ++ expectedMain) := by native_decide
example : ∀ p ∈ pairs, LeanSlang.emit (shader p.1 p.2) = retype (LeanSlang.emit (shader .f16 .f32)) p.1 p.2 := by
  native_decide

end Ggml.SlangCodegen.MulMatVec
