import LeanSlang
import Ggml.SlangCodegen.Common
import Ggml.SlangCodegen.MulMat

/-!
# `Ggml.SlangCodegen.MulMatTiled` — MUL_MAT, a 64 × 64 tile per work group

The matrix-matrix kernel (ne11 > 4). A group of 16 × 16 threads computes a
64 × 64 tile of one batch's dst: rows `m0 = gid.x * 64` of src0, columns
`n0 = gid.y * 64` of src1, batch `gid.z` (`MulMat.preamble`). K is walked
in slabs of BK = 16:

    As[16][65], Bs[16][65]   groupshared f32 (65: a padded row, no bank conflicts)
    load    thread t = 16 lid.y + lid.x fills 4 elements of each slab:
            idx = t + 256 l, k = idx % 16, row = idx / 16 (k fastest: 16
            neighbouring threads read 16 neighbouring k of one row),
            converted to f32 on load, 0 outside [0, K) × [0, M) (or N)
    barrier
    compute a 4 × 4 register block per thread: rows lid.x + 16 r,
            columns lid.y + 16 c (r, c in 0..3), c_rc += As[k][row] · Bs[k][col]
            for the slab's 16 k in order
    barrier

and stores the block's in-range elements. Each output is one f32 sum over
k = 0, 1, …, K-1 in order (plus zero products for the padded slab tail),
which is what `MulMatSerial.tiled` computes with one thread per 4 × 4
block and no group memory: that sibling is the cpp target's
(`slangc -target cpp` rejects `GroupMemoryBarrierWithGroupSync`, E36107).

Grid (`guest/ggml-rd/ops/mul_mat.cpp`): (⌈M/64⌉, ⌈N/64⌉, ne12·ne13).
-/

namespace Ggml.SlangCodegen.MulMatTiled

open LeanSlang
open Ggml.SlangCodegen.Common
open Ggml.SlangCodegen.MulMat

def tile : Nat := 64
def bk : Nat := 16
def threads : Nat := 16
def block : Nat := 4

example : threads * block = tile ∧ tile * bk / (threads * threads) = 4 := by native_decide

def cName (r c : Nat) : String := "c" ++ toString r ++ toString c

private def idx2 (arr : String) (i j : SlangExpr) : SlangExpr := .index (.index (v arr) i) j

/-- Rows / columns of the 4 × 4 block: `lid.x + 16 r`, `lid.y + 16 c`. -/
def rowOf (r : Nat) : SlangExpr := if r == 0 then lid "x" else add (lid "x") (u (16 * r))
def colOf (c : Nat) : SlangExpr := if c == 0 then lid "y" else add (lid "y") (u (16 * c))

def rc : List (Nat × Nat) := (List.range block).flatMap (fun r => (List.range block).map (fun c => (r, c)))

/-- One slab's loads: element `idx` of the 16 × 64 slab of A and of B. -/
def loadSlab : List SlangStmt :=
  [ .declInit uintTy "idx" (add (v "t") (mul (v "l") (u (threads * threads))))
  , .declInit uintTy "kk" (urem (v "idx") (u bk))
  , .declInit uintTy "rr" (udiv (v "idx") (u bk))
  , .declInit uintTy "gk" (add (v "k0") (v "kk"))
  , .declInit uintTy "gm" (add (v "m0") (v "rr"))
  , .declInit floatTy "av" f0
  , .ifNoElse (and_ (ltE (v "gk") (v "K")) (ltE (v "gm") (v "M")))
      [ .assign (v "av") (lda (add (add (v "aBase") (mul (v "gk") (v "ak"))) (mul (v "gm") (v "am")))) ]
  , .assign (idx2 "As" (v "kk") (v "rr")) (v "av")
  , .declInit uintTy "gn" (add (v "n0") (v "rr"))
  , .declInit floatTy "bv" f0
  , .ifNoElse (and_ (ltE (v "gk") (v "K")) (ltE (v "gn") (v "N")))
      [ .assign (v "bv") (ldb (add (add (v "bBase") (mul (v "gk") (v "bk"))) (mul (v "gn") (v "bn")))) ]
  , .assign (idx2 "Bs" (v "kk") (v "rr")) (v "bv") ]

/-- One k of the slab: 4 + 4 loads from group memory (x_r, y_c), 16 multiply-adds. -/
def computeK : List SlangStmt :=
  (List.range block).map (fun r => .declInit floatTy ("x" ++ toString r) (idx2 "As" (v "q") (rowOf r))) ++
  (List.range block).map (fun c => .declInit floatTy ("y" ++ toString c) (idx2 "Bs" (v "q") (colOf c))) ++
  rc.map (fun (r, c) =>
    .assign (v (cName r c)) (add (v (cName r c)) (mul (v ("x" ++ toString r)) (v ("y" ++ toString c)))))

/-- The block's dst rows `i_r` and columns `j_c`. -/
def indexDecls : List SlangStmt :=
  (List.range block).map (fun r => .declInit uintTy ("i" ++ toString r) (add (v "m0") (rowOf r))) ++
  (List.range block).map (fun c => .declInit uintTy ("j" ++ toString c) (add (v "n0") (colOf c)))

/-- The block's stores, each bounds-checked. -/
def stores : List SlangStmt :=
  rc.map (fun (r, c) =>
    .ifNoElse (and_ (ltE (v ("i" ++ toString r)) (v "M")) (ltE (v ("j" ++ toString c)) (v "N")))
      [ store (v ("i" ++ toString r)) (v ("j" ++ toString c)) (v (cName r c)) ])

def storeBlock : List SlangStmt := indexDecls ++ stores

def body : List SlangStmt :=
  preamble ++
  [ .declInit uintTy "m0" (mul (gid "x") (u tile))
  , .declInit uintTy "n0" (mul (gid "y") (u tile))
  , .declInit uintTy "t" (add (mul (lid "y") (u threads)) (lid "x")) ] ++
  rc.map (fun (r, c) => .declInit floatTy (cName r c) f0) ++
  [ .declInit uintTy "nt" (udiv (add (v "K") (u (bk - 1))) (u bk))
  , .forCount "kt" (u 0) (v "nt")
      [ .declInit uintTy "k0" (mul (v "kt") (u bk))
      , .forCount "l" (u 0) (u (tile * bk / (threads * threads))) loadSlab
      , barrier
      , .forCount "q" (u 0) (u bk) computeK
      , barrier ] ] ++
  storeBlock

def groupShared : List SlangGroupSharedDecl :=
  [ { name := "As", elemType := floatTy, dims := [bk, tile + 1] }
  , { name := "Bs", elemType := floatTy, dims := [bk, tile + 1] } ]

def shader (a b : Ty) : SlangShaderModule :=
  module a b groupShared (entry threads threads body)

def kernels : List (String × SlangShaderModule) :=
  pairs.map (fun (a, b) => ("mul_mat_tiled_" ++ suffix a b, shader a b))

def expected : String :=
"struct Slot {
  uint base;
  uint pad0;
  uint pad1;
  uint pad2;
};

groupshared float As[16][65];
groupshared float Bs[16][65];

[[vk::binding(0, 0)]]
StructuredBuffer<uint> params;
[[vk::binding(1, 0)]]
RWStructuredBuffer<uint> s0;
[[vk::binding(2, 0)]]
RWStructuredBuffer<float> s1;
[[vk::binding(3, 0)]]
RWStructuredBuffer<uint> s2;
[[vk::binding(4, 0)]]
RWStructuredBuffer<float> dst;
[[vk::binding(0, 1)]]
ConstantBuffer<Slot> slot;

uint pw(uint k) {
  return params[(slot.base + k)];
}

float lda(uint e) {
  uint w = s0[(e >> 1u)];
  uint h = (((e & 1u) != 0u) ? (w >> 16u) : (w & 65535u));
  return f16tof32(h);
}

float ldb(uint e) {
  return s1[e];
}

[shader(\"compute\")] [numthreads(16, 16, 1)]
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
  uint m0 = (gid.x * 64u);
  uint n0 = (gid.y * 64u);
  uint t = ((lid.y * 16u) + lid.x);
  float c00 = 0.0f;
  float c01 = 0.0f;
  float c02 = 0.0f;
  float c03 = 0.0f;
  float c10 = 0.0f;
  float c11 = 0.0f;
  float c12 = 0.0f;
  float c13 = 0.0f;
  float c20 = 0.0f;
  float c21 = 0.0f;
  float c22 = 0.0f;
  float c23 = 0.0f;
  float c30 = 0.0f;
  float c31 = 0.0f;
  float c32 = 0.0f;
  float c33 = 0.0f;
  uint nt = ((K + 15u) / 16u);
  for (uint kt = 0u; kt < nt; ++kt) {
    uint k0 = (kt * 16u);
    for (uint l = 0u; l < 4u; ++l) {
      uint idx = (t + (l * 256u));
      uint kk = (idx % 16u);
      uint rr = (idx / 16u);
      uint gk = (k0 + kk);
      uint gm = (m0 + rr);
      float av = 0.0f;
      if (((gk < K) && (gm < M))) {
        av = lda(((aBase + (gk * ak)) + (gm * am)));
      }
      As[kk][rr] = av;
      uint gn = (n0 + rr);
      float bv = 0.0f;
      if (((gk < K) && (gn < N))) {
        bv = ldb(((bBase + (gk * bk)) + (gn * bn)));
      }
      Bs[kk][rr] = bv;
    }
    GroupMemoryBarrierWithGroupSync();
    for (uint q = 0u; q < 16u; ++q) {
      float x0 = As[q][lid.x];
      float x1 = As[q][(lid.x + 16u)];
      float x2 = As[q][(lid.x + 32u)];
      float x3 = As[q][(lid.x + 48u)];
      float y0 = Bs[q][lid.y];
      float y1 = Bs[q][(lid.y + 16u)];
      float y2 = Bs[q][(lid.y + 32u)];
      float y3 = Bs[q][(lid.y + 48u)];
      c00 = (c00 + (x0 * y0));
      c01 = (c01 + (x0 * y1));
      c02 = (c02 + (x0 * y2));
      c03 = (c03 + (x0 * y3));
      c10 = (c10 + (x1 * y0));
      c11 = (c11 + (x1 * y1));
      c12 = (c12 + (x1 * y2));
      c13 = (c13 + (x1 * y3));
      c20 = (c20 + (x2 * y0));
      c21 = (c21 + (x2 * y1));
      c22 = (c22 + (x2 * y2));
      c23 = (c23 + (x2 * y3));
      c30 = (c30 + (x3 * y0));
      c31 = (c31 + (x3 * y1));
      c32 = (c32 + (x3 * y2));
      c33 = (c33 + (x3 * y3));
    }
    GroupMemoryBarrierWithGroupSync();
  }
  uint i0 = (m0 + lid.x);
  uint i1 = (m0 + (lid.x + 16u));
  uint i2 = (m0 + (lid.x + 32u));
  uint i3 = (m0 + (lid.x + 48u));
  uint j0 = (n0 + lid.y);
  uint j1 = (n0 + (lid.y + 16u));
  uint j2 = (n0 + (lid.y + 32u));
  uint j3 = (n0 + (lid.y + 48u));
  if (((i0 < M) && (j0 < N))) {
    dst[((dBase + (i0 * dm)) + (j0 * dn))] = c00;
  }
  if (((i0 < M) && (j1 < N))) {
    dst[((dBase + (i0 * dm)) + (j1 * dn))] = c01;
  }
  if (((i0 < M) && (j2 < N))) {
    dst[((dBase + (i0 * dm)) + (j2 * dn))] = c02;
  }
  if (((i0 < M) && (j3 < N))) {
    dst[((dBase + (i0 * dm)) + (j3 * dn))] = c03;
  }
  if (((i1 < M) && (j0 < N))) {
    dst[((dBase + (i1 * dm)) + (j0 * dn))] = c10;
  }
  if (((i1 < M) && (j1 < N))) {
    dst[((dBase + (i1 * dm)) + (j1 * dn))] = c11;
  }
  if (((i1 < M) && (j2 < N))) {
    dst[((dBase + (i1 * dm)) + (j2 * dn))] = c12;
  }
  if (((i1 < M) && (j3 < N))) {
    dst[((dBase + (i1 * dm)) + (j3 * dn))] = c13;
  }
  if (((i2 < M) && (j0 < N))) {
    dst[((dBase + (i2 * dm)) + (j0 * dn))] = c20;
  }
  if (((i2 < M) && (j1 < N))) {
    dst[((dBase + (i2 * dm)) + (j1 * dn))] = c21;
  }
  if (((i2 < M) && (j2 < N))) {
    dst[((dBase + (i2 * dm)) + (j2 * dn))] = c22;
  }
  if (((i2 < M) && (j3 < N))) {
    dst[((dBase + (i2 * dm)) + (j3 * dn))] = c23;
  }
  if (((i3 < M) && (j0 < N))) {
    dst[((dBase + (i3 * dm)) + (j0 * dn))] = c30;
  }
  if (((i3 < M) && (j1 < N))) {
    dst[((dBase + (i3 * dm)) + (j1 * dn))] = c31;
  }
  if (((i3 < M) && (j2 < N))) {
    dst[((dBase + (i3 * dm)) + (j2 * dn))] = c32;
  }
  if (((i3 < M) && (j3 < N))) {
    dst[((dBase + (i3 * dm)) + (j3 * dn))] = c33;
  }
}"

/-- f16 × f32, the whole text. -/
example : LeanSlang.emit (shader .f16 .f32) = expected := by native_decide
/-- The other pairs differ in the two buffer declarations and the two load
    helpers only. -/
example : ∀ p ∈ pairs, LeanSlang.emit (shader p.1 p.2) = retype expected p.1 p.2 := by native_decide
example : (kernels.map Prod.fst) =
    ["mul_mat_tiled_f32_f32", "mul_mat_tiled_f16_f32", "mul_mat_tiled_bf16_f32", "mul_mat_tiled_f16_f16"] := by
  native_decide

end Ggml.SlangCodegen.MulMatTiled
