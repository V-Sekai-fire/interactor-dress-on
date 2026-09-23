import LeanSlang
import Ggml.SlangCodegen.Common
import Ggml.SlangCodegen.MulMat
import Ggml.SlangCodegen.MulMatTiled
import Ggml.SlangCodegen.MulMatVec

/-!
# `Ggml.SlangCodegen.MulMatSerial` — the MUL_MAT kernels without group memory

`slangc -target cpp` rejects `GroupMemoryBarrierWithGroupSync` (E36107: its
per-thread dispatch runs a group's threads one after another), so the cpp
target cannot run `MulMatTiled` or `MulMatVec`. Each has a sibling here
that takes the same params words, the same thread-group size and the same
grid, and computes every output as the same f32 sum in the same order,
with no group memory and no barrier:

- `tiled`  (16 × 16 threads): each thread's 4 × 4 block of the 64 × 64
  tile, `c_rc += a[k, i_r] · b[k, j_c]` for k = 0 … K-1, operands loaded
  straight from s0/s1 (0 outside M or N, as the tiled slabs are). The
  tiled kernel's extra zero products for the padded slab tail leave the
  sums unchanged.
- `vec`    (32 × 8 threads): lane 0 of each row computes all 32 lanes'
  partials (k = l, l + 32, …) into local arrays and adds them in the
  same tree (16, 8, 4, 2, 1); the other lanes return.

`kernels/ggml/cpp_siblings.txt` pairs each GPU kernel with its sibling:
gen.sh emits cpp for the sibling only, and the L2 harness
(`tests/ggml_rd_kernels`) runs the sibling's cpp wherever the packer chose
the GPU kernel. The siblings are ordinary kernels on the fixed layout
(they compile to SPIR-V and pass the layout check too).
-/

namespace Ggml.SlangCodegen.MulMatSerial

open LeanSlang
open Ggml.SlangCodegen.Common
open Ggml.SlangCodegen.MulMat

/-! ## tiled -/

section
open Ggml.SlangCodegen.MulMatTiled

def tiledBody : List SlangStmt :=
  preamble ++
  [ .declInit uintTy "m0" (mul (gid "x") (u tile))
  , .declInit uintTy "n0" (mul (gid "y") (u tile)) ] ++
  rc.map (fun (r, c) => .declInit floatTy (cName r c) f0) ++
  indexDecls ++
  [ .forCount "k" (u 0) (v "K") (
      (List.range block).flatMap (fun r =>
        let i := v ("i" ++ toString r)
        [ .declInit floatTy ("x" ++ toString r) f0
        , .ifNoElse (ltE i (v "M"))
            [ .assign (v ("x" ++ toString r))
                (lda (add (add (v "aBase") (mul (v "k") (v "ak"))) (mul i (v "am")))) ] ]) ++
      (List.range block).flatMap (fun c =>
        let j := v ("j" ++ toString c)
        [ .declInit floatTy ("y" ++ toString c) f0
        , .ifNoElse (ltE j (v "N"))
            [ .assign (v ("y" ++ toString c))
                (ldb (add (add (v "bBase") (mul (v "k") (v "bk"))) (mul j (v "bn")))) ] ]) ++
      computeK.drop (2 * block)) ] ++
  stores

def tiled (a b : Ty) : SlangShaderModule :=
  module a b [] (entry threads threads tiledBody)

end

/-! ## vec -/

section
open Ggml.SlangCodegen.MulMatVec

def qName (j : Nat) : String := "q" ++ toString j

def vecBody : List SlangStmt :=
  preamble ++ rowDecls ++
  [ .declInit uintTy "lane" (lid "x")
  , .ifNoElse (.bin "||" (.bin "!=" (v "lane") (u 0)) (.bin ">=" (v "row") (v "M"))) [ .ret none ] ] ++
  (List.range maxCols).map (fun j => .declareArray floatTy (qName j) lanes) ++
  [ .forCount "l" (u 0) (u lanes) (
      (List.range maxCols).map (fun j => .assign (v (pName j)) f0) ++
      [ .declInit uintTy "k" (v "l"), dotLoop ] ++
      (List.range maxCols).map (fun j => .assign (.index (v (qName j)) (v "l")) (v (pName j)))) ] ++
  steps.map (fun s =>
    .forCount "l" (u 0) (u s)
      ((List.range maxCols).map (fun j =>
        .assign (.index (v (qName j)) (v "l"))
          (add (.index (v (qName j)) (v "l")) (.index (v (qName j)) (add (v "l") (u s))))))) ++
  (List.range maxCols).map (fun j =>
    .ifNoElse (.bin ">" (v "N") (u j)) [ store (v "row") (u j) (.index (v (qName j)) (u 0)) ])

def vec (a b : Ty) : SlangShaderModule :=
  module a b [] (entry lanes rows vecBody)

end

def kernels : List (String × SlangShaderModule) :=
  pairs.map (fun (a, b) => ("mul_mat_serial_tiled_" ++ suffix a b, tiled a b)) ++
  pairs.map (fun (a, b) => ("mul_mat_serial_vec_" ++ suffix a b, vec a b))

/-- The (GPU kernel, cpp sibling) pairs, as `kernels/ggml/cpp_siblings.txt`
    lists them. -/
def siblings : List (String × String) :=
  pairs.map (fun (a, b) => ("mul_mat_tiled_" ++ suffix a b, "mul_mat_serial_tiled_" ++ suffix a b)) ++
  pairs.map (fun (a, b) => ("mul_mat_vec_" ++ suffix a b, "mul_mat_serial_vec_" ++ suffix a b))

def expectedTiledMain : String :=
"[shader(\"compute\")] [numthreads(16, 16, 1)]
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
  uint i0 = (m0 + lid.x);
  uint i1 = (m0 + (lid.x + 16u));
  uint i2 = (m0 + (lid.x + 32u));
  uint i3 = (m0 + (lid.x + 48u));
  uint j0 = (n0 + lid.y);
  uint j1 = (n0 + (lid.y + 16u));
  uint j2 = (n0 + (lid.y + 32u));
  uint j3 = (n0 + (lid.y + 48u));
  for (uint k = 0u; k < K; ++k) {
    float x0 = 0.0f;
    if ((i0 < M)) {
      x0 = lda(((aBase + (k * ak)) + (i0 * am)));
    }
    float x1 = 0.0f;
    if ((i1 < M)) {
      x1 = lda(((aBase + (k * ak)) + (i1 * am)));
    }
    float x2 = 0.0f;
    if ((i2 < M)) {
      x2 = lda(((aBase + (k * ak)) + (i2 * am)));
    }
    float x3 = 0.0f;
    if ((i3 < M)) {
      x3 = lda(((aBase + (k * ak)) + (i3 * am)));
    }
    float y0 = 0.0f;
    if ((j0 < N)) {
      y0 = ldb(((bBase + (k * bk)) + (j0 * bn)));
    }
    float y1 = 0.0f;
    if ((j1 < N)) {
      y1 = ldb(((bBase + (k * bk)) + (j1 * bn)));
    }
    float y2 = 0.0f;
    if ((j2 < N)) {
      y2 = ldb(((bBase + (k * bk)) + (j2 * bn)));
    }
    float y3 = 0.0f;
    if ((j3 < N)) {
      y3 = ldb(((bBase + (k * bk)) + (j3 * bn)));
    }
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
"

/-- The serial tiled entry: the tiled kernel's preamble, block and stores
    around one k loop over global memory (the stores, from `if (((i0 < M)`
    on, are the tiled kernel's text). -/
example : LeanSlang.emitFunction (entry 16 16 tiledBody) =
    expectedTiledMain ++ "  if (((i0 < M)" ++
      "  if (((i0 < M)".intercalate ((Ggml.SlangCodegen.MulMatTiled.expected.splitOn "  if (((i0 < M)").drop 1) := by
  native_decide

def expectedVecTail : String :=
"  uint lane = lid.x;
  if (((lane != 0u) || (row >= M))) {
    return;
  }
  float q0[32];
  float q1[32];
  float q2[32];
  float q3[32];
  for (uint l = 0u; l < 32u; ++l) {
    p0 = 0.0f;
    p1 = 0.0f;
    p2 = 0.0f;
    p3 = 0.0f;
    uint k = l;
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
    q0[l] = p0;
    q1[l] = p1;
    q2[l] = p2;
    q3[l] = p3;
  }
  for (uint l = 0u; l < 16u; ++l) {
    q0[l] = (q0[l] + q0[(l + 16u)]);
    q1[l] = (q1[l] + q1[(l + 16u)]);
    q2[l] = (q2[l] + q2[(l + 16u)]);
    q3[l] = (q3[l] + q3[(l + 16u)]);
  }
  for (uint l = 0u; l < 8u; ++l) {
    q0[l] = (q0[l] + q0[(l + 8u)]);
    q1[l] = (q1[l] + q1[(l + 8u)]);
    q2[l] = (q2[l] + q2[(l + 8u)]);
    q3[l] = (q3[l] + q3[(l + 8u)]);
  }
  for (uint l = 0u; l < 4u; ++l) {
    q0[l] = (q0[l] + q0[(l + 4u)]);
    q1[l] = (q1[l] + q1[(l + 4u)]);
    q2[l] = (q2[l] + q2[(l + 4u)]);
    q3[l] = (q3[l] + q3[(l + 4u)]);
  }
  for (uint l = 0u; l < 2u; ++l) {
    q0[l] = (q0[l] + q0[(l + 2u)]);
    q1[l] = (q1[l] + q1[(l + 2u)]);
    q2[l] = (q2[l] + q2[(l + 2u)]);
    q3[l] = (q3[l] + q3[(l + 2u)]);
  }
  for (uint l = 0u; l < 1u; ++l) {
    q0[l] = (q0[l] + q0[(l + 1u)]);
    q1[l] = (q1[l] + q1[(l + 1u)]);
    q2[l] = (q2[l] + q2[(l + 1u)]);
    q3[l] = (q3[l] + q3[(l + 1u)]);
  }
  if ((N > 0u)) {
    dst[((dBase + (row * dm)) + (0u * dn))] = q0[0u];
  }
  if ((N > 1u)) {
    dst[((dBase + (row * dm)) + (1u * dn))] = q1[0u];
  }
  if ((N > 2u)) {
    dst[((dBase + (row * dm)) + (2u * dn))] = q2[0u];
  }
  if ((N > 3u)) {
    dst[((dBase + (row * dm)) + (3u * dn))] = q3[0u];
  }
}"

/-- The serial vec entry: the vec kernel's text up to its partials'
    declarations, then the per-lane loop and the same tree. -/
example : LeanSlang.emitFunction (entry 32 8 vecBody) =
    (Ggml.SlangCodegen.MulMatVec.expectedMain.splitOn "  uint lane = lid.x;\n").head! ++ expectedVecTail := by
  native_decide

/-- The modules: no group memory, the kernels' declarations and loads. -/
example : ∀ p ∈ pairs, LeanSlang.emit (tiled p.1 p.2) =
    retype ((Ggml.SlangCodegen.MulMatTiled.expected.replace
      "groupshared float As[16][65];\ngroupshared float Bs[16][65];\n\n" "").splitOn
      "[shader(\"compute\")]").head! p.1 p.2 ++ LeanSlang.emitFunction (entry 16 16 tiledBody) := by
  native_decide
example : ∀ p ∈ pairs, LeanSlang.emit (vec p.1 p.2) =
    retype ((Ggml.SlangCodegen.MulMatTiled.expected.replace
      "groupshared float As[16][65];\ngroupshared float Bs[16][65];\n\n" "").splitOn
      "[shader(\"compute\")]").head! p.1 p.2 ++ LeanSlang.emitFunction (entry 32 8 vecBody) := by
  native_decide
/-- Every GPU kernel of the family has a sibling with its thread-group size. -/
example : siblings.map Prod.fst =
    (Ggml.SlangCodegen.MulMatTiled.kernels ++ Ggml.SlangCodegen.MulMatVec.kernels).map Prod.fst := by native_decide
example : siblings.map Prod.snd = kernels.map Prod.fst := by native_decide

end Ggml.SlangCodegen.MulMatSerial
