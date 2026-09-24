import LeanSlang
import Ggml.SlangCodegen.Common
import Ggml.SlangCodegen.Rows

/-!
# `Ggml.SlangCodegen.Reduce` — ggml SUM and SUM_ROWS, f32

Two sums, each tree-reduced in group-shared memory with a Serial sibling
for the cpp target (`Rows` says how the pair keeps the same partial sums
in the same order):

- `SUM_ROWS` (`ggml_sum_rows`): dst `[1, ne1, ne2, ne3]`, each element the
  sum of one row of src0. A row kernel on `Rows` (one 256-thread work group
  per row, grid-strided partials, the tree), MEAN without the divide:
  interactor-nx-ggml's `add_sum_last_axis`, rf-detr's layer norm
  (`ops.cpp`) and keypoint boost (`keypoints.cpp`).
- `SUM` (`ggml_sum`): dst one f32, the sum of every element of src0. One
  work group (the packer launches exactly one, `grid_1d(p, tg)`), whose
  256 threads grid-stride over src0's `ne0 * ne1 * ne2 * ne3` elements in
  ggml order (`unravel4` over src0's ne, then its strides, so a permuted
  source reads right), then the same tree; thread 0 writes `dst[offset]`.
  Word 53 is the thread count (256, or 1 for the sibling: `entry1D`'s
  guard), so both variants run one group of the packer's grid:
  interactor-nx-ggml's `add_sum_all`, rf-detr's losses.

ggml-cpu sums both in double (`ggml_vec_sum_f32`, `ggml_vec_sum_f32_ggf`)
and rounds once; these sum in f32 in the tree's order, so they match within
test-backend-ops' NMSE (1e-7), not bit for bit, like MEAN.
-/

namespace Ggml.SlangCodegen.Reduce

open LeanSlang
open Ggml.SlangCodegen.Common
open Ggml.SlangCodegen.Rows

/-! ## SUM_ROWS: a row kernel -/

def sumRowsBody (tg : Nat) (var : Variant) : List SlangStmt :=
  reduce tg var "sumx" .sum (fLit 0.0) (v "x") ++
  storeFirst var (v "sumx")

def sumRowsF32 : SlangShaderModule :=
  rowModule threadgroup .par [fnPw, fnOff4] (sumRowsBody threadgroup .par)
def sumRowsF32Serial : SlangShaderModule :=
  rowModule threadgroup .ser [fnPw, fnOff4] (sumRowsBody threadgroup .ser)

/-! ## SUM: one work group over the whole tensor -/

/-- Element `i` of src0 in ggml order, through its strides. -/
def loadAll (i : SlangExpr) : List SlangStmt :=
  [ .decl ui "j0", .decl ui "j1", .decl ui "j2", .decl ui "j3"
  , .expr (.call "unravel4" [i, u (wSrc 0), v "j0", v "j1", v "j2", v "j3"])
  , .declInit fl "x" (.index (v "s0") (.call "off4" [u (wSrc 0), v "j0", v "j1", v "j2", v "j3"])) ]

/-- Thread `t`'s strided partial over all `n` elements, into `sh[t]`. -/
def partialAll (tg : Nat) : List SlangStmt :=
  [ .declInit fl "acc" (fLit 0.0)
  , .declInit ui "i" (v "t")
  , .whileLoop (lt (v "i") (v "n"))
      (loadAll (v "i") ++
       [ .assign (v "acc") (add (v "acc") (v "x"))
       , .assign (v "i") (add (v "i") (u tg)) ])
  , .assign (sh (v "t")) (v "acc") ]

/-- The whole-tensor sum into `total`: the partials, then `Rows`' tree. -/
def reduceAll (tg : Nat) (var : Variant) : List SlangStmt :=
  let step (s : Nat) : SlangStmt := .assign (sh (v "t")) (add (sh (v "t")) (sh (add (v "t") (u s))))
  match var with
  | .par =>
      [ .declInit ui "t" (v "e") ] ++ partialAll tg ++ [barrier] ++
      ((treeSteps tg).map (fun s => [ .ifNoElse (lt (v "t") (u s)) [step s], barrier ])).flatten ++
      [ .declInit fl "total" (sh (u 0)) ]
  | .ser =>
      [ .declareArray fl "sh" tg
      , .forCount "t" (u 0) (u tg) (partialAll tg) ] ++
      (treeSteps tg).map (fun s => .forCount "t" (u 0) (u s) [step s]) ++
      [ .declInit fl "total" (sh (u 0)) ]

def sumBody (tg : Nat) (var : Variant) : List SlangStmt :=
  [ .declInit ui "n" (mul (mul (mul (pwN (wSrc 0 + tNe)) (pwN (wSrc 0 + tNe + 1)))
                                (pwN (wSrc 0 + tNe + 2))) (pwN (wSrc 0 + tNe + 3))) ] ++
  reduceAll tg var ++
  (let st := SlangStmt.assign (.index (v "dst") (pwN (wDst + tOff))) (v "total")
   match var with
   | .par => [ .ifNoElse (.bin "==" (v "e") (u 0)) [st] ]
   | .ser => [ st ])

def sumModule (tg : Nat) (var : Variant) : SlangShaderModule :=
  let m := kernelModule .float .uint .uint .float [fnPw, fnUnravel4, fnOff4]
    (entry1D (if var == .par then tg else 1) (sumBody tg var))
  match var with
  | .par => { m with groupShared := [ { name := "sh", elemType := fl, dims := [tg] } ] }
  | .ser => m

def sumF32 : SlangShaderModule := sumModule threadgroup .par
def sumF32Serial : SlangShaderModule := sumModule threadgroup .ser

/-- The kernels this module contributes, by their kernels.txt names; each
    `<k>_serial` is `<k>`'s cpp-target sibling. -/
def kernels : List (String × SlangShaderModule) :=
  [ ("sum_f32", sumF32)
  , ("sum_f32_serial", sumF32Serial)
  , ("sum_rows_f32", sumRowsF32)
  , ("sum_rows_f32_serial", sumRowsF32Serial) ]

/-! ## Pins -/

/-- SUM_ROWS is MEAN's kernel with the divide gone: one reduction, then
    thread 0 writes the row's one element. -/
example : (LeanSlang.emit sumRowsF32).endsWith
"  float sumx = sh[0u];
  GroupMemoryBarrierWithGroupSync();
  if ((t == 0u)) {
    dst[yb] = sumx;
  }
}" := by native_decide

example : (LeanSlang.emit sumRowsF32Serial).endsWith
"  float sumx = sh[0u];
  dst[yb] = sumx;
}" := by native_decide

example : ((LeanSlang.emit sumRowsF32).splitOn "GroupMemoryBarrierWithGroupSync();").length - 1 = 10 ∧
    ((LeanSlang.emit sumRowsF32).splitOn "groupshared float sh[256];").length = 2 ∧
    ((LeanSlang.emit sumRowsF32Serial).splitOn "groupshared").length = 1 ∧
    ((LeanSlang.emit sumRowsF32Serial).splitOn "GroupMemoryBarrier").length = 1 := by
  native_decide

def expectedSum : String :=
"struct Slot {
  uint base;
  uint pad0;
  uint pad1;
  uint pad2;
};

groupshared float sh[256];

[[vk::binding(0, 0)]]
StructuredBuffer<uint> params;
[[vk::binding(1, 0)]]
RWStructuredBuffer<float> s0;
[[vk::binding(2, 0)]]
RWStructuredBuffer<uint> s1;
[[vk::binding(3, 0)]]
RWStructuredBuffer<uint> s2;
[[vk::binding(4, 0)]]
RWStructuredBuffer<float> dst;
[[vk::binding(0, 1)]]
ConstantBuffer<Slot> slot;

uint pw(uint k) {
  return params[(slot.base + k)];
}

void unravel4(uint e, uint w, out uint i0, out uint i1, out uint i2, out uint i3) {
  uint n0 = pw(w);
  uint n1 = pw((w + 1u));
  uint n2 = pw((w + 2u));
  i0 = (e % n0);
  uint r = (e / n0);
  i1 = (r % n1);
  r = (r / n1);
  i2 = (r % n2);
  i3 = (r / n2);
}

uint off4(uint w, uint i0, uint i1, uint i2, uint i3) {
  return ((((pw((w + 8u)) + (i0 * pw((w + 4u)))) + (i1 * pw((w + 5u)))) + (i2 * pw((w + 6u)))) + (i3 * pw((w + 7u))));
}

[shader(\"compute\")] [numthreads(256, 1, 1)]
void main(uint3 gid : SV_GroupID, uint3 lid : SV_GroupThreadID) {
  uint e = ((((gid.y * pw(54u)) + gid.x) * 256u) + lid.x);
  if ((e >= pw(53u))) {
    return;
  }
  uint n = (((pw(10u) * pw(11u)) * pw(12u)) * pw(13u));
  uint t = e;
  float acc = 0.0f;
  uint i = t;
  while ((i < n)) {
    uint j0;
    uint j1;
    uint j2;
    uint j3;
    unravel4(i, 10u, j0, j1, j2, j3);
    float x = s0[off4(10u, j0, j1, j2, j3)];
    acc = (acc + x);
    i = (i + 256u);
  }
  sh[t] = acc;
  GroupMemoryBarrierWithGroupSync();
  if ((t < 128u)) {
    sh[t] = (sh[t] + sh[(t + 128u)]);
  }
  GroupMemoryBarrierWithGroupSync();
  if ((t < 64u)) {
    sh[t] = (sh[t] + sh[(t + 64u)]);
  }
  GroupMemoryBarrierWithGroupSync();
  if ((t < 32u)) {
    sh[t] = (sh[t] + sh[(t + 32u)]);
  }
  GroupMemoryBarrierWithGroupSync();
  if ((t < 16u)) {
    sh[t] = (sh[t] + sh[(t + 16u)]);
  }
  GroupMemoryBarrierWithGroupSync();
  if ((t < 8u)) {
    sh[t] = (sh[t] + sh[(t + 8u)]);
  }
  GroupMemoryBarrierWithGroupSync();
  if ((t < 4u)) {
    sh[t] = (sh[t] + sh[(t + 4u)]);
  }
  GroupMemoryBarrierWithGroupSync();
  if ((t < 2u)) {
    sh[t] = (sh[t] + sh[(t + 2u)]);
  }
  GroupMemoryBarrierWithGroupSync();
  if ((t < 1u)) {
    sh[t] = (sh[t] + sh[(t + 1u)]);
  }
  GroupMemoryBarrierWithGroupSync();
  float total = sh[0u];
  if ((e == 0u)) {
    dst[pw(9u)] = total;
  }
}"

example : LeanSlang.emit sumF32 = expectedSum := by native_decide

/-- The Serial sibling: one thread, `sh` local, each phase a loop over `t`. -/
example : (LeanSlang.emit sumF32Serial).endsWith
"[shader(\"compute\")] [numthreads(1, 1, 1)]
void main(uint3 gid : SV_GroupID, uint3 lid : SV_GroupThreadID) {
  uint e = ((((gid.y * pw(54u)) + gid.x) * 1u) + lid.x);
  if ((e >= pw(53u))) {
    return;
  }
  uint n = (((pw(10u) * pw(11u)) * pw(12u)) * pw(13u));
  float sh[256];
  for (uint t = 0u; t < 256u; ++t) {
    float acc = 0.0f;
    uint i = t;
    while ((i < n)) {
      uint j0;
      uint j1;
      uint j2;
      uint j3;
      unravel4(i, 10u, j0, j1, j2, j3);
      float x = s0[off4(10u, j0, j1, j2, j3)];
      acc = (acc + x);
      i = (i + 256u);
    }
    sh[t] = acc;
  }
  for (uint t = 0u; t < 128u; ++t) {
    sh[t] = (sh[t] + sh[(t + 128u)]);
  }
  for (uint t = 0u; t < 64u; ++t) {
    sh[t] = (sh[t] + sh[(t + 64u)]);
  }
  for (uint t = 0u; t < 32u; ++t) {
    sh[t] = (sh[t] + sh[(t + 32u)]);
  }
  for (uint t = 0u; t < 16u; ++t) {
    sh[t] = (sh[t] + sh[(t + 16u)]);
  }
  for (uint t = 0u; t < 8u; ++t) {
    sh[t] = (sh[t] + sh[(t + 8u)]);
  }
  for (uint t = 0u; t < 4u; ++t) {
    sh[t] = (sh[t] + sh[(t + 4u)]);
  }
  for (uint t = 0u; t < 2u; ++t) {
    sh[t] = (sh[t] + sh[(t + 2u)]);
  }
  for (uint t = 0u; t < 1u; ++t) {
    sh[t] = (sh[t] + sh[(t + 1u)]);
  }
  float total = sh[0u];
  dst[pw(9u)] = total;
}" := by native_decide

example : ((LeanSlang.emit sumF32Serial).splitOn "groupshared").length = 1 ∧
    ((LeanSlang.emit sumF32Serial).splitOn "GroupMemoryBarrier").length = 1 := by native_decide

end Ggml.SlangCodegen.Reduce
