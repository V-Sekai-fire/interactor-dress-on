import LeanSlang
import Ggml.SlangCodegen.Common
import Ggml.SlangCodegen.Rows

/-!
# `Ggml.SlangCodegen.Norm` — ggml NORM, RMS_NORM and MEAN, f32 (family K3)

Row kernels on `Rows` (one work group per row, grid-strided, tree-reduced
in group-shared memory; 256 threads, or 64 in the `_t64` kernels the packer
picks for rows of at most 1024), each with its Serial sibling for the cpp
target. ggml-cpu's arithmetic, per row of n = ne0 elements:

    NORM      mean = Σx / n;  var = Σ(x - mean)² / n          (two passes: the
              y = (x - mean) · (1 / sqrt(var + eps))           centred variance)
    RMS_NORM  ms = Σx² / n;   y = x · (1 / sqrt(ms + eps))
    MEAN      y[0] = Σx / n                                    (dst ne0 = 1)

`eps` is op_params[0] (word 37). The packer (`guest/ggml-rd/ops/norm.cpp`)
checks the shapes and sets word 55 (rows). Sums are f32 in the tree's order
(ggml-cpu sums in its own order, NORM's and MEAN's first sum and RMS_NORM's
in double), so results match within test-backend-ops' NMSE, not bit for bit.
-/

namespace Ggml.SlangCodegen.Norm

open LeanSlang
open Ggml.SlangCodegen.Common
open Ggml.SlangCodegen.Rows

def helpers : List SlangFunctionDecl := [fnPw, fnPf, fnOff4]

def x : SlangExpr := v "x"

/-- `1.0f / sqrt(m + eps)` into `scale`. -/
def scaleOf (m : String) : SlangStmt :=
  .declInit fl "scale" (fdiv (fLit 1.0) (.call "sqrt" [add (v m) (v "eps")]))

def normBody (tg : Nat) (var : Variant) : List SlangStmt :=
  [ .declInit fl "eps" (pfN wOpParams) ] ++
  reduce tg var "sumx" .sum (fLit 0.0) x ++
  [ .declInit fl "mean" (fdiv (v "sumx") (floatOf (v "n"))) ] ++
  reduce tg var "sumd" .sum (fLit 0.0) (mul (sub x (v "mean")) (sub x (v "mean"))) ++
  [ .declInit fl "variance" (fdiv (v "sumd") (floatOf (v "n")))
  , scaleOf "variance" ] ++
  mapRow tg var (mul (sub x (v "mean")) (v "scale"))

def rmsNormBody (tg : Nat) (var : Variant) : List SlangStmt :=
  [ .declInit fl "eps" (pfN wOpParams) ] ++
  reduce tg var "sumsq" .sum (fLit 0.0) (mul x x) ++
  [ .declInit fl "ms" (fdiv (v "sumsq") (floatOf (v "n")))
  , scaleOf "ms" ] ++
  mapRow tg var (mul x (v "scale"))

def meanBody (tg : Nat) (var : Variant) : List SlangStmt :=
  reduce tg var "sumx" .sum (fLit 0.0) x ++
  storeFirst var (fdiv (v "sumx") (floatOf (v "n")))

def norm (tg : Nat) (var : Variant) : SlangShaderModule := rowModule tg var helpers (normBody tg var)
def rmsNorm (tg : Nat) (var : Variant) : SlangShaderModule := rowModule tg var helpers (rmsNormBody tg var)

def normF32 : SlangShaderModule := norm threadgroup .par
def normF32Serial : SlangShaderModule := norm threadgroup .ser
def rmsNormF32 : SlangShaderModule := rmsNorm threadgroup .par
def rmsNormF32Serial : SlangShaderModule := rmsNorm threadgroup .ser
def meanF32 : SlangShaderModule := rowModule threadgroup .par [fnPw, fnOff4] (meanBody threadgroup .par)
def meanF32Serial : SlangShaderModule := rowModule threadgroup .ser [fnPw, fnOff4] (meanBody threadgroup .ser)
/-- Short rows (the census's 32-, 64- and 128-long NORM and RMS_NORM rows):
    64 threads per row, so four rows' groups fit where one 256-thread group
    would leave most of its threads idle. -/
def normF32T64 : SlangShaderModule := norm threadgroupShort .par
def normF32T64Serial : SlangShaderModule := norm threadgroupShort .ser
def rmsNormF32T64 : SlangShaderModule := rmsNorm threadgroupShort .par
def rmsNormF32T64Serial : SlangShaderModule := rmsNorm threadgroupShort .ser

/-- The kernels this module contributes, by their kernels.txt names. A
    `<k>_serial` is `<k>`'s cpp-target sibling (tests/ggml_rd_kernels runs
    it in place of `<k>`, whose group-shared memory slangc's cpp rejects). -/
def kernels : List (String × SlangShaderModule) :=
  [ ("norm_f32", normF32)
  , ("norm_f32_serial", normF32Serial)
  , ("rms_norm_f32", rmsNormF32)
  , ("rms_norm_f32_serial", rmsNormF32Serial)
  , ("mean_f32", meanF32)
  , ("mean_f32_serial", meanF32Serial)
  , ("norm_f32_t64", normF32T64)
  , ("norm_f32_t64_serial", normF32T64Serial)
  , ("rms_norm_f32_t64", rmsNormF32T64)
  , ("rms_norm_f32_t64_serial", rmsNormF32T64Serial) ]

/-! ## Pins -/

def expectedNorm : String :=
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

float pf(uint k) {
  return asfloat(pw(k));
}

uint off4(uint w, uint i0, uint i1, uint i2, uint i3) {
  return ((((pw((w + 8u)) + (i0 * pw((w + 4u)))) + (i1 * pw((w + 5u)))) + (i2 * pw((w + 6u)))) + (i3 * pw((w + 7u))));
}

[shader(\"compute\")] [numthreads(256, 1, 1)]
void main(uint3 gid : SV_GroupID, uint3 lid : SV_GroupThreadID) {
  uint row = ((gid.y * pw(54u)) + gid.x);
  if ((row >= pw(55u))) {
    return;
  }
  uint t = lid.x;
  uint n = pw(10u);
  uint r1 = (row % pw(11u));
  uint rr = (row / pw(11u));
  uint r2 = (rr % pw(12u));
  uint r3 = (rr / pw(12u));
  uint xb = off4(10u, 0u, r1, r2, r3);
  uint xs = pw(14u);
  uint yb = off4(1u, 0u, r1, r2, r3);
  uint ys = pw(5u);
  float eps = pf(37u);
  float acc_sumx = 0.0f;
  uint i_sumx = t;
  while ((i_sumx < n)) {
    float x = s0[(xb + (i_sumx * xs))];
    acc_sumx = (acc_sumx + x);
    i_sumx = (i_sumx + 256u);
  }
  sh[t] = acc_sumx;
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
  float sumx = sh[0u];
  GroupMemoryBarrierWithGroupSync();
  float mean = (sumx / float(n));
  float acc_sumd = 0.0f;
  uint i_sumd = t;
  while ((i_sumd < n)) {
    float x = s0[(xb + (i_sumd * xs))];
    acc_sumd = (acc_sumd + ((x - mean) * (x - mean)));
    i_sumd = (i_sumd + 256u);
  }
  sh[t] = acc_sumd;
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
  float sumd = sh[0u];
  GroupMemoryBarrierWithGroupSync();
  float variance = (sumd / float(n));
  float scale = (1.0f / sqrt((variance + eps)));
  uint j = t;
  while ((j < n)) {
    float x = s0[(xb + (j * xs))];
    dst[(yb + (j * ys))] = ((x - mean) * scale);
    j = (j + 256u);
  }
}"

example : LeanSlang.emit normF32 = expectedNorm := by native_decide

/-- Every pinned piece of the parallel kernels and their Serial siblings
    comes from `Rows` (whose reductions are pinned there); the siblings
    differ from their kernels in exactly the ways `Rows` states. -/
example : LeanSlang.emit normF32Serial =
"struct Slot {
  uint base;
  uint pad0;
  uint pad1;
  uint pad2;
};

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

float pf(uint k) {
  return asfloat(pw(k));
}

uint off4(uint w, uint i0, uint i1, uint i2, uint i3) {
  return ((((pw((w + 8u)) + (i0 * pw((w + 4u)))) + (i1 * pw((w + 5u)))) + (i2 * pw((w + 6u)))) + (i3 * pw((w + 7u))));
}

[shader(\"compute\")] [numthreads(1, 1, 1)]
void main(uint3 gid : SV_GroupID, uint3 lid : SV_GroupThreadID) {
  uint row = ((gid.y * pw(54u)) + gid.x);
  if ((row >= pw(55u))) {
    return;
  }
  float sh[256];
  uint n = pw(10u);
  uint r1 = (row % pw(11u));
  uint rr = (row / pw(11u));
  uint r2 = (rr % pw(12u));
  uint r3 = (rr / pw(12u));
  uint xb = off4(10u, 0u, r1, r2, r3);
  uint xs = pw(14u);
  uint yb = off4(1u, 0u, r1, r2, r3);
  uint ys = pw(5u);
  float eps = pf(37u);
  for (uint t = 0u; t < 256u; ++t) {
    float acc_sumx = 0.0f;
    uint i_sumx = t;
    while ((i_sumx < n)) {
      float x = s0[(xb + (i_sumx * xs))];
      acc_sumx = (acc_sumx + x);
      i_sumx = (i_sumx + 256u);
    }
    sh[t] = acc_sumx;
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
  float sumx = sh[0u];
  float mean = (sumx / float(n));
  for (uint t = 0u; t < 256u; ++t) {
    float acc_sumd = 0.0f;
    uint i_sumd = t;
    while ((i_sumd < n)) {
      float x = s0[(xb + (i_sumd * xs))];
      acc_sumd = (acc_sumd + ((x - mean) * (x - mean)));
      i_sumd = (i_sumd + 256u);
    }
    sh[t] = acc_sumd;
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
  float sumd = sh[0u];
  float variance = (sumd / float(n));
  float scale = (1.0f / sqrt((variance + eps)));
  for (uint j = 0u; j < n; ++j) {
    float x = s0[(xb + (j * xs))];
    dst[(yb + (j * ys))] = ((x - mean) * scale);
  }
}" := by native_decide

/-- RMS_NORM's tail after its one reduction: the mean square, the scale, x · scale. -/
example : (LeanSlang.emit rmsNormF32).endsWith
"  float sumsq = sh[0u];
  GroupMemoryBarrierWithGroupSync();
  float ms = (sumsq / float(n));
  float scale = (1.0f / sqrt((ms + eps)));
  uint j = t;
  while ((j < n)) {
    float x = s0[(xb + (j * xs))];
    dst[(yb + (j * ys))] = (x * scale);
    j = (j + 256u);
  }
}" := by native_decide

example : ((LeanSlang.emit rmsNormF32).splitOn "acc_sumsq = (acc_sumsq + (x * x));").length = 2 := by
  native_decide

/-- RMS_NORM is NORM without the mean: one reduction, not two. -/
example : ((LeanSlang.emit rmsNormF32).splitOn "GroupMemoryBarrierWithGroupSync();").length - 1 = 10 ∧
    ((LeanSlang.emit normF32).splitOn "GroupMemoryBarrierWithGroupSync();").length - 1 = 20 := by
  native_decide

/-- MEAN: one reduction, then thread 0 writes the row's one element. -/
example : (LeanSlang.emit meanF32).endsWith
"  float sumx = sh[0u];
  GroupMemoryBarrierWithGroupSync();
  if ((t == 0u)) {
    dst[yb] = (sumx / float(n));
  }
}" := by native_decide

example : (LeanSlang.emit meanF32Serial).endsWith
"  float sumx = sh[0u];
  dst[yb] = (sumx / float(n));
}" := by native_decide

/-- No Serial sibling touches group-shared memory or a barrier (the cpp
    target rejects both); every parallel kernel declares `sh` in it. -/
example : [normF32Serial, rmsNormF32Serial, meanF32Serial].all (fun m =>
      ((LeanSlang.emit m).splitOn "groupshared").length == 1 &&
      ((LeanSlang.emit m).splitOn "GroupMemoryBarrier").length == 1) ∧
    [normF32, rmsNormF32, meanF32].all (fun m =>
      ((LeanSlang.emit m).splitOn "groupshared float sh[256];").length == 2) := by
  native_decide

/-- The t64 kernels are the 256-thread ones with 64 threads, 64 partials, a
    stride of 64 and the tree's first two steps gone, and nothing else. -/
def drop128 (s : String) : String :=
  ((s.replace "  if ((t < 128u)) {\n    sh[t] = (sh[t] + sh[(t + 128u)]);\n  }\n  GroupMemoryBarrierWithGroupSync();\n" "").replace
    "  if ((t < 64u)) {\n    sh[t] = (sh[t] + sh[(t + 64u)]);\n  }\n  GroupMemoryBarrierWithGroupSync();\n" "")

example : LeanSlang.emit normF32T64 =
    ((((drop128 expectedNorm).replace "[numthreads(256, 1, 1)]" "[numthreads(64, 1, 1)]").replace
      "groupshared float sh[256];" "groupshared float sh[64];").replace " + 256u)" " + 64u)") := by
  native_decide

example : ((LeanSlang.emit rmsNormF32T64).splitOn "GroupMemoryBarrierWithGroupSync();").length - 1 = 8 ∧
    ((LeanSlang.emit normF32T64Serial).splitOn "float sh[64];").length = 2 ∧
    ((LeanSlang.emit rmsNormF32T64Serial).splitOn "for (uint t = 0u; t < 64u; ++t) {").length = 2 := by
  native_decide

end Ggml.SlangCodegen.Norm
