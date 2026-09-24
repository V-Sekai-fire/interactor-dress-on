import LeanSlang
import Ggml.SlangCodegen.Common
import Ggml.SlangCodegen.Rows

/-!
# `Ggml.SlangCodegen.GroupNorm` — ggml GROUP_NORM, f32

`ggml_group_norm(a, n_groups, eps)` (op_params word 0 = `n_groups`, word 1
= `eps`): dst has src0's shape; the channels (dim 2) are cut into
`n_groups` groups of `cpg = ceil(ne2 / n_groups)` channels (the last
group shorter, or empty, when `n_groups` does not divide `ne2`), and each
(group, batch i3) is normalised over its `ne0 * ne1 * step` elements
(`step` = the channels in the group):

    mean     = Σx / n,  variance = Σ(x - mean)² / n          (two passes)
    y        = (x - mean) · (1 / sqrt(variance + eps))

ggml-cpu's `ggml_compute_forward_group_norm_f32` arithmetic (which sums
in double and writes `x - mean` then scales it in place: the same
`(x - mean) * scale`). One 256-thread work group per (group, batch), in
`Rows`' shape: the group's elements are walked linearly (`e` unravelled
as `i0 = e mod ne0, i1 = e / ne0 mod ne1, i2 = start + e / (ne0 ne1)`),
grid-strided by the 256 threads, the partials tree-reduced in
group-shared memory; every operand strided (ggml-cpu asserts nb0 = 4).
An empty group (`start >= ne2`) has n = 0: its loops do nothing and it
writes nothing, as ggml-cpu. Sums are f32 in the tree's order, so
results match within test-backend-ops' NMSE, not bit for bit.

Derived words: 55 = rows = `n_groups * ne3` (the work groups). The
Serial sibling `group_norm_f32_serial` is the same program for the cpp
target (one thread per group, `sh` local, each phase a loop over `t`,
the same partials in the same order), as `Rows` describes.
-/

namespace Ggml.SlangCodegen.GroupNorm

open LeanSlang
open Ggml.SlangCodegen.Common
open Ggml.SlangCodegen.Rows

private def uintParam (n : String) : SlangBinding :=
  ⟨n, .scalar .uint, Semantic.none, none, none, .qIn⟩

/-- `uint gidx(uint w, uint e, uint n0, uint n1, uint c0, uint i3)`: the
    element offset, in the tensor whose block starts at word `w`, of the
    `e`-th element of the group that starts at channel `c0` of batch `i3`. -/
def fnGidx : SlangFunctionDecl :=
  { retType := ui
  , name := "gidx"
  , params := [ uintParam "w", uintParam "e", uintParam "n0", uintParam "n1", uintParam "c0", uintParam "i3" ]
  , body :=
      [ .declInit ui "r" (udiv (v "e") (v "n0"))
      , .ret (some (.call "off4" [ v "w", urem (v "e") (v "n0"), urem (v "r") (v "n1")
                                 , add (v "c0") (udiv (v "r") (v "n1")), v "i3" ])) ] }

def helpers : List SlangFunctionDecl := [fnPw, fnPf, fnOff4, fnGidx]

/-- Element `i` of this group of s0. -/
def ldG (i : SlangExpr) : SlangExpr :=
  .index (v "s0") (.call "gidx" [u (wSrc 0), i, v "ne0", v "ne1", v "start", v "i3"])
/-- `dst[gidx(dst, i)] = val;` -/
def stG (i val : SlangExpr) : SlangStmt :=
  .assign (.index (v "dst") (.call "gidx" [u wDst, i, v "ne0", v "ne1", v "start", v "i3"])) val

/-- The group's geometry: which group and batch this work group is, its
    channel range, its element count `n`, and eps. -/
def prologueG : List SlangStmt :=
  let s0 := wSrc 0
  [ .declInit ui "ng" (pwN wOpParams)
  , .declInit ui "g" (urem (v "row") (v "ng"))
  , .declInit ui "i3" (udiv (v "row") (v "ng"))
  , .declInit ui "ne0" (pwN (s0 + tNe))
  , .declInit ui "ne1" (pwN (s0 + tNe + 1))
  , .declInit ui "ne2" (pwN (s0 + tNe + 2))
  , .declInit ui "cpg" (udiv (.bin "-" (add (v "ne2") (v "ng")) (u 1)) (v "ng"))
  , .declInit ui "start" (mul (v "g") (v "cpg"))
  , .declInit ui "stop" (.ternary (lt (add (v "start") (v "cpg")) (v "ne2")) (add (v "start") (v "cpg")) (v "ne2"))
  , .declInit ui "n" (.ternary (lt (v "start") (v "stop")) (mul (mul (v "ne0") (v "ne1")) (.bin "-" (v "stop") (v "start"))) (u 0))
  , .declInit fl "eps" (pfN (wOpParams + 1)) ]

/-- `Rows.partialSum` over the group's elements. -/
def partialG (tg : Nat) (name : String) (init : SlangExpr) (f : SlangExpr) : List SlangStmt :=
  let acc := "acc_" ++ name
  let i := "i_" ++ name
  [ .declInit fl acc init
  , .declInit ui i (v "t")
  , .whileLoop (lt (v i) (v "n"))
      [ .declInit fl "x" (ldG (v i))
      , .assign (v acc) (add (v acc) f)
      , .assign (v i) (add (v i) (u tg)) ]
  , .assign (sh (v "t")) (v acc) ]

/-- `Rows.reduce` (a sum) over the group's elements. -/
def reduceG (tg : Nat) (var : Variant) (name : String) (f : SlangExpr) : List SlangStmt :=
  let step (s : Nat) : SlangStmt := .assign (sh (v "t")) (add (sh (v "t")) (sh (add (v "t") (u s))))
  match var with
  | .par =>
      partialG tg name (fLit 0.0) f ++ [barrier] ++
      ((treeSteps tg).map (fun s => [ .ifNoElse (lt (v "t") (u s)) [step s], barrier ])).flatten ++
      [ .declInit fl name (sh (u 0)), barrier ]
  | .ser =>
      [ .forCount "t" (u 0) (u tg) (partialG tg name (fLit 0.0) f) ] ++
      (treeSteps tg).map (fun s => .forCount "t" (u 0) (u s) [step s]) ++
      [ .declInit fl name (sh (u 0)) ]

/-- `Rows.mapRow` over the group's elements. -/
def mapG (tg : Nat) (var : Variant) (g : SlangExpr) : List SlangStmt :=
  let body := [ .declInit fl "x" (ldG (v "j")), stG (v "j") g ]
  match var with
  | .par =>
      [ .declInit ui "j" (v "t")
      , .whileLoop (lt (v "j") (v "n")) (body ++ [ .assign (v "j") (add (v "j") (u tg)) ]) ]
  | .ser => [ .forCount "j" (u 0) (v "n") body ]

def x : SlangExpr := v "x"

def groupNormBody (tg : Nat) (var : Variant) : List SlangStmt :=
  prologueG ++
  reduceG tg var "sumx" x ++
  [ .declInit fl "mean" (fdiv (v "sumx") (floatOf (v "n"))) ] ++
  reduceG tg var "sumd" (mul (sub x (v "mean")) (sub x (v "mean"))) ++
  [ .declInit fl "variance" (fdiv (v "sumd") (floatOf (v "n")))
  , .declInit fl "scale" (fdiv (fLit 1.0) (.call "sqrt" [add (v "variance") (v "eps")])) ] ++
  mapG tg var (mul (sub x (v "mean")) (v "scale"))

/-- The entry: `Rows.entryRows` without its row prologue (the geometry
    here is the group's). -/
def entryG (tg : Nat) (var : Variant) (body : List SlangStmt) : SlangFunctionDecl :=
  { attrs := [.shaderCompute, .numthreads (if var == .par then tg else 1) 1 1]
  , name := "main"
  , params := [ ⟨"gid", .vec .uint 3, .svGroupId, none, none, .qIn⟩
              , ⟨"lid", .vec .uint 3, .svGroupThreadId, none, none, .qIn⟩ ]
  , body :=
      [ .declInit ui "row" (add (mul (.member (v "gid") "y") (pwN wGroupsX)) (.member (v "gid") "x"))
      , .ifNoElse (.bin ">=" (v "row") (pwN wRows)) [ .ret none ] ] ++
      (match var with
       | .par => [ .declInit ui "t" (.member (v "lid") "x") ]
       | .ser => [ .declareArray fl "sh" tg ]) ++
      body }

def groupNorm (tg : Nat) (var : Variant) : SlangShaderModule :=
  let m := kernelModule .float .uint .uint .float helpers (entryG tg var (groupNormBody tg var))
  match var with
  | .par => { m with groupShared := [ { name := "sh", elemType := fl, dims := [tg] } ] }
  | .ser => m

def groupNormF32 : SlangShaderModule := groupNorm threadgroup .par
def groupNormF32Serial : SlangShaderModule := groupNorm threadgroup .ser

/-- The kernels this module contributes, by their kernels.txt names. -/
def kernels : List (String × SlangShaderModule) :=
  [ ("group_norm_f32", groupNormF32)
  , ("group_norm_f32_serial", groupNormF32Serial) ]

/-! ## Pins -/

def expectedHead : String :=
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

uint gidx(uint w, uint e, uint n0, uint n1, uint c0, uint i3) {
  uint r = (e / n0);
  return off4(w, (e % n0), (r % n1), (c0 + (r / n1)), i3);
}

[shader(\"compute\")] [numthreads(256, 1, 1)]
void main(uint3 gid : SV_GroupID, uint3 lid : SV_GroupThreadID) {
  uint row = ((gid.y * pw(54u)) + gid.x);
  if ((row >= pw(55u))) {
    return;
  }
  uint t = lid.x;
  uint ng = pw(37u);
  uint g = (row % ng);
  uint i3 = (row / ng);
  uint ne0 = pw(10u);
  uint ne1 = pw(11u);
  uint ne2 = pw(12u);
  uint cpg = (((ne2 + ng) - 1u) / ng);
  uint start = (g * cpg);
  uint stop = (((start + cpg) < ne2) ? (start + cpg) : ne2);
  uint n = ((start < stop) ? ((ne0 * ne1) * (stop - start)) : 0u);
  float eps = pf(38u);
"

/-- The parallel kernel: the head, then `Rows`' two reductions (their
    text pinned there) with the group's loads, then the map. -/
def reduceText (tg : Nat) (var : Variant) (name : String) (f : SlangExpr) : String :=
  String.intercalate "\n" ((reduceG tg var name f).map (emitStmt 1))

example : LeanSlang.emit groupNormF32 =
    expectedHead ++ reduceText 256 .par "sumx" x ++ "\n" ++
"  float mean = (sumx / float(n));
" ++ reduceText 256 .par "sumd" (mul (sub x (v "mean")) (sub x (v "mean"))) ++ "\n" ++
"  float variance = (sumd / float(n));
  float scale = (1.0f / sqrt((variance + eps)));
  uint j = t;
  while ((j < n)) {
    float x = s0[gidx(10u, j, ne0, ne1, start, i3)];
    dst[gidx(1u, j, ne0, ne1, start, i3)] = ((x - mean) * scale);
    j = (j + 256u);
  }
}" := by native_decide

/-- The group reduction is `Rows`' with the group's load in place of the
    row's, and nothing else. -/
example : reduceText 256 .par "sumx" x =
    (String.intercalate "\n" ((reduce 256 .par "sumx" .sum (fLit 0.0) x).map (emitStmt 1))).replace
      "s0[(xb + (i_sumx * xs))]" "s0[gidx(10u, i_sumx, ne0, ne1, start, i3)]" := by native_decide

example : reduceText 256 .ser "sumx" x =
    (String.intercalate "\n" ((reduce 256 .ser "sumx" .sum (fLit 0.0) x).map (emitStmt 1))).replace
      "s0[(xb + (i_sumx * xs))]" "s0[gidx(10u, i_sumx, ne0, ne1, start, i3)]" := by native_decide

/-- The Serial sibling: the parallel head with `sh` local and one thread,
    the Serial reductions, and the map as a loop. -/
example : LeanSlang.emit groupNormF32Serial =
    ((expectedHead.replace "groupshared float sh[256];\n\n" "").replace
        "[numthreads(256, 1, 1)]" "[numthreads(1, 1, 1)]").replace
        "  uint t = lid.x;\n" "  float sh[256];\n"
      ++ reduceText 256 .ser "sumx" x ++ "\n" ++
"  float mean = (sumx / float(n));
" ++ reduceText 256 .ser "sumd" (mul (sub x (v "mean")) (sub x (v "mean"))) ++ "\n" ++
"  float variance = (sumd / float(n));
  float scale = (1.0f / sqrt((variance + eps)));
  for (uint j = 0u; j < n; ++j) {
    float x = s0[gidx(10u, j, ne0, ne1, start, i3)];
    dst[gidx(1u, j, ne0, ne1, start, i3)] = ((x - mean) * scale);
  }
}" := by native_decide

/-- The sibling has no group memory and no barrier; the kernel has both. -/
example : ((LeanSlang.emit groupNormF32Serial).splitOn "groupshared").length = 1 ∧
    ((LeanSlang.emit groupNormF32Serial).splitOn "GroupMemoryBarrier").length = 1 ∧
    ((LeanSlang.emit groupNormF32).splitOn "GroupMemoryBarrierWithGroupSync();").length - 1 = 20 := by
  native_decide

example : groupNormF32.entryPointName = "main" ∧ groupNormF32Serial.entryPointName = "main" := by
  native_decide

end Ggml.SlangCodegen.GroupNorm
