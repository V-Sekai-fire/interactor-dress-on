import LeanSlang
import Ggml.SlangCodegen.Common

/-!
# `Ggml.SlangCodegen.Upscale` — ggml UPSCALE, modes NEAREST and BILINEAR, f32

## NEAREST

`ggml_interpolate(a, ne0..ne3, GGML_SCALE_MODE_NEAREST)` (and `ggml_upscale`,
which is it with `ne = a.ne * factor`): dst has the shape it was given and
each destination element copies the source element at

    i_src[k] = (i[k] * src0.ne[k]) / dst.ne[k]      (integer division)

per dimension, a gather in the shape of `Repeat` (one thread per
destination element, `dst[off(dst, i)] = s0[off(s0, i_src)]`, both sides
strided). No arithmetic on the value, so it is a bit copy.

ggml-cpu (`ggml_compute_forward_upscale_f32`) writes the same index as
`(int64_t)(i / sf)` with `sf = (float) dst.ne / src0.ne`: a float divide
truncated. For an integer factor `sf` is exact and `i / sf` truncates to
`i div sf` for every i below 2^24, so the two agree; for a ratio that is
not an integer they agree wherever the float quotient does not fall on the
wrong side of an integer, which the L2 cases (`tests/ggml_rd_kernels/
cases/move.cpp`, test-backend-ops' [2,5,7,11] <-> [5,7,11,13]) check for
the shapes they run. The integer form is kept because a GPU divide may be
2.5 ULP off (the Vulkan precision floor), which for an exact integer
quotient could truncate one step low. The packer
(`guest/ggml-rd/ops/upscale.cpp`) refuses every mode but plain NEAREST
(no `GGML_SCALE_FLAG_ALIGN_CORNERS`, which changes `sf`, no ANTIALIAS)
and any dimension where `i * src0.ne` would overflow a word.
-/

namespace Ggml.SlangCodegen.Upscale

open LeanSlang
open Ggml.SlangCodegen.Common

/-- Threads per work group. -/
def threadgroup : Nat := 256

private def uintParam (n : String) : SlangBinding :=
  ⟨n, .scalar .uint, Semantic.none, none, none, .qIn⟩

/-- `uint src_idx(uint i, uint nd, uint ns) { return (i * ns) / nd; }`:
    the nearest source index of destination index `i` in a dimension of
    `nd` destination and `ns` source elements. -/
def fnSrcIdx : SlangFunctionDecl :=
  { retType := .scalar .uint
  , name := "src_idx"
  , params := [uintParam "i", uintParam "nd", uintParam "ns"]
  , body := [ .ret (some (udiv (mul (v "i") (v "ns")) (v "nd"))) ] }

/-- The body after the 1-D prologue: unravel, the source index per
    dimension, one copy. -/
def body : List SlangStmt :=
  let s0 := wSrc 0
  let si (k : Nat) : SlangExpr :=
    .call "src_idx" [v ("i" ++ toString k), pwN (wDst + tNe + k), pwN (s0 + tNe + k)]
  [ .decl (.scalar .uint) "i0"
  , .decl (.scalar .uint) "i1"
  , .decl (.scalar .uint) "i2"
  , .decl (.scalar .uint) "i3"
  , .expr (.call "unravel4" [v "e", u wDst, v "i0", v "i1", v "i2", v "i3"])
  , .declInit (.scalar .uint) "d" (.call "off4" [u wDst, v "i0", v "i1", v "i2", v "i3"])
  , .declInit (.scalar .uint) "a" (.call "off4" [u s0, si 0, si 1, si 2, si 3])
  , .assign (.index (v "dst") (v "d")) (.index (v "s0") (v "a")) ]

def upscaleNearestF32 : SlangShaderModule :=
  kernelModule .float .uint .uint .float
    [fnPw, fnUnravel4, fnOff4, fnSrcIdx]
    (entry1D threadgroup body)

/-! ## BILINEAR

`ggml_interpolate(a, ne0..ne3, GGML_SCALE_MODE_BILINEAR [| ALIGN_CORNERS])`
(rf-detr's segmentation head, `segmentation.cpp`), ggml-cpu's
`ggml_compute_forward_upscale_f32` bilinear branch, f32 op for f32 op:

    sf0 = float(ne0) / float(ne00),  sf1 likewise;  po = 0.5
    ALIGN_CORNERS (op_params word 37 bit 8): po = 0, and sf0 = (ne0 - 1) /
      (ne00 - 1) when both are > 1 (sf1 alike)
    y  = (float(i1) + po) / sf1 - po;  y0 = int(floor(y)); y1 = y0 + 1
    y0, y1 clamped to [0, ne01 - 1];  dy = y - float(y0), clamped to [0, 1]
    (x0, x1, dx from i0, sf0, ne00 the same way)
    val = a (1-dx)(1-dy) + b dx (1-dy) + c (1-dx) dy + d dx dy
          a = src[x0, y0], b = src[x1, y0], c = src[x0, y1], d = src[x1, y1]

in ggml-cpu's order of operations, with its `std::min`/`std::max` as the
ternaries they are; dims 2 and 3 are nearest (`src_idx`, as ggml-cpu's
`i2 / sf2` truncated; the same argument as above). The host L2 reference
(ggml-cpu built with `-mfma`) contracts the four-term blend into FMAs
where the cpp emit does not, so the two agree to one ULP (NMSE ~1e-15),
not bit for bit; on the GPU a divide 2.5 ULP off can also move `floor`
at an integer boundary, where the two taps it chooses between are one
source pixel apart with a weight at 0 or 1, so the value moves by float
noise. -/

private def intParam (n : String) : SlangBinding :=
  ⟨n, .scalar .int, Semantic.none, none, none, .qIn⟩
private def floatParam (n : String) : SlangBinding :=
  ⟨n, .scalar .float, Semantic.none, none, none, .qIn⟩

def ti : SlangType := .scalar .int
def tf : SlangType := .scalar .float
def fLit (x : Float) : SlangExpr := .litFloatExact x
def toF (e : SlangExpr) : SlangExpr := .cast tf e
def toI (e : SlangExpr) : SlangExpr := .cast ti e
def toU (e : SlangExpr) : SlangExpr := .cast (.scalar .uint) e
def fsub (a b : SlangExpr) : SlangExpr := .bin "-" a b
def fdiv (a b : SlangExpr) : SlangExpr := .bin "/" a b

/-- `int clamp_idx(int i, int hi)`: `std::max(0, std::min(i, hi))`. -/
def fnClampIdx : SlangFunctionDecl :=
  { retType := ti
  , name := "clamp_idx"
  , params := [intParam "i", intParam "hi"]
  , body := [ .declInit ti "m" (.ternary (.bin "<" (v "hi") (v "i")) (v "hi") (v "i"))
            , .ret (some (.ternary (.bin "<" (v "m") (.litInt 0)) (.litInt 0) (v "m"))) ] }

/-- `float clamp01(float t)`: `std::max(0.0f, std::min(t, 1.0f))`. -/
def fnClamp01 : SlangFunctionDecl :=
  { retType := tf
  , name := "clamp01"
  , params := [floatParam "t"]
  , body := [ .declInit tf "m" (.ternary (.bin "<" (fLit 1.0) (v "t")) (fLit 1.0) (v "t"))
            , .ret (some (.ternary (.bin "<" (fLit 0.0) (v "m")) (v "m") (fLit 0.0))) ] }

/-- The scale factor of one dimension: `nd / ns`, or `(nd - 1) / (ns - 1)`
    under ALIGN_CORNERS when both exceed 1. -/
def sfOf (nd ns : SlangExpr) : SlangExpr :=
  .ternary (.bin "&&" (v "ac") (.bin "&&" (.bin ">" nd (u 1)) (.bin ">" ns (u 1))))
    (fdiv (toF (.bin "-" nd (u 1))) (toF (.bin "-" ns (u 1))))
    (fdiv (toF nd) (toF ns))

/-- The two source indices and the weight of one dimension: `i` the
    destination index, `sf` its factor, `ns` the source extent. -/
def axis (nm : String) (i sf ns : SlangExpr) : List SlangStmt :=
  let c := nm ++ "c"
  let c0 := nm ++ "0"
  let c1 := nm ++ "1"
  [ .declInit tf c (fsub (fdiv (add (toF i) (v "po")) sf) (v "po"))
  , .declInit ti c0 (.call "clamp_idx" [toI (.call "floor" [v c]), toI (.bin "-" ns (u 1))])
  , .declInit ti c1 (.call "clamp_idx" [add (toI (.call "floor" [v c])) (.litInt 1), toI (.bin "-" ns (u 1))])
  , .declInit tf ("d" ++ nm) (.call "clamp01" [fsub (v c) (toF (v c0))]) ]

def bodyBilinear : List SlangStmt :=
  let s0 := wSrc 0
  let tap (nm : String) (x y : SlangExpr) : SlangStmt :=
    .declInit tf nm (.index (v "s0") (.call "off4" [u s0, toU x, toU y, v "i02", v "i03"]))
  let omdx := fsub (fLit 1.0) (v "dx")
  let omdy := fsub (fLit 1.0) (v "dy")
  [ .decl (.scalar .uint) "i0"
  , .decl (.scalar .uint) "i1"
  , .decl (.scalar .uint) "i2"
  , .decl (.scalar .uint) "i3"
  , .expr (.call "unravel4" [v "e", u wDst, v "i0", v "i1", v "i2", v "i3"])
  , .declInit (.scalar .uint) "d" (.call "off4" [u wDst, v "i0", v "i1", v "i2", v "i3"])
  , .declInit (.scalar .bool) "ac" (.bin "!=" (.bin "&" (pwN wOpParams) (u 256)) (u 0))
  , .declInit tf "po" (.ternary (v "ac") (fLit 0.0) (fLit 0.5))
  , .declInit tf "sf0" (sfOf (pwN (wDst + tNe)) (pwN (s0 + tNe)))
  , .declInit tf "sf1" (sfOf (pwN (wDst + tNe + 1)) (pwN (s0 + tNe + 1))) ] ++
  axis "y" (v "i1") (v "sf1") (pwN (s0 + tNe + 1)) ++
  axis "x" (v "i0") (v "sf0") (pwN (s0 + tNe)) ++
  [ .declInit (.scalar .uint) "i02" (.call "src_idx" [v "i2", pwN (wDst + tNe + 2), pwN (s0 + tNe + 2)])
  , .declInit (.scalar .uint) "i03" (.call "src_idx" [v "i3", pwN (wDst + tNe + 3), pwN (s0 + tNe + 3)])
  , tap "a" (v "x0") (v "y0")
  , tap "b" (v "x1") (v "y0")
  , tap "c" (v "x0") (v "y1")
  , tap "dd" (v "x1") (v "y1")
  , .assign (.index (v "dst") (v "d"))
      (add (add (add (mul (mul (v "a") omdx) omdy)
                     (mul (mul (v "b") (v "dx")) omdy))
                (mul (mul (v "c") omdx) (v "dy")))
           (mul (mul (v "dd") (v "dx")) (v "dy"))) ]

def upscaleBilinearF32 : SlangShaderModule :=
  kernelModule .float .uint .uint .float
    [fnPw, fnUnravel4, fnOff4, fnSrcIdx, fnClampIdx, fnClamp01]
    (entry1D threadgroup bodyBilinear)

/-- The kernels this module contributes, by their kernels.txt names. -/
def kernels : List (String × SlangShaderModule) :=
  [ ("upscale_nearest_f32", upscaleNearestF32)
  , ("upscale_bilinear_f32", upscaleBilinearF32) ]

def expected : String :=
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

uint src_idx(uint i, uint nd, uint ns) {
  return ((i * ns) / nd);
}

[shader(\"compute\")] [numthreads(256, 1, 1)]
void main(uint3 gid : SV_GroupID, uint3 lid : SV_GroupThreadID) {
  uint e = ((((gid.y * pw(54u)) + gid.x) * 256u) + lid.x);
  if ((e >= pw(53u))) {
    return;
  }
  uint i0;
  uint i1;
  uint i2;
  uint i3;
  unravel4(e, 1u, i0, i1, i2, i3);
  uint d = off4(1u, i0, i1, i2, i3);
  uint a = off4(10u, src_idx(i0, pw(1u), pw(10u)), src_idx(i1, pw(2u), pw(11u)), src_idx(i2, pw(3u), pw(12u)), src_idx(i3, pw(4u), pw(13u)));
  dst[d] = s0[a];
}"

example : LeanSlang.emit upscaleNearestF32 = expected := by native_decide
example : upscaleNearestF32.entryPointName = "main" := by native_decide

/-- The bilinear kernel: the nearest kernel's prelude (Slot, the globals,
    pw, unravel4, off4, src_idx), then its two clamps and its entry. -/
example : LeanSlang.emit upscaleBilinearF32 =
    (expected.splitOn "[shader(").head! ++
"int clamp_idx(int i, int hi) {
  int m = ((hi < i) ? hi : i);
  return ((m < 0) ? 0 : m);
}

float clamp01(float t) {
  float m = ((1.0f < t) ? 1.0f : t);
  return ((0.0f < m) ? m : 0.0f);
}

[shader(\"compute\")] [numthreads(256, 1, 1)]
void main(uint3 gid : SV_GroupID, uint3 lid : SV_GroupThreadID) {
  uint e = ((((gid.y * pw(54u)) + gid.x) * 256u) + lid.x);
  if ((e >= pw(53u))) {
    return;
  }
  uint i0;
  uint i1;
  uint i2;
  uint i3;
  unravel4(e, 1u, i0, i1, i2, i3);
  uint d = off4(1u, i0, i1, i2, i3);
  bool ac = ((pw(37u) & 256u) != 0u);
  float po = (ac ? 0.0f : 0.5f);
  float sf0 = ((ac && ((pw(1u) > 1u) && (pw(10u) > 1u))) ? (float((pw(1u) - 1u)) / float((pw(10u) - 1u))) : (float(pw(1u)) / float(pw(10u))));
  float sf1 = ((ac && ((pw(2u) > 1u) && (pw(11u) > 1u))) ? (float((pw(2u) - 1u)) / float((pw(11u) - 1u))) : (float(pw(2u)) / float(pw(11u))));
  float yc = (((float(i1) + po) / sf1) - po);
  int y0 = clamp_idx(int(floor(yc)), int((pw(11u) - 1u)));
  int y1 = clamp_idx((int(floor(yc)) + 1), int((pw(11u) - 1u)));
  float dy = clamp01((yc - float(y0)));
  float xc = (((float(i0) + po) / sf0) - po);
  int x0 = clamp_idx(int(floor(xc)), int((pw(10u) - 1u)));
  int x1 = clamp_idx((int(floor(xc)) + 1), int((pw(10u) - 1u)));
  float dx = clamp01((xc - float(x0)));
  uint i02 = src_idx(i2, pw(3u), pw(12u));
  uint i03 = src_idx(i3, pw(4u), pw(13u));
  float a = s0[off4(10u, uint(x0), uint(y0), i02, i03)];
  float b = s0[off4(10u, uint(x1), uint(y0), i02, i03)];
  float c = s0[off4(10u, uint(x0), uint(y1), i02, i03)];
  float dd = s0[off4(10u, uint(x1), uint(y1), i02, i03)];
  dst[d] = (((((a * (1.0f - dx)) * (1.0f - dy)) + ((b * dx) * (1.0f - dy))) + ((c * (1.0f - dx)) * dy)) + ((dd * dx) * dy));
}" := by native_decide

end Ggml.SlangCodegen.Upscale
