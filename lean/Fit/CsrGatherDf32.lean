import LeanSlang
import Fit.Df32

/-!
# `Fit.SlangCodegen.CsrGatherDf32` — weighted block entries into a fixed sparse pattern

The CPU path scatters every hinge's 144 weighted entries as triplets and
lets Eigen's `setFromTriplets` sum the duplicates. The pattern that sum
produces depends only on the mesh, so the host builds it once
(`guest/fit/fit_gpu.cpp`) and, for every nonzero of the compressed
matrix, lists the `(hinge, entry)` pairs that land on it, in ascending
order. This kernel is that sum: one thread per nonzero, a counted loop
over its contributions, each `weight[hinge] * blocks[hinge * 144 + entry]`
in df32, accumulated in the listed order. No atomics, no scatter, so a
run is bitwise repeatable (AGENTS: fixed-order gathers).

Bindings:

  0  StructuredBuffer<uint> ptr      nnz + 1: contribution range per nonzero
  1  StructuredBuffer<uint> src      hinge * 144 + entry, per contribution
  2  StructuredBuffer<df>   blocks   144 per hinge (after ProjectPsd12)
  3  StructuredBuffer<df>   weight   per hinge: 0.5 (area_i + area_j)
  4  RWStructuredBuffer<df> values   per nonzero; the host reads hi + lo as a double
  5  ConstantBuffer<GatherParams> params { uint nnz; }
-/

namespace Fit.SlangCodegen.CsrGatherDf32

open LeanSlang Fit.SlangCodegen.Df32

private def ix (buf : String) (i : E) : E := .index (v buf) i

def body : List St :=
  [ .declare uT "i" (some (.member (v "tid") "x"))
  , .ifThen (.bin ">=" (v "i") (.member (v "params") "nnz")) [.ret none] []
  , ld "acc" (call "df_make" [fl 0, fl 0])
  , .forCount "j" (ix "ptr" (v "i")) (ix "ptr" (add (v "i") (u 1)))
      [ .declare uT "s" (some (ix "src" (v "j")))
      , .declare uT "h" (some (div (v "s") (u 144)))
      , .assign (v "acc") (call "df_add" [v "acc", call "df_mul" [ix "weight" (v "h"), ix "blocks" (v "s")]]) ]
  , .assign (ix "values" (v "i")) (v "acc") ]

private def glob (n : String) (ty : SlangType) (b : Nat) : SlangBinding :=
  ⟨n, ty, Semantic.none, some b, some 0, .qIn⟩

def shader : SlangShaderModule :=
  { structs :=
      [ Df32.structDecl
      , { name := "GatherParams", fields := [pIn "nnz" uT] } ]
  , globals :=
      [ glob "ptr"    (.roBuf uT) 0
      , glob "src"    (.roBuf uT) 1
      , glob "blocks" (.roBuf dfT) 2
      , glob "weight" (.roBuf dfT) 3
      , glob "values" (.rwBuf dfT) 4
      , glob "params" (.const "GatherParams") 5 ]
  , functions := Df32.decls ++
      [ { attrs  := [.shaderCompute, .numthreads 64 1 1]
          name   := "main"
          params := [{ name := "tid", type := .vec .uint 3
                     , semantic := Semantic.svDispatchThreadId }]
          body   := body } ] }

def expected : String :=
"struct df {
  float hi;
  float lo;
};

struct GatherParams {
  uint nnz;
};

[[vk::binding(0, 0)]]
StructuredBuffer<uint> ptr;
[[vk::binding(1, 0)]]
StructuredBuffer<uint> src;
[[vk::binding(2, 0)]]
StructuredBuffer<df> blocks;
[[vk::binding(3, 0)]]
StructuredBuffer<df> weight;
[[vk::binding(4, 0)]]
RWStructuredBuffer<df> values;
[[vk::binding(5, 0)]]
ConstantBuffer<GatherParams> params;

df df_make(float h, float l) {
  df r;
  r.hi = h;
  r.lo = l;
  return r;
}

df df_f(float a) {
  return df_make(a, 0.0f);
}

df two_sum(float a, float b) {
  precise float s = (a + b);
  precise float bb = (s - a);
  precise float ea = (a - (s - bb));
  precise float eb = (b - bb);
  precise float e = (ea + eb);
  return df_make(s, e);
}

df quick_two_sum(float a, float b) {
  precise float s = (a + b);
  precise float e = (b - (s - a));
  return df_make(s, e);
}

df two_prod(float a, float b) {
  precise float p = (a * b);
  precise float e = fma(a, b, (-p));
  return df_make(p, e);
}

df df_add(df x, df y) {
  df s = two_sum(x.hi, y.hi);
  df t = two_sum(x.lo, y.lo);
  precise float sl = (s.lo + t.hi);
  df u = quick_two_sum(s.hi, sl);
  precise float ul = (u.lo + t.lo);
  return quick_two_sum(u.hi, ul);
}

df df_neg(df x) {
  return df_make((-x.hi), (-x.lo));
}

df df_sub(df x, df y) {
  return df_add(x, df_neg(y));
}

df df_mul(df x, df y) {
  df p = two_prod(x.hi, y.hi);
  precise float c = (x.hi * y.lo);
  precise float d = (x.lo * y.hi);
  precise float cd = (c + d);
  precise float pl = (p.lo + cd);
  return quick_two_sum(p.hi, pl);
}

df df_mulf(df x, float c) {
  df p = two_prod(x.hi, c);
  precise float d = (x.lo * c);
  precise float pl = (p.lo + d);
  return quick_two_sum(p.hi, pl);
}

df df_div(df x, df y) {
  float q1 = (x.hi / y.hi);
  df r1 = df_sub(x, df_mulf(y, q1));
  float q2 = (r1.hi / y.hi);
  df r2 = df_sub(r1, df_mulf(y, q2));
  float q3 = (r2.hi / y.hi);
  df q = quick_two_sum(q1, q2);
  return df_add(q, df_f(q3));
}

df df_sqrt(df x) {
  if ((x.hi <= 0.0f)) {
    return df_make(0.0f, 0.0f);
  }
  float r = sqrt(x.hi);
  df rr = two_prod(r, r);
  df d = df_sub(x, rr);
  precise float dd = (d.hi + d.lo);
  float corr = (dd / (2.0f * r));
  return quick_two_sum(r, corr);
}

float df_to_f(df x) {
  return (x.hi + x.lo);
}

bool df_lt(df x, df y) {
  return ((x.hi < y.hi) || ((x.hi == y.hi) && (x.lo < y.lo)));
}

df df_abs(df x) {
  if (df_lt(x, df_make(0.0f, 0.0f))) {
    return df_neg(x);
  }
  return x;
}

[shader(\"compute\")] [numthreads(64, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
  uint i = tid.x;
  if ((i >= params.nnz)) {
    return;
  }
  df acc = df_make(0.0f, 0.0f);
  for (uint j = ptr[i]; j < ptr[(i + 1u)]; ++j) {
    uint s = src[j];
    uint h = (s / 144u);
    acc = df_add(acc, df_mul(weight[h], blocks[s]));
  }
  values[i] = acc;
}"

example : shader.entryPointName = "main" := by native_decide
example : LeanSlang.emit shader = expected := by native_decide

end Fit.SlangCodegen.CsrGatherDf32
