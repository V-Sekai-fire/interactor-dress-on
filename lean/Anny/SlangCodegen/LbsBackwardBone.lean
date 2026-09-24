import Drape.SlangCodegen.Dsl
import Drape.SlangCodegen.Common

/-!
# `Anny.SlangCodegen.LbsBackwardBone` — `dbone` of `anny_lbs`

    dbone_j[r] = Σ_v w[v,j] · dverts[v][r] · [bind[v] | 1]

One thread per bone row (3·J threads, t = 3j + r), each a V-long sum of
four terms accumulated in df32 (`Drape.SlangCodegen.Common.df_acc`) in
vertex order: the same terms in the same order on the CPU and the GPU,
and ~48 mantissa bits over ANNY's 13,718 vertices.

Bindings (set 0):

  0  ConstantBuffer<AnnyLbsParams> { uint V; uint J; }
  1  StructuredBuffer<float>   bind     (V·3)
  2  StructuredBuffer<float>   weights  (V·J)
  3  StructuredBuffer<float>   dverts   (V·3)
  4  RWStructuredBuffer<float> dbone    (J·12)
-/

namespace Anny.SlangCodegen.LbsBackwardBone

open LeanSlang
open Drape.SlangCodegen.Dsl

private def acc (k : Nat) (x : E) : St :=
  do_ (call "df_acc" [v s!"h{k}", v s!"l{k}", v "g", x])

def shader : SlangShaderModule :=
  { structs := [ { name := "AnnyLbsParams", fields := [fld "V" uT, fld "J" uT] } ]
  , globals :=
      [ paramsCB "AnnyLbsParams", roF "bind" 1, roF "weights" 2, roF "dverts" 3, rwF "dbone" 4 ]
  , functions := Drape.SlangCodegen.Common.dfHelpers ++
      [ entry 64 [dtid]
          ([ let_ uT "t" (.member (v "tid") "x")
           , if_ (ge (v "t") (p "J" * u 3)) [ ret ]
           , let_ uT "j" (v "t" / u 3)
           , let_ uT "r" (v "t" % u 3) ] ++
           ([0, 1, 2, 3].flatMap fun k => [let_ fT s!"h{k}" (fl 0.0), let_ fT s!"l{k}" (fl 0.0)]) ++
           [ for_ "i" (u 0) (p "V")
               [ let_ fT "w" (at_ "weights" (v "i" * p "J" + v "j"))
               , if_ (ne (v "w") (fl 0.0))
                   [ let_ fT "g" (v "w" * at_ "dverts" (v "i" * u 3 + v "r"))
                   , acc 0 (at_ "bind" (v "i" * u 3))
                   , acc 1 (at_ "bind" (v "i" * u 3 + u 1))
                   , acc 2 (at_ "bind" (v "i" * u 3 + u 2))
                   , acc 3 (fl 1.0) ] ] ] ++
           ([0, 1, 2, 3].map fun k =>
              setAt "dbone" (v "j" * u 12 + v "r" * u 4 + u k) (v s!"h{k}" + v s!"l{k}"))) ] }

-- BEGIN PIN
def expected : String :=
"struct AnnyLbsParams {
  uint V;
  uint J;
};

[[vk::binding(0, 0)]]
ConstantBuffer<AnnyLbsParams> params;
[[vk::binding(1, 0)]]
StructuredBuffer<float> bind;
[[vk::binding(2, 0)]]
StructuredBuffer<float> weights;
[[vk::binding(3, 0)]]
StructuredBuffer<float> dverts;
[[vk::binding(4, 0)]]
RWStructuredBuffer<float> dbone;

void two_sum(float a, float b, out float hi, out float lo) {
  float h = (a + b);
  float bb = (h - a);
  float ah = (h - bb);
  float lo_a = (a - ah);
  float lo_b = (b - bb);
  hi = h;
  lo = (lo_a + lo_b);
  return;
}

void quick_two_sum(float a, float b, out float hi, out float lo) {
  float h = (a + b);
  float t = (h - a);
  hi = h;
  lo = (b - t);
  return;
}

void two_prod(float a, float b, out float hi, out float lo) {
  float h = (a * b);
  hi = h;
  lo = fma(a, b, (-h));
  return;
}

void df_add(float x_hi, float x_lo, float y_hi, float y_lo, out float z_hi, out float z_lo) {
  float sh;
  float sl;
  two_sum(x_hi, y_hi, sh, sl);
  float xy_lo = (x_lo + y_lo);
  float sl2 = (sl + xy_lo);
  quick_two_sum(sh, sl2, z_hi, z_lo);
  return;
}

void df_acc(inout float hi, inout float lo, float a, float b) {
  float p_hi;
  float p_lo;
  two_prod(a, b, p_hi, p_lo);
  float n_hi;
  float n_lo;
  df_add(hi, lo, p_hi, p_lo, n_hi, n_lo);
  hi = n_hi;
  lo = n_lo;
  return;
}

[shader(\"compute\")] [numthreads(64, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
  uint t = tid.x;
  if ((t >= (params.J * 3u))) {
    return;
  }
  uint j = (t / 3u);
  uint r = (t % 3u);
  float h0 = 0.000000;
  float l0 = 0.000000;
  float h1 = 0.000000;
  float l1 = 0.000000;
  float h2 = 0.000000;
  float l2 = 0.000000;
  float h3 = 0.000000;
  float l3 = 0.000000;
  for (uint i = 0u; i < params.V; ++i) {
    float w = weights[((i * params.J) + j)];
    if ((w != 0.000000)) {
      float g = (w * dverts[((i * 3u) + r)]);
      df_acc(h0, l0, g, bind[(i * 3u)]);
      df_acc(h1, l1, g, bind[((i * 3u) + 1u)]);
      df_acc(h2, l2, g, bind[((i * 3u) + 2u)]);
      df_acc(h3, l3, g, 1.000000);
    }
  }
  dbone[(((j * 12u) + (r * 4u)) + 0u)] = (h0 + l0);
  dbone[(((j * 12u) + (r * 4u)) + 1u)] = (h1 + l1);
  dbone[(((j * 12u) + (r * 4u)) + 2u)] = (h2 + l2);
  dbone[(((j * 12u) + (r * 4u)) + 3u)] = (h3 + l3);
}"

example : LeanSlang.emit shader = expected := by native_decide
example : shader.entryPointName = "main" := by native_decide
-- END PIN

end Anny.SlangCodegen.LbsBackwardBone
