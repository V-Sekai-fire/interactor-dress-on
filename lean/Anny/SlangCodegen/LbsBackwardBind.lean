import Drape.SlangCodegen.Dsl

/-!
# `Anny.SlangCodegen.LbsBackwardBind` — `dbind` of `anny_lbs`

    dbind[v] = Σ_j w[v,j] · R_jᵀ · dverts[v]

One thread per vertex; overwrites `dbind`, which is the start of
`dvanny` (`anny_csr_gemv3` then adds the joint regressor's share,
`Mᵀ · djpos`, and `anny_blend_backward` reads the sum).

Bindings (set 0):

  0  ConstantBuffer<AnnyLbsParams> { uint V; uint J; }
  1  StructuredBuffer<float>   weights  (V·J)
  2  StructuredBuffer<float>   bone     (J·12)
  3  StructuredBuffer<float>   dverts   (V·3)
  4  RWStructuredBuffer<float> dbind    (V·3)
-/

namespace Anny.SlangCodegen.LbsBackwardBind

open LeanSlang
open Drape.SlangCodegen.Dsl

/-- `bone[12j + 4r + c]`. -/
private def bn (r c : Nat) : E := at_ "bone" (v "b" + u (4 * r + c))

def shader : SlangShaderModule :=
  { structs := [ { name := "AnnyLbsParams", fields := [fld "V" uT, fld "J" uT] } ]
  , globals :=
      [ paramsCB "AnnyLbsParams", roF "weights" 1, roF "bone" 2, roF "dverts" 3, rwF "dbind" 4 ]
  , functions :=
      [ entry 64 [dtid]
          [ let_ uT "i" (.member (v "tid") "x")
          , if_ (ge (v "i") (p "V")) [ ret ]
          , let_ fT "gx" (at_ "dverts" (v "i" * u 3))
          , let_ fT "gy" (at_ "dverts" (v "i" * u 3 + u 1))
          , let_ fT "gz" (at_ "dverts" (v "i" * u 3 + u 2))
          , let_ fT "x" (fl 0.0)
          , let_ fT "y" (fl 0.0)
          , let_ fT "z" (fl 0.0)
          , for_ "j" (u 0) (p "J")
              [ let_ fT "w" (at_ "weights" (v "i" * p "J" + v "j"))
              , if_ (ne (v "w") (fl 0.0))
                  [ let_ uT "b" (v "j" * u 12)
                  , setv "x" (v "x" + v "w" * (bn 0 0 * v "gx" + bn 1 0 * v "gy" + bn 2 0 * v "gz"))
                  , setv "y" (v "y" + v "w" * (bn 0 1 * v "gx" + bn 1 1 * v "gy" + bn 2 1 * v "gz"))
                  , setv "z" (v "z" + v "w" * (bn 0 2 * v "gx" + bn 1 2 * v "gy" + bn 2 2 * v "gz")) ] ]
          , setAt "dbind" (v "i" * u 3) (v "x")
          , setAt "dbind" (v "i" * u 3 + u 1) (v "y")
          , setAt "dbind" (v "i" * u 3 + u 2) (v "z") ] ] }

-- BEGIN PIN
def expected : String :=
"struct AnnyLbsParams {
  uint V;
  uint J;
};

[[vk::binding(0, 0)]]
ConstantBuffer<AnnyLbsParams> params;
[[vk::binding(1, 0)]]
StructuredBuffer<float> weights;
[[vk::binding(2, 0)]]
StructuredBuffer<float> bone;
[[vk::binding(3, 0)]]
StructuredBuffer<float> dverts;
[[vk::binding(4, 0)]]
RWStructuredBuffer<float> dbind;

[shader(\"compute\")] [numthreads(64, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
  uint i = tid.x;
  if ((i >= params.V)) {
    return;
  }
  float gx = dverts[(i * 3u)];
  float gy = dverts[((i * 3u) + 1u)];
  float gz = dverts[((i * 3u) + 2u)];
  float x = 0.000000;
  float y = 0.000000;
  float z = 0.000000;
  for (uint j = 0u; j < params.J; ++j) {
    float w = weights[((i * params.J) + j)];
    if ((w != 0.000000)) {
      uint b = (j * 12u);
      x = (x + (w * (((bone[(b + 0u)] * gx) + (bone[(b + 4u)] * gy)) + (bone[(b + 8u)] * gz))));
      y = (y + (w * (((bone[(b + 1u)] * gx) + (bone[(b + 5u)] * gy)) + (bone[(b + 9u)] * gz))));
      z = (z + (w * (((bone[(b + 2u)] * gx) + (bone[(b + 6u)] * gy)) + (bone[(b + 10u)] * gz))));
    }
  }
  dbind[(i * 3u)] = x;
  dbind[((i * 3u) + 1u)] = y;
  dbind[((i * 3u) + 2u)] = z;
}"

example : LeanSlang.emit shader = expected := by native_decide
example : shader.entryPointName = "main" := by native_decide
-- END PIN

end Anny.SlangCodegen.LbsBackwardBind
