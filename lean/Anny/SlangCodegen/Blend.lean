import Drape.SlangCodegen.Dsl

/-!
# `Anny.SlangCodegen.Blend` — the blendshape sum, one thread per vertex

    vanny[p] = templ[p] + Σ_c coeffs[c] · blend[c·P + p]

`Sinew.SlangCodegen.Pheno.blendshape` on flat float buffers and the
drape kernels' layout (a params block at binding 0, so a zero-padded
buffer does not change the counts). `coeffs` are ANNY's blendshape
weights, which the phenotype's piecewise-linear interpolation gives.

Bindings (set 0):

  0  ConstantBuffer<AnnyBlendParams> { uint P; uint NB; }
  1  StructuredBuffer<float>   templ   (P·3)
  2  StructuredBuffer<float>   blend   (NB·P·3)
  3  StructuredBuffer<float>   coeffs  (NB)
  4  RWStructuredBuffer<float> vanny   (P·3)
-/

namespace Anny.SlangCodegen.Blend

open LeanSlang
open Drape.SlangCodegen.Dsl

def shader : SlangShaderModule :=
  { structs := [ { name := "AnnyBlendParams", fields := [fld "P" uT, fld "NB" uT] } ]
  , globals :=
      [ paramsCB "AnnyBlendParams", roF "templ" 1, roF "blend" 2, roF "coeffs" 3, rwF "vanny" 4 ]
  , functions :=
      [ entry 64 [dtid]
          [ let_ uT "i" (.member (v "tid") "x")
          , if_ (ge (v "i") (p "P")) [ ret ]
          , let_ fT "x" (at_ "templ" (v "i" * u 3))
          , let_ fT "y" (at_ "templ" (v "i" * u 3 + u 1))
          , let_ fT "z" (at_ "templ" (v "i" * u 3 + u 2))
          , for_ "c" (u 0) (p "NB")
              [ let_ fT "w" (at_ "coeffs" (v "c"))
              , let_ uT "o" ((v "c" * p "P" + v "i") * u 3)
              , setv "x" (v "x" + v "w" * at_ "blend" (v "o"))
              , setv "y" (v "y" + v "w" * at_ "blend" (v "o" + u 1))
              , setv "z" (v "z" + v "w" * at_ "blend" (v "o" + u 2)) ]
          , setAt "vanny" (v "i" * u 3) (v "x")
          , setAt "vanny" (v "i" * u 3 + u 1) (v "y")
          , setAt "vanny" (v "i" * u 3 + u 2) (v "z") ] ] }

-- BEGIN PIN
def expected : String :=
"struct AnnyBlendParams {
  uint P;
  uint NB;
};

[[vk::binding(0, 0)]]
ConstantBuffer<AnnyBlendParams> params;
[[vk::binding(1, 0)]]
StructuredBuffer<float> templ;
[[vk::binding(2, 0)]]
StructuredBuffer<float> blend;
[[vk::binding(3, 0)]]
StructuredBuffer<float> coeffs;
[[vk::binding(4, 0)]]
RWStructuredBuffer<float> vanny;

[shader(\"compute\")] [numthreads(64, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
  uint i = tid.x;
  if ((i >= params.P)) {
    return;
  }
  float x = templ[(i * 3u)];
  float y = templ[((i * 3u) + 1u)];
  float z = templ[((i * 3u) + 2u)];
  for (uint c = 0u; c < params.NB; ++c) {
    float w = coeffs[c];
    uint o = (((c * params.P) + i) * 3u);
    x = (x + (w * blend[o]));
    y = (y + (w * blend[(o + 1u)]));
    z = (z + (w * blend[(o + 2u)]));
  }
  vanny[(i * 3u)] = x;
  vanny[((i * 3u) + 1u)] = y;
  vanny[((i * 3u) + 2u)] = z;
}"

example : LeanSlang.emit shader = expected := by native_decide
example : shader.entryPointName = "main" := by native_decide
-- END PIN

end Anny.SlangCodegen.Blend
