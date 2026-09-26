import Drape.SlangCodegen.Dsl

/-!
# `Anny.SlangCodegen.Lbs` — linear-blend skinning, one thread per vertex

    verts[v] = Σ_j w[v,j] · (R_j · bind[v] + t_j),   bone_j = [R_j | t_j]

`Sinew.SlangCodegen.Lbs` on flat float buffers and the drape kernels'
layout; `bone` is `anny_fk`'s output (row r of joint j at `12j + 4r`,
translation in column 3). `bind` is the shaped rest mesh, `anny_blend`'s
`vanny`. Zero weights are skipped, as in Sinew's kernel.

Bindings (set 0):

  0  ConstantBuffer<AnnyLbsParams> { uint V; uint J; }
  1  StructuredBuffer<float>   bind     (V·3)
  2  StructuredBuffer<float>   weights  (V·J)  row-major
  3  StructuredBuffer<float>   bone     (J·12)
  4  RWStructuredBuffer<float> verts    (V·3)
-/

namespace Anny.SlangCodegen.Lbs

open LeanSlang
open Drape.SlangCodegen.Dsl

/-- `bone[12j + 4r + c]`. -/
def bn (r c : Nat) : E := at_ "bone" (v "b" + u (4 * r + c))

def shader : SlangShaderModule :=
  { structs := [ { name := "AnnyLbsParams", fields := [fld "V" uT, fld "J" uT] } ]
  , globals :=
      [ paramsCB "AnnyLbsParams", roF "bind" 1, roF "weights" 2, roF "bone" 3, rwF "verts" 4 ]
  , functions :=
      [ entry 64 [dtid]
          [ let_ uT "i" (.member (v "tid") "x")
          , if_ (ge (v "i") (p "V")) [ ret ]
          , let_ fT "px" (at_ "bind" (v "i" * u 3))
          , let_ fT "py" (at_ "bind" (v "i" * u 3 + u 1))
          , let_ fT "pz" (at_ "bind" (v "i" * u 3 + u 2))
          , let_ fT "x" (fl 0.0)
          , let_ fT "y" (fl 0.0)
          , let_ fT "z" (fl 0.0)
          , for_ "j" (u 0) (p "J")
              [ let_ fT "w" (at_ "weights" (v "i" * p "J" + v "j"))
              , if_ (ne (v "w") (fl 0.0))
                  [ let_ uT "b" (v "j" * u 12)
                  , setv "x" (v "x" + v "w" * (bn 0 0 * v "px" + bn 0 1 * v "py" + bn 0 2 * v "pz" + bn 0 3))
                  , setv "y" (v "y" + v "w" * (bn 1 0 * v "px" + bn 1 1 * v "py" + bn 1 2 * v "pz" + bn 1 3))
                  , setv "z" (v "z" + v "w" * (bn 2 0 * v "px" + bn 2 1 * v "py" + bn 2 2 * v "pz" + bn 2 3)) ] ]
          , setAt "verts" (v "i" * u 3) (v "x")
          , setAt "verts" (v "i" * u 3 + u 1) (v "y")
          , setAt "verts" (v "i" * u 3 + u 2) (v "z") ] ] }

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
StructuredBuffer<float> bone;
[[vk::binding(4, 0)]]
RWStructuredBuffer<float> verts;

[shader(\"compute\")] [numthreads(64, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
  uint i = tid.x;
  if ((i >= params.V)) {
    return;
  }
  float px = bind[(i * 3u)];
  float py = bind[((i * 3u) + 1u)];
  float pz = bind[((i * 3u) + 2u)];
  float x = 0.000000;
  float y = 0.000000;
  float z = 0.000000;
  for (uint j = 0u; j < params.J; ++j) {
    float w = weights[((i * params.J) + j)];
    if ((w != 0.000000)) {
      uint b = (j * 12u);
      x = (x + (w * ((((bone[(b + 0u)] * px) + (bone[(b + 1u)] * py)) + (bone[(b + 2u)] * pz)) + bone[(b + 3u)])));
      y = (y + (w * ((((bone[(b + 4u)] * px) + (bone[(b + 5u)] * py)) + (bone[(b + 6u)] * pz)) + bone[(b + 7u)])));
      z = (z + (w * ((((bone[(b + 8u)] * px) + (bone[(b + 9u)] * py)) + (bone[(b + 10u)] * pz)) + bone[(b + 11u)])));
    }
  }
  verts[(i * 3u)] = x;
  verts[((i * 3u) + 1u)] = y;
  verts[((i * 3u) + 2u)] = z;
}"

example : LeanSlang.emit shader = expected := by native_decide
example : shader.entryPointName = "main" := by native_decide
-- END PIN

end Anny.SlangCodegen.Lbs
