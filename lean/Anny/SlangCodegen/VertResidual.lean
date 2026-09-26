import Drape.SlangCodegen.Dsl

/-!
# `Anny.SlangCodegen.VertResidual` — `dverts = verts − target`

The cotangent of the vertex loss `L = ½ Σ |verts − target|²`, one thread
per float; L itself is `½ dverts · dverts` (the driver's `dot_reduce`).
This is the per-vertex error `AnnyInverter` minimises (the PVE).

Bindings (set 0):

  0  ConstantBuffer<AnnyResidualParams> { uint n; }   n = V·3
  1  StructuredBuffer<float>   verts   (n)
  2  StructuredBuffer<float>   target  (n)
  3  RWStructuredBuffer<float> dverts  (n)
-/

namespace Anny.SlangCodegen.VertResidual

open LeanSlang
open Drape.SlangCodegen.Dsl

def shader : SlangShaderModule :=
  { structs := [ { name := "AnnyResidualParams", fields := [fld "n" uT] } ]
  , globals := [ paramsCB "AnnyResidualParams", roF "verts" 1, roF "target" 2, rwF "dverts" 3 ]
  , functions :=
      [ entry 256 [dtid]
          [ let_ uT "i" (.member (v "tid") "x")
          , if_ (ge (v "i") (p "n")) [ ret ]
          , setAt "dverts" (v "i") (at_ "verts" (v "i") - at_ "target" (v "i")) ] ] }

-- BEGIN PIN
def expected : String :=
"struct AnnyResidualParams {
  uint n;
};

[[vk::binding(0, 0)]]
ConstantBuffer<AnnyResidualParams> params;
[[vk::binding(1, 0)]]
StructuredBuffer<float> verts;
[[vk::binding(2, 0)]]
StructuredBuffer<float> target;
[[vk::binding(3, 0)]]
RWStructuredBuffer<float> dverts;

[shader(\"compute\")] [numthreads(256, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
  uint i = tid.x;
  if ((i >= params.n)) {
    return;
  }
  dverts[i] = (verts[i] - target[i]);
}"

example : LeanSlang.emit shader = expected := by native_decide
example : shader.entryPointName = "main" := by native_decide
-- END PIN

end Anny.SlangCodegen.VertResidual
