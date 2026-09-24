import Drape.SlangCodegen.Dsl

/-!
# `Anny.SlangCodegen.CsrGemv3` — `y (+)= M · x` over 3-vectors, M in CSR

One thread per row i:

    y[i] = (accumulate ? y[i] : 0) + Σ_{rowptr[i] ≤ k < rowptr[i+1]} val[k] · x[col[k]]

where `x[·]` and `y[·]` are 3-vectors. Forward, M is ANNY's sparse joint
regressor and this is `Sinew.SlangCodegen.Pheno.rbf` (`jpos = M · vanny`,
accumulate 0). Backward, the host passes Mᵀ in CSR (M's CSC) and this is
the vector-Jacobian product `dvanny += Mᵀ · djpos` (accumulate 1, onto
`anny_lbs_backward_bind`'s output): a gather per vertex, no atomics, and
every sum in the same order on the CPU and the GPU.

Bindings (set 0):

  0  ConstantBuffer<AnnyCsrParams> { uint rows; uint accumulate; }
  1  StructuredBuffer<uint>    rowptr  (rows + 1)
  2  StructuredBuffer<uint>    col     (nnz)
  3  StructuredBuffer<float>   val     (nnz)
  4  StructuredBuffer<float>   x       (cols·3)
  5  RWStructuredBuffer<float> y       (rows·3)
-/

namespace Anny.SlangCodegen.CsrGemv3

open LeanSlang
open Drape.SlangCodegen.Dsl

def shader : SlangShaderModule :=
  { structs := [ { name := "AnnyCsrParams", fields := [fld "rows" uT, fld "accumulate" uT] } ]
  , globals :=
      [ paramsCB "AnnyCsrParams", roU "rowptr" 1, roU "col" 2, roF "val" 3, roF "x" 4, rwF "y" 5 ]
  , functions :=
      [ entry 64 [dtid]
          [ let_ uT "i" (.member (v "tid") "x")
          , if_ (ge (v "i") (p "rows")) [ ret ]
          , let_ bT "acc" (ne (p "accumulate") (u 0))
          , let_ fT "a" (sel (v "acc") (at_ "y" (v "i" * u 3)) (fl 0.0))
          , let_ fT "b" (sel (v "acc") (at_ "y" (v "i" * u 3 + u 1)) (fl 0.0))
          , let_ fT "c" (sel (v "acc") (at_ "y" (v "i" * u 3 + u 2)) (fl 0.0))
          , for_ "k" (at_ "rowptr" (v "i")) (at_ "rowptr" (v "i" + u 1))
              [ let_ fT "w" (at_ "val" (v "k"))
              , let_ uT "o" (at_ "col" (v "k") * u 3)
              , setv "a" (v "a" + v "w" * at_ "x" (v "o"))
              , setv "b" (v "b" + v "w" * at_ "x" (v "o" + u 1))
              , setv "c" (v "c" + v "w" * at_ "x" (v "o" + u 2)) ]
          , setAt "y" (v "i" * u 3) (v "a")
          , setAt "y" (v "i" * u 3 + u 1) (v "b")
          , setAt "y" (v "i" * u 3 + u 2) (v "c") ] ] }

-- BEGIN PIN
def expected : String :=
"struct AnnyCsrParams {
  uint rows;
  uint accumulate;
};

[[vk::binding(0, 0)]]
ConstantBuffer<AnnyCsrParams> params;
[[vk::binding(1, 0)]]
StructuredBuffer<uint> rowptr;
[[vk::binding(2, 0)]]
StructuredBuffer<uint> col;
[[vk::binding(3, 0)]]
StructuredBuffer<float> val;
[[vk::binding(4, 0)]]
StructuredBuffer<float> x;
[[vk::binding(5, 0)]]
RWStructuredBuffer<float> y;

[shader(\"compute\")] [numthreads(64, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
  uint i = tid.x;
  if ((i >= params.rows)) {
    return;
  }
  bool acc = (params.accumulate != 0u);
  float a = (acc ? y[(i * 3u)] : 0.000000);
  float b = (acc ? y[((i * 3u) + 1u)] : 0.000000);
  float c = (acc ? y[((i * 3u) + 2u)] : 0.000000);
  for (uint k = rowptr[i]; k < rowptr[(i + 1u)]; ++k) {
    float w = val[k];
    uint o = (col[k] * 3u);
    a = (a + (w * x[o]));
    b = (b + (w * x[(o + 1u)]));
    c = (c + (w * x[(o + 2u)]));
  }
  y[(i * 3u)] = a;
  y[((i * 3u) + 1u)] = b;
  y[((i * 3u) + 2u)] = c;
}"

example : LeanSlang.emit shader = expected := by native_decide
example : shader.entryPointName = "main" := by native_decide
-- END PIN

end Anny.SlangCodegen.CsrGemv3
