import LeanSlang

/-!
# `Probes.HalfLoad` — Gate 0F probe 14: an f16 storage read on the GPU

    o[i] = float(w[i]) * 2

Reads `StructuredBuffer<half>` and widens to `float`, so the GPU's 16-bit
storage path is checked bit-exactly against a CPU decode over every binary16
class (interactor-dress-on `gates/0f-runtime`). It is the `emit-fp` fixture
`LeanSlang.TestFp.halfLoadShader` restated here, so the kernel's source lives
in this repo; the second `example` keeps the two from drifting apart.

Bindings (set 0):

  0  StructuredBuffer<half>     w
  1  RWStructuredBuffer<float>  o

No bound check: the probe dispatches exactly one group of 64 over 64 halves.
-/

namespace Probes.HalfLoad

open LeanSlang

def shader : SlangShaderModule :=
  { globals :=
      [ ⟨"w", .roBuf (.scalar .half),  Semantic.none, some 0, some 0, .qIn⟩
      , ⟨"o", .rwBuf (.scalar .float), Semantic.none, some 1, some 0, .qIn⟩ ]
  , functions := [{
      attrs  := [.shaderCompute, .numthreads 64 1 1]
      name   := "main"
      params := [⟨"tid", .vec .uint 3, .svDispatchThreadId, none, none, .qIn⟩]
      body   :=
        [ .declInit (.scalar .uint) "i" (.member (.var "tid") "x")
        , .assign (.index (.var "o") (.var "i"))
            (.bin "*" (.cast (.scalar .float) (.index (.var "w") (.var "i")))
                      (.litFloat 2.0))
        , .ret none ]
    }] }

def expected : String :=
"[[vk::binding(0, 0)]]
StructuredBuffer<half> w;
[[vk::binding(1, 0)]]
RWStructuredBuffer<float> o;

[shader(\"compute\")] [numthreads(64, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
  uint i = tid.x;
  o[i] = (float(w[i]) * 2.000000);
  return;
}"

example : LeanSlang.emit shader = expected := by native_decide
example : LeanSlang.emit shader = LeanSlang.emit halfLoadShader := by native_decide

end Probes.HalfLoad
