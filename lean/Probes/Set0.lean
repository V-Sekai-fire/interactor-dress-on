import LeanSlang

/-!
# `Probes.Set0` — Gate 0F: one set-0 layout shared by several pipelines

Three kernels with the fixed set-0 layout Cut 3 (ggml-rd) plans for every
op, so one uniform set can be built once and bound under any of them:

  0  ConstantBuffer<ProbeParams> { uint n; float alpha; }
  1  StructuredBuffer<float>    src0
  2  StructuredBuffer<float>    src1
  3  StructuredBuffer<float>    src2
  4  RWStructuredBuffer<float>  dst

* `probe_add`   dst[i] = src0[i] + src1[i]          (src2 unused)
* `probe_scale` dst[i] = src0[i] * alpha            (src1, src2 unused)
* `probe_acc`   dst[i] = dst[i] + src0[i]           (src1, src2 unused;
  in place through the one read-write binding)

The unused sources are the point: slangc drops unused parameters unless
`-preserve-params` is given, and then the layouts differ and a set built
for one pipeline is not valid under another. `kernels/probes/gen.sh`
compiles each kernel both ways (the stripped copy is the control).

`probe_add` with src0 and dst the same buffer is the aliased in-place shape
(read-only at b1, read-write at b4); `probe_acc` is the same arithmetic with
the buffer bound once. Gate 0F measures both across barriers.
-/

namespace Probes.Set0

open LeanSlang

def params : SlangStructDecl :=
  { name := "ProbeParams"
  , fields :=
      [ ⟨"n",     .scalar .uint,  Semantic.none, none, none, .qIn⟩
      , ⟨"alpha", .scalar .float, Semantic.none, none, none, .qIn⟩ ] }

def layout : List SlangBinding :=
  [ ⟨"params", .const "ProbeParams",   Semantic.none, some 0, some 0, .qIn⟩
  , ⟨"src0",   .roBuf (.scalar .float), Semantic.none, some 1, some 0, .qIn⟩
  , ⟨"src1",   .roBuf (.scalar .float), Semantic.none, some 2, some 0, .qIn⟩
  , ⟨"src2",   .roBuf (.scalar .float), Semantic.none, some 3, some 0, .qIn⟩
  , ⟨"dst",    .rwBuf (.scalar .float), Semantic.none, some 4, some 0, .qIn⟩ ]

/-- One thread per element, bounded by `params.n`, then `dst[i] = rhs`. -/
def kernel (rhs : SlangExpr) : SlangShaderModule :=
  { structs := [params]
  , globals := layout
  , functions := [{
      attrs  := [.shaderCompute, .numthreads 64 1 1]
      name   := "main"
      params := [⟨"tid", .vec .uint 3, .svDispatchThreadId, none, none, .qIn⟩]
      body   :=
        [ .declInit (.scalar .uint) "i" (.member (.var "tid") "x")
        , .ifNoElse (.bin ">=" (.var "i") (.member (.var "params") "n"))
            [ .ret none ]
        , .assign (.index (.var "dst") (.var "i")) rhs
        ] }] }

def add : SlangShaderModule :=
  kernel (.bin "+" (.index (.var "src0") (.var "i")) (.index (.var "src1") (.var "i")))

def scale : SlangShaderModule :=
  kernel (.bin "*" (.index (.var "src0") (.var "i")) (.member (.var "params") "alpha"))

def acc : SlangShaderModule :=
  kernel (.bin "+" (.index (.var "dst") (.var "i")) (.index (.var "src0") (.var "i")))

def header : String :=
"struct ProbeParams {
  uint n;
  float alpha;
};

[[vk::binding(0, 0)]]
ConstantBuffer<ProbeParams> params;
[[vk::binding(1, 0)]]
StructuredBuffer<float> src0;
[[vk::binding(2, 0)]]
StructuredBuffer<float> src1;
[[vk::binding(3, 0)]]
StructuredBuffer<float> src2;
[[vk::binding(4, 0)]]
RWStructuredBuffer<float> dst;

[shader(\"compute\")] [numthreads(64, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
  uint i = tid.x;
  if ((i >= params.n)) {
    return;
  }
"

example : LeanSlang.emit add   = header ++ "  dst[i] = (src0[i] + src1[i]);\n}" := by native_decide
example : LeanSlang.emit scale = header ++ "  dst[i] = (src0[i] * params.alpha);\n}" := by native_decide
example : LeanSlang.emit acc   = header ++ "  dst[i] = (dst[i] + src0[i]);\n}" := by native_decide

end Probes.Set0
