[[vk::binding(0, 0)]]
StructuredBuffer<float3> positions;
[[vk::binding(1, 0)]]
StructuredBuffer<uint> vertIdx;
[[vk::binding(2, 0)]]
StructuredBuffer<float3> fixedPos;
[[vk::binding(3, 0)]]
StructuredBuffer<float> stiffness;
[[vk::binding(4, 0)]]
StructuredBuffer<float3> lambda;
[[vk::binding(5, 0)]]
RWStructuredBuffer<float3> gradV;
[[vk::binding(6, 0)]]
RWStructuredBuffer<float> hessScalar;

[shader("compute")] [numthreads(64, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
  uint c = tid.x;
  uint v = vertIdx[c];
  float3 p = positions[v];
  float3 fp = fixedPos[c];
  float k = stiffness[c];
  float3 lam = lambda[c];
  gradV[c] = float3(((k * (p.x - fp.x)) + lam.x), ((k * (p.y - fp.y)) + lam.y), ((k * (p.z - fp.z)) + lam.z));
  hessScalar[c] = k;
}
