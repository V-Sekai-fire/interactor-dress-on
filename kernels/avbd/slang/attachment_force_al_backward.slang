[[vk::binding(0, 0)]]
StructuredBuffer<float3> positions;
[[vk::binding(1, 0)]]
StructuredBuffer<uint> vertIdx;
[[vk::binding(2, 0)]]
StructuredBuffer<float3> fixedPos;
[[vk::binding(3, 0)]]
StructuredBuffer<float> stiffness;
[[vk::binding(4, 0)]]
StructuredBuffer<float3> v_gradV;
[[vk::binding(5, 0)]]
StructuredBuffer<float> v_hessScalar;
[[vk::binding(6, 0)]]
RWStructuredBuffer<float3> v_positions;
[[vk::binding(7, 0)]]
RWStructuredBuffer<float3> v_fixedPos;
[[vk::binding(8, 0)]]
RWStructuredBuffer<float3> v_lambda;
[[vk::binding(9, 0)]]
RWStructuredBuffer<float> v_stiffness;

[shader("compute")] [numthreads(64, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
  uint c = tid.x;
  uint v = vertIdx[c];
  float3 p = positions[v];
  float3 fp = fixedPos[c];
  float k = stiffness[c];
  float3 vG = v_gradV[c];
  float vH = v_hessScalar[c];
  v_positions[v] = float3((k * vG.x), (k * vG.y), (k * vG.z));
  v_fixedPos[c] = float3(((0.000000 - k) * vG.x), ((0.000000 - k) * vG.y), ((0.000000 - k) * vG.z));
  v_lambda[c] = float3(vG.x, vG.y, vG.z);
  float dotCv = ((((p.x - fp.x) * vG.x) + ((p.y - fp.y) * vG.y)) + ((p.z - fp.z) * vG.z));
  v_stiffness[c] = (dotCv + vH);
}
