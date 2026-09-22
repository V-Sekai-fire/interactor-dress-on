[[vk::binding(0, 0)]]
StructuredBuffer<uint> bendIdx;
[[vk::binding(1, 0)]]
StructuredBuffer<float3> v_g;
[[vk::binding(2, 0)]]
StructuredBuffer<float> v_H;
[[vk::binding(3, 0)]]
RWStructuredBuffer<float3> v_bendGrad;
[[vk::binding(4, 0)]]
RWStructuredBuffer<float> v_bendHessScalar;

[shader("compute")] [numthreads(64, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
  uint slot = tid.x;
  uint v = bendIdx[slot];
  uint hb = (6u * v);
  float3 vg = v_g[v];
  float vHxx = v_H[hb];
  float vHyy = v_H[(hb + 3u)];
  float vHzz = v_H[(hb + 5u)];
  v_bendGrad[slot] = float3(vg.x, vg.y, vg.z);
  v_bendHessScalar[slot] = ((vHxx + vHyy) + vHzz);
}
