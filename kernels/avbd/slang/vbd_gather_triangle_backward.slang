[[vk::binding(0, 0)]]
StructuredBuffer<uint> triIdx;
[[vk::binding(1, 0)]]
StructuredBuffer<float3> v_g;
[[vk::binding(2, 0)]]
StructuredBuffer<float> v_H;
[[vk::binding(3, 0)]]
RWStructuredBuffer<float3> v_triGrad;
[[vk::binding(4, 0)]]
RWStructuredBuffer<float> v_triHessScalar;

[shader("compute")] [numthreads(64, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
  uint slot = tid.x;
  uint v = triIdx[slot];
  uint hb = (6u * v);
  float3 vg = v_g[v];
  float vHxx = v_H[hb];
  float vHyy = v_H[(hb + 3u)];
  float vHzz = v_H[(hb + 5u)];
  v_triGrad[slot] = float3(vg.x, vg.y, vg.z);
  v_triHessScalar[slot] = ((vHxx + vHyy) + vHzz);
}
