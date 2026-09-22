[[vk::binding(0, 0)]]
StructuredBuffer<uint> springP1Idx;
[[vk::binding(1, 0)]]
StructuredBuffer<uint> springP2Idx;
[[vk::binding(2, 0)]]
StructuredBuffer<float3> v_g;
[[vk::binding(3, 0)]]
StructuredBuffer<float> v_H;
[[vk::binding(4, 0)]]
RWStructuredBuffer<float3> v_springGradA;
[[vk::binding(5, 0)]]
RWStructuredBuffer<float> v_springHess;

[shader("compute")] [numthreads(64, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
  uint c = tid.x;
  uint p1 = springP1Idx[c];
  uint p2 = springP2Idx[c];
  uint hb1 = (6u * p1);
  uint hb2 = (6u * p2);
  uint hbc = (6u * c);
  float3 vg1 = v_g[p1];
  float3 vg2 = v_g[p2];
  v_springGradA[c] = float3((vg1.x - vg2.x), (vg1.y - vg2.y), (vg1.z - vg2.z));
  v_springHess[hbc] = (v_H[hb1] + v_H[hb2]);
  v_springHess[(hbc + 1u)] = (v_H[(hb1 + 1u)] + v_H[(hb2 + 1u)]);
  v_springHess[(hbc + 2u)] = (v_H[(hb1 + 2u)] + v_H[(hb2 + 2u)]);
  v_springHess[(hbc + 3u)] = (v_H[(hb1 + 3u)] + v_H[(hb2 + 3u)]);
  v_springHess[(hbc + 4u)] = (v_H[(hb1 + 4u)] + v_H[(hb2 + 4u)]);
  v_springHess[(hbc + 5u)] = (v_H[(hb1 + 5u)] + v_H[(hb2 + 5u)]);
}
