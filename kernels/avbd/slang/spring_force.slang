[[vk::binding(0, 0)]]
StructuredBuffer<float3> positions;
[[vk::binding(1, 0)]]
StructuredBuffer<uint> p1Idx;
[[vk::binding(2, 0)]]
StructuredBuffer<uint> p2Idx;
[[vk::binding(3, 0)]]
StructuredBuffer<float> restLen;
[[vk::binding(4, 0)]]
StructuredBuffer<float> stiffness;
[[vk::binding(5, 0)]]
RWStructuredBuffer<float3> gradA;
[[vk::binding(6, 0)]]
RWStructuredBuffer<float> hess;

[shader("compute")] [numthreads(64, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
  uint i = tid.x;
  uint a = p1Idx[i];
  uint b = p2Idx[i];
  float3 p1 = positions[a];
  float3 p2 = positions[b];
  float dx = (p1.x - p2.x);
  float dy = (p1.y - p2.y);
  float dz = (p1.z - p2.z);
  float len2 = (((dx * dx) + (dy * dy)) + (dz * dz));
  float len = sqrt(len2);
  float k = stiffness[i];
  float L = restLen[i];
  float c = (len - L);
  float gScale = ((k * c) / len);
  gradA[i] = float3((gScale * dx), (gScale * dy), (gScale * dz));
  float h = (k / len2);
  uint h0 = (6u * i);
  hess[h0] = (h * (dx * dx));
  hess[(h0 + 1u)] = (h * (dx * dy));
  hess[(h0 + 2u)] = (h * (dx * dz));
  hess[(h0 + 3u)] = (h * (dy * dy));
  hess[(h0 + 4u)] = (h * (dy * dz));
  hess[(h0 + 5u)] = (h * (dz * dz));
}
