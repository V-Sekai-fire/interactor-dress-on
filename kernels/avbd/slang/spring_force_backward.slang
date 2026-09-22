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
StructuredBuffer<float3> v_gradA;
[[vk::binding(6, 0)]]
StructuredBuffer<float> v_springHess;
[[vk::binding(7, 0)]]
RWStructuredBuffer<float3> v_p_d;
[[vk::binding(8, 0)]]
RWStructuredBuffer<float> v_restLen;
[[vk::binding(9, 0)]]
RWStructuredBuffer<float> v_stiffness;

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
  float3 vG = v_gradA[i];
  uint hbi = (6u * i);
  float vH0 = v_springHess[hbi];
  float vH1 = v_springHess[(hbi + 1u)];
  float vH2 = v_springHess[(hbi + 2u)];
  float vH3 = v_springHess[(hbi + 3u)];
  float vH4 = v_springHess[(hbi + 4u)];
  float vH5 = v_springHess[(hbi + 5u)];
  float gScale = ((k * (len - L)) / len);
  float dotVGd = (((vG.x * dx) + (vG.y * dy)) + (vG.z * dz));
  float len3 = (len * len2);
  float kLd = ((k * (L * dotVGd)) / len3);
  float len4 = (len2 * len2);
  float S = ((((((vH0 * (dx * dx)) + (vH1 * (dx * dy))) + (vH2 * (dx * dz))) + (vH3 * (dy * dy))) + (vH4 * (dy * dz))) + (vH5 * (dz * dz)));
  float Tx = (((vH0 * (2.000000 * dx)) + (vH1 * dy)) + (vH2 * dz));
  float Ty = (((vH1 * dx) + (vH3 * (2.000000 * dy))) + (vH4 * dz));
  float Tz = (((vH2 * dx) + (vH4 * dy)) + (vH5 * (2.000000 * dz)));
  float hs = ((2.000000 * (k * S)) / len4);
  float kOverLen2 = (k / len2);
  v_p_d[i] = float3((((gScale * vG.x) + (kLd * dx)) + ((kOverLen2 * Tx) - (hs * dx))), (((gScale * vG.y) + (kLd * dy)) + ((kOverLen2 * Ty) - (hs * dy))), (((gScale * vG.z) + (kLd * dz)) + ((kOverLen2 * Tz) - (hs * dz))));
  v_restLen[i] = (0.000000 - ((k * dotVGd) / len));
  v_stiffness[i] = ((((len - L) * dotVGd) / len) + (S / len2));
}
