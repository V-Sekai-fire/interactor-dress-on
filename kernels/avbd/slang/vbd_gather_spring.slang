struct VbdGatherSpringParams {
  uint colorOffset;
  uint count;
};

[[vk::binding(0, 0)]]
StructuredBuffer<float3> springGradA;
[[vk::binding(1, 0)]]
StructuredBuffer<float> springHess;
[[vk::binding(2, 0)]]
StructuredBuffer<uint> vertSpringOffset;
[[vk::binding(3, 0)]]
StructuredBuffer<uint> vertSpringIdx;
[[vk::binding(4, 0)]]
StructuredBuffer<uint> vertSpringRole;
[[vk::binding(5, 0)]]
RWStructuredBuffer<float3> gScratch;
[[vk::binding(6, 0)]]
RWStructuredBuffer<float> hScratch;
[[vk::binding(7, 0)]]
StructuredBuffer<uint> vertPerm;
[[vk::binding(8, 0)]]
ConstantBuffer<VbdGatherSpringParams> params;

[shader("compute")] [numthreads(64, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
  uint lane = tid.x;
  if ((lane >= params.count)) {
    return;
  }
  uint v = vertPerm[(lane + params.colorOffset)];
  uint sStart = vertSpringOffset[v];
  uint sEnd = vertSpringOffset[(v + 1u)];
  uint hb = (6u * v);
  float3 g = gScratch[v];
  float gx = g.x;
  float gy = g.y;
  float gz = g.z;
  float Hxx = hScratch[hb];
  float Hxy = hScratch[(hb + 1u)];
  float Hxz = hScratch[(hb + 2u)];
  float Hyy = hScratch[(hb + 3u)];
  float Hyz = hScratch[(hb + 4u)];
  float Hzz = hScratch[(hb + 5u)];
  for (uint k = sStart; k < sEnd; ++k) {
    uint c = vertSpringIdx[k];
    uint r = vertSpringRole[k];
    float sign = (1.000000 - (2.000000 * float(r)));
    float3 ga = springGradA[c];
    gx = (gx + (sign * ga.x));
    gy = (gy + (sign * ga.y));
    gz = (gz + (sign * ga.z));
    uint shb = (6u * c);
    Hxx = (Hxx + springHess[shb]);
    Hxy = (Hxy + springHess[(shb + 1u)]);
    Hxz = (Hxz + springHess[(shb + 2u)]);
    Hyy = (Hyy + springHess[(shb + 3u)]);
    Hyz = (Hyz + springHess[(shb + 4u)]);
    Hzz = (Hzz + springHess[(shb + 5u)]);
  }
  gScratch[v] = float3(gx, gy, gz);
  hScratch[hb] = Hxx;
  hScratch[(hb + 1u)] = Hxy;
  hScratch[(hb + 2u)] = Hxz;
  hScratch[(hb + 3u)] = Hyy;
  hScratch[(hb + 4u)] = Hyz;
  hScratch[(hb + 5u)] = Hzz;
}
