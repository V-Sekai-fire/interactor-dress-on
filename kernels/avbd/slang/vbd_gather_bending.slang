struct VbdGatherBendingParams {
  uint colorOffset;
  uint count;
};

[[vk::binding(0, 0)]]
StructuredBuffer<float3> bendGrad;
[[vk::binding(1, 0)]]
StructuredBuffer<float> bendHessScalar;
[[vk::binding(2, 0)]]
StructuredBuffer<uint> vertBendOffset;
[[vk::binding(3, 0)]]
StructuredBuffer<uint> vertBendIdx;
[[vk::binding(4, 0)]]
StructuredBuffer<uint> vertBendRole;
[[vk::binding(5, 0)]]
RWStructuredBuffer<float3> gScratch;
[[vk::binding(6, 0)]]
RWStructuredBuffer<float> hScratch;
[[vk::binding(7, 0)]]
StructuredBuffer<uint> vertPerm;
[[vk::binding(8, 0)]]
ConstantBuffer<VbdGatherBendingParams> params;

[shader("compute")] [numthreads(64, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
  uint lane = tid.x;
  if ((lane >= params.count)) {
    return;
  }
  uint v = vertPerm[(lane + params.colorOffset)];
  uint sStart = vertBendOffset[v];
  uint sEnd = vertBendOffset[(v + 1u)];
  uint hb = (6u * v);
  float3 g = gScratch[v];
  float gx = g.x;
  float gy = g.y;
  float gz = g.z;
  float Hxx = hScratch[hb];
  float Hyy = hScratch[(hb + 3u)];
  float Hzz = hScratch[(hb + 5u)];
  for (uint k = sStart; k < sEnd; ++k) {
    uint c = vertBendIdx[k];
    uint r = vertBendRole[k];
    uint gi = ((4u * c) + r);
    float3 ga = bendGrad[gi];
    float hs = bendHessScalar[gi];
    gx = (gx + ga.x);
    gy = (gy + ga.y);
    gz = (gz + ga.z);
    Hxx = (Hxx + hs);
    Hyy = (Hyy + hs);
    Hzz = (Hzz + hs);
  }
  gScratch[v] = float3(gx, gy, gz);
  hScratch[hb] = Hxx;
  hScratch[(hb + 3u)] = Hyy;
  hScratch[(hb + 5u)] = Hzz;
}
