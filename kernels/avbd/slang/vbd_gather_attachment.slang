struct VbdGatherAttachmentParams {
  uint colorOffset;
  uint count;
};

[[vk::binding(0, 0)]]
StructuredBuffer<float3> attachGradV;
[[vk::binding(1, 0)]]
StructuredBuffer<float> attachHessScalar;
[[vk::binding(2, 0)]]
StructuredBuffer<uint> vertAttachOffset;
[[vk::binding(3, 0)]]
StructuredBuffer<uint> vertAttachIdx;
[[vk::binding(4, 0)]]
RWStructuredBuffer<float3> gScratch;
[[vk::binding(5, 0)]]
RWStructuredBuffer<float> hScratch;
[[vk::binding(6, 0)]]
StructuredBuffer<uint> vertPerm;
[[vk::binding(7, 0)]]
ConstantBuffer<VbdGatherAttachmentParams> params;

[shader("compute")] [numthreads(64, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
  uint lane = tid.x;
  if ((lane >= params.count)) {
    return;
  }
  uint v = vertPerm[(lane + params.colorOffset)];
  uint sStart = vertAttachOffset[v];
  uint sEnd = vertAttachOffset[(v + 1u)];
  uint hb = (6u * v);
  float3 g = gScratch[v];
  float gx = g.x;
  float gy = g.y;
  float gz = g.z;
  float Hxx = hScratch[hb];
  float Hyy = hScratch[(hb + 3u)];
  float Hzz = hScratch[(hb + 5u)];
  for (uint k = sStart; k < sEnd; ++k) {
    uint c = vertAttachIdx[k];
    float3 ga = attachGradV[c];
    float hs = attachHessScalar[c];
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
