struct VbdSolveApplyParams {
  uint colorOffset;
  uint count;
};

[[vk::binding(0, 0)]]
StructuredBuffer<float3> gScratch;
[[vk::binding(1, 0)]]
StructuredBuffer<float> hScratch;
[[vk::binding(2, 0)]]
RWStructuredBuffer<float3> positions;
[[vk::binding(3, 0)]]
StructuredBuffer<uint> vertPerm;
[[vk::binding(4, 0)]]
ConstantBuffer<VbdSolveApplyParams> params;

[shader("compute")] [numthreads(64, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
  uint lane = tid.x;
  if ((lane >= params.count)) {
    return;
  }
  uint v = vertPerm[(lane + params.colorOffset)];
  uint hb = (6u * v);
  float3 g = gScratch[v];
  float Hxx = hScratch[hb];
  float Hxy = hScratch[(hb + 1u)];
  float Hxz = hScratch[(hb + 2u)];
  float Hyy = hScratch[(hb + 3u)];
  float Hyz = hScratch[(hb + 4u)];
  float Hzz = hScratch[(hb + 5u)];
  float a00 = ((Hyy * Hzz) - (Hyz * Hyz));
  float a11 = ((Hxx * Hzz) - (Hxz * Hxz));
  float a22 = ((Hxx * Hyy) - (Hxy * Hxy));
  float a01 = ((Hxz * Hyz) - (Hxy * Hzz));
  float a02 = ((Hxy * Hyz) - (Hxz * Hyy));
  float a12 = ((Hxy * Hxz) - (Hxx * Hyz));
  float det = (((Hxx * a00) + (Hxy * a01)) + (Hxz * a02));
  float invDet = (1.000000 / det);
  float gx = g.x;
  float gy = g.y;
  float gz = g.z;
  float dx = ((0.000000 - invDet) * (((a00 * gx) + (a01 * gy)) + (a02 * gz)));
  float dy = ((0.000000 - invDet) * (((a01 * gx) + (a11 * gy)) + (a12 * gz)));
  float dz = ((0.000000 - invDet) * (((a02 * gx) + (a12 * gy)) + (a22 * gz)));
  float3 p = positions[v];
  positions[v] = float3((p.x + dx), (p.y + dy), (p.z + dz));
}
