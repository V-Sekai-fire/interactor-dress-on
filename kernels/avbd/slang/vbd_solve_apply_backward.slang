[[vk::binding(0, 0)]]
StructuredBuffer<float3> v_out;
[[vk::binding(1, 0)]]
StructuredBuffer<float> hScratch;
[[vk::binding(2, 0)]]
StructuredBuffer<float3> deltaX;
[[vk::binding(3, 0)]]
RWStructuredBuffer<float3> v_g;
[[vk::binding(4, 0)]]
RWStructuredBuffer<float> v_H;

[shader("compute")] [numthreads(64, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
  uint v = tid.x;
  uint hb = (6u * v);
  float3 vo = v_out[v];
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
  float ux = vo.x;
  float uy = vo.y;
  float uz = vo.z;
  float yx = (invDet * (((a00 * ux) + (a01 * uy)) + (a02 * uz)));
  float yy = (invDet * (((a01 * ux) + (a11 * uy)) + (a12 * uz)));
  float yz = (invDet * (((a02 * ux) + (a12 * uy)) + (a22 * uz)));
  v_g[v] = float3((0.000000 - yx), (0.000000 - yy), (0.000000 - yz));
  float3 dx = deltaX[v];
  float dxx = dx.x;
  float dxy = dx.y;
  float dxz = dx.z;
  v_H[hb] = (0.000000 - (yx * dxx));
  v_H[(hb + 1u)] = (0.000000 - ((yx * dxy) + (yy * dxx)));
  v_H[(hb + 2u)] = (0.000000 - ((yx * dxz) + (yz * dxx)));
  v_H[(hb + 3u)] = (0.000000 - (yy * dxy));
  v_H[(hb + 4u)] = (0.000000 - ((yy * dxz) + (yz * dxy)));
  v_H[(hb + 5u)] = (0.000000 - (yz * dxz));
}
