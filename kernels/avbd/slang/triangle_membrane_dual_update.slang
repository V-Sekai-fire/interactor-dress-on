struct TriangleMembraneDualUpdateParams {
  float beta;
  float penaltyMax;
  uint count;
};

[[vk::binding(0, 0)]]
StructuredBuffer<float3> positions;
[[vk::binding(1, 0)]]
StructuredBuffer<uint> idx;
[[vk::binding(2, 0)]]
RWStructuredBuffer<float> gamma;
[[vk::binding(3, 0)]]
RWStructuredBuffer<float3> lambda0;
[[vk::binding(4, 0)]]
RWStructuredBuffer<float3> lambda1;
[[vk::binding(5, 0)]]
StructuredBuffer<float> inv_deltaUV;
[[vk::binding(6, 0)]]
ConstantBuffer<TriangleMembraneDualUpdateParams> params;

[shader("compute")] [numthreads(64, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
  uint c = tid.x;
  if ((c >= params.count)) {
    return;
  }
  uint base = (c * 3u);
  uint i0 = idx[base];
  uint i1 = idx[(base + 1u)];
  uint i2 = idx[(base + 2u)];
  float3 p0 = positions[i0];
  float3 p1 = positions[i1];
  float3 p2 = positions[i2];
  uint iub = (c * 4u);
  float m00 = inv_deltaUV[iub];
  float m01 = inv_deltaUV[(iub + 1u)];
  float m10 = inv_deltaUV[(iub + 2u)];
  float m11 = inv_deltaUV[(iub + 3u)];
  float ex0 = (p1.x - p0.x);
  float ey0 = (p1.y - p0.y);
  float ez0 = (p1.z - p0.z);
  float ex1 = (p2.x - p0.x);
  float ey1 = (p2.y - p0.y);
  float ez1 = (p2.z - p0.z);
  float f0x = ((ex0 * m00) + (ex1 * m10));
  float f0y = ((ey0 * m00) + (ey1 * m10));
  float f0z = ((ez0 * m00) + (ez1 * m10));
  float f1x = ((ex0 * m01) + (ex1 * m11));
  float f1y = ((ey0 * m01) + (ey1 * m11));
  float f1z = ((ez0 * m01) + (ez1 * m11));
  float a = sqrt((((f0x * f0x) + (f0y * f0y)) + (f0z * f0z)));
  float e1x = (f0x / a);
  float e1y = (f0y / a);
  float e1z = (f0z / a);
  float b = (((e1x * f1x) + (e1y * f1y)) + (e1z * f1z));
  float tx = (f1x - (b * e1x));
  float ty = (f1y - (b * e1y));
  float tz = (f1z - (b * e1z));
  float d = sqrt((((tx * tx) + (ty * ty)) + (tz * tz)));
  float e2x = (tx / d);
  float e2y = (ty / d);
  float e2z = (tz / d);
  float xv = (a + d);
  float yv = (0.000000 - b);
  float rn = sqrt(((xv * xv) + (yv * yv)));
  float r00 = (xv / rn);
  float r01 = (b / rn);
  float r10 = (yv / rn);
  float r11 = (xv / rn);
  float n0x = ((e1x * r00) + (e2x * r10));
  float n0y = ((e1y * r00) + (e2y * r10));
  float n0z = ((e1z * r00) + (e2z * r10));
  float n1x = ((e1x * r01) + (e2x * r11));
  float n1y = ((e1y * r01) + (e2y * r11));
  float n1z = ((e1z * r01) + (e2z * r11));
  float e0x = (f0x - n0x);
  float e0y = (f0y - n0y);
  float e0z = (f0z - n0z);
  float e1rx = (f1x - n1x);
  float e1ry = (f1y - n1y);
  float e1rz = (f1z - n1z);
  float g = gamma[c];
  float3 l0 = lambda0[c];
  float3 l1 = lambda1[c];
  lambda0[c] = float3((l0.x + (g * e0x)), (l0.y + (g * e0y)), (l0.z + (g * e0z)));
  lambda1[c] = float3((l1.x + (g * e1rx)), (l1.y + (g * e1ry)), (l1.z + (g * e1rz)));
  float Cmag = sqrt(((((((e0x * e0x) + (e0y * e0y)) + (e0z * e0z)) + (e1rx * e1rx)) + (e1ry * e1ry)) + (e1rz * e1rz)));
  gamma[c] = min((g + (params.beta * Cmag)), params.penaltyMax);
}
