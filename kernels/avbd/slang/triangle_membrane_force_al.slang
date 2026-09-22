[[vk::binding(0, 0)]]
StructuredBuffer<float3> positions;
[[vk::binding(1, 0)]]
StructuredBuffer<uint> idx;
[[vk::binding(2, 0)]]
StructuredBuffer<float> stiffness;
[[vk::binding(3, 0)]]
StructuredBuffer<float3> lambda0;
[[vk::binding(4, 0)]]
StructuredBuffer<float3> lambda1;
[[vk::binding(5, 0)]]
RWStructuredBuffer<float3> grad;
[[vk::binding(6, 0)]]
RWStructuredBuffer<float> hessScalar;
[[vk::binding(7, 0)]]
StructuredBuffer<float> inv_deltaUV;

[shader("compute")] [numthreads(64, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
  uint c = tid.x;
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
  float k = stiffness[c];
  float3 l0 = lambda0[c];
  float3 l1 = lambda1[c];
  float te0x = ((k * e0x) + l0.x);
  float te0y = ((k * e0y) + l0.y);
  float te0z = ((k * e0z) + l0.z);
  float te1x = ((k * e1rx) + l1.x);
  float te1y = ((k * e1ry) + l1.y);
  float te1z = ((k * e1rz) + l1.z);
  float w0a = (m00 + m10);
  float w0b = (m01 + m11);
  uint gb = (c * 3u);
  grad[gb] = float3((0.000000 - ((te0x * w0a) + (te1x * w0b))), (0.000000 - ((te0y * w0a) + (te1y * w0b))), (0.000000 - ((te0z * w0a) + (te1z * w0b))));
  grad[(gb + 1u)] = float3(((te0x * m00) + (te1x * m01)), ((te0y * m00) + (te1y * m01)), ((te0z * m00) + (te1z * m01)));
  grad[(gb + 2u)] = float3(((te0x * m10) + (te1x * m11)), ((te0y * m10) + (te1y * m11)), ((te0z * m10) + (te1z * m11)));
  hessScalar[gb] = (k * ((w0a * w0a) + (w0b * w0b)));
  hessScalar[(gb + 1u)] = (k * ((m00 * m00) + (m01 * m01)));
  hessScalar[(gb + 2u)] = (k * ((m10 * m10) + (m11 * m11)));
}
