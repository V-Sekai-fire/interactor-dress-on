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
StructuredBuffer<float> inv_deltaUV;
[[vk::binding(6, 0)]]
StructuredBuffer<float3> v_grad;
[[vk::binding(7, 0)]]
StructuredBuffer<float> v_hessScalar;
[[vk::binding(8, 0)]]
RWStructuredBuffer<float3> v_p;
[[vk::binding(9, 0)]]
RWStructuredBuffer<float> v_stiffness;
[[vk::binding(10, 0)]]
RWStructuredBuffer<float3> v_lambda0;
[[vk::binding(11, 0)]]
RWStructuredBuffer<float3> v_lambda1;

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
  float rn = sqrt(((xv * xv) + (b * b)));
  float r00 = (xv / rn);
  float r01 = (b / rn);
  float r10 = (0.000000 - (b / rn));
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
  float w0a = (m00 + m10);
  float w0b = (m01 + m11);
  float k = stiffness[c];
  float3 vg0 = v_grad[base];
  float3 vg1 = v_grad[(base + 1u)];
  float3 vg2 = v_grad[(base + 2u)];
  float vh0 = v_hessScalar[base];
  float vh1 = v_hessScalar[(base + 1u)];
  float vh2 = v_hessScalar[(base + 2u)];
  float Bx = ((((0.000000 - w0a) * vg0.x) + (m00 * vg1.x)) + (m10 * vg2.x));
  float By = ((((0.000000 - w0a) * vg0.y) + (m00 * vg1.y)) + (m10 * vg2.y));
  float Bz = ((((0.000000 - w0a) * vg0.z) + (m00 * vg1.z)) + (m10 * vg2.z));
  float Cx = ((((0.000000 - w0b) * vg0.x) + (m01 * vg1.x)) + (m11 * vg2.x));
  float Cy = ((((0.000000 - w0b) * vg0.y) + (m01 * vg1.y)) + (m11 * vg2.y));
  float Cz = ((((0.000000 - w0b) * vg0.z) + (m01 * vg1.z)) + (m11 * vg2.z));
  float H = (((vh0 * ((w0a * w0a) + (w0b * w0b))) + (vh1 * ((m00 * m00) + (m01 * m01)))) + (vh2 * ((m10 * m10) + (m11 * m11))));
  float BdotE0 = (((Bx * e0x) + (By * e0y)) + (Bz * e0z));
  float CdotE1 = (((Cx * e1rx) + (Cy * e1ry)) + (Cz * e1rz));
  v_lambda0[c] = float3(Bx, By, Bz);
  v_lambda1[c] = float3(Cx, Cy, Cz);
  v_stiffness[c] = ((BdotE0 + CdotE1) + H);
  float vf0x = (k * Bx);
  float vf0y = (k * By);
  float vf0z = (k * Bz);
  float vf1x = (k * Cx);
  float vf1y = (k * Cy);
  float vf1z = (k * Cz);
  float vn0x = (0.000000 - vf0x);
  float vn0y = (0.000000 - vf0y);
  float vn0z = (0.000000 - vf0z);
  float vn1x = (0.000000 - vf1x);
  float vn1y = (0.000000 - vf1y);
  float vn1z = (0.000000 - vf1z);
  float ve1x = ((vn0x * r00) + (vn1x * r01));
  float ve1y = ((vn0y * r00) + (vn1y * r01));
  float ve1z = ((vn0z * r00) + (vn1z * r01));
  float ve2x = ((vn0x * r10) + (vn1x * r11));
  float ve2y = ((vn0y * r10) + (vn1y * r11));
  float ve2z = ((vn0z * r10) + (vn1z * r11));
  float vr00 = (((vn0x * e1x) + (vn0y * e1y)) + (vn0z * e1z));
  float vr01 = (((vn1x * e1x) + (vn1y * e1y)) + (vn1z * e1z));
  float vr10 = (((vn0x * e2x) + (vn0y * e2y)) + (vn0z * e2z));
  float vr11 = (((vn1x * e2x) + (vn1y * e2y)) + (vn1z * e2z));
  float rn3 = (rn * (rn * rn));
  float trR = (vr00 + vr11);
  float antR = (vr10 - vr01);
  float antR2 = (vr01 - vr10);
  float vxv = (((trR * (b * b)) / rn3) + ((antR * (xv * b)) / rn3));
  float vb_polar = ((0.000000 - ((trR * (xv * b)) / rn3)) + ((antR2 * (xv * xv)) / rn3));
  float va = vxv;
  float vd = vxv;
  float ve2dote2 = (((ve2x * e2x) + (ve2y * e2y)) + (ve2z * e2z));
  float vtx = (((ve2x - (ve2dote2 * e2x)) / d) + (vd * e2x));
  float vty = (((ve2y - (ve2dote2 * e2y)) / d) + (vd * e2y));
  float vtz = (((ve2z - (ve2dote2 * e2z)) / d) + (vd * e2z));
  vf1x = (vf1x + vtx);
  vf1y = (vf1y + vty);
  vf1z = (vf1z + vtz);
  ve1x = (ve1x - (b * vtx));
  ve1y = (ve1y - (b * vty));
  ve1z = (ve1z - (b * vtz));
  float vt_dot_e1 = (((vtx * e1x) + (vty * e1y)) + (vtz * e1z));
  float vb = (vb_polar + (0.000000 - vt_dot_e1));
  ve1x = (ve1x + (vb * f1x));
  ve1y = (ve1y + (vb * f1y));
  ve1z = (ve1z + (vb * f1z));
  vf1x = (vf1x + (vb * e1x));
  vf1y = (vf1y + (vb * e1y));
  vf1z = (vf1z + (vb * e1z));
  float ve1dote1 = (((ve1x * e1x) + (ve1y * e1y)) + (ve1z * e1z));
  vf0x = (vf0x + ((ve1x - (ve1dote1 * e1x)) / a));
  vf0y = (vf0y + ((ve1y - (ve1dote1 * e1y)) / a));
  vf0z = (vf0z + ((ve1z - (ve1dote1 * e1z)) / a));
  vf0x = (vf0x + (va * e1x));
  vf0y = (vf0y + (va * e1y));
  vf0z = (vf0z + (va * e1z));
  v_p[base] = float3((0.000000 - ((w0a * vf0x) + (w0b * vf1x))), (0.000000 - ((w0a * vf0y) + (w0b * vf1y))), (0.000000 - ((w0a * vf0z) + (w0b * vf1z))));
  v_p[(base + 1u)] = float3(((m00 * vf0x) + (m01 * vf1x)), ((m00 * vf0y) + (m01 * vf1y)), ((m00 * vf0z) + (m01 * vf1z)));
  v_p[(base + 2u)] = float3(((m10 * vf0x) + (m11 * vf1x)), ((m10 * vf0y) + (m11 * vf1y)), ((m10 * vf0z) + (m11 * vf1z)));
}
