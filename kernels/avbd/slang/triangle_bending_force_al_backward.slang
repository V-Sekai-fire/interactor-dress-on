[[vk::binding(0, 0)]]
StructuredBuffer<float3> positions;
[[vk::binding(1, 0)]]
StructuredBuffer<uint> idx;
[[vk::binding(2, 0)]]
StructuredBuffer<float> weight;
[[vk::binding(3, 0)]]
StructuredBuffer<float> nTarget;
[[vk::binding(4, 0)]]
StructuredBuffer<float> stiffness;
[[vk::binding(5, 0)]]
StructuredBuffer<float3> lambda;
[[vk::binding(6, 0)]]
StructuredBuffer<float3> v_grad;
[[vk::binding(7, 0)]]
StructuredBuffer<float> v_hessScalar;
[[vk::binding(8, 0)]]
RWStructuredBuffer<float3> v_p;
[[vk::binding(9, 0)]]
RWStructuredBuffer<float> v_nTarget;
[[vk::binding(10, 0)]]
RWStructuredBuffer<float> v_stiffness;
[[vk::binding(11, 0)]]
RWStructuredBuffer<float3> v_lambda;

[shader("compute")] [numthreads(64, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
  uint c = tid.x;
  uint base = (c * 4u);
  float n_c = nTarget[c];
  float w0 = weight[base];
  float w1 = weight[(base + 1u)];
  float w2 = weight[(base + 2u)];
  float w3 = weight[(base + 3u)];
  float3 vg0 = v_grad[base];
  float3 vg1 = v_grad[(base + 1u)];
  float3 vg2 = v_grad[(base + 2u)];
  float3 vg3 = v_grad[(base + 3u)];
  float vh0 = v_hessScalar[base];
  float vh1 = v_hessScalar[(base + 1u)];
  float vh2 = v_hessScalar[(base + 2u)];
  float vh3 = v_hessScalar[(base + 3u)];
  float Ax = (((w0 * vg0.x) + (w1 * vg1.x)) + ((w2 * vg2.x) + (w3 * vg3.x)));
  float Ay = (((w0 * vg0.y) + (w1 * vg1.y)) + ((w2 * vg2.y) + (w3 * vg3.y)));
  float Az = (((w0 * vg0.z) + (w1 * vg1.z)) + ((w2 * vg2.z) + (w3 * vg3.z)));
  float HW = ((((w0 * w0) * vh0) + ((w1 * w1) * vh1)) + (((w2 * w2) * vh2) + ((w3 * w3) * vh3)));
  float vpx0 = 0.000000;
  float vpy0 = 0.000000;
  float vpz0 = 0.000000;
  float vpx1 = 0.000000;
  float vpy1 = 0.000000;
  float vpz1 = 0.000000;
  float vpx2 = 0.000000;
  float vpy2 = 0.000000;
  float vpz2 = 0.000000;
  float vpx3 = 0.000000;
  float vpy3 = 0.000000;
  float vpz3 = 0.000000;
  float vlx = 0.000000;
  float vly = 0.000000;
  float vlz = 0.000000;
  float vk = 0.000000;
  float vn = 0.000000;
  if ((n_c > 0.000001)) {
    uint i0 = idx[base];
    uint i1 = idx[(base + 1u)];
    uint i2 = idx[(base + 2u)];
    uint i3 = idx[(base + 3u)];
    float3 p0 = positions[i0];
    float3 p1 = positions[i1];
    float3 p2 = positions[i2];
    float3 p3 = positions[i3];
    float sx = (((w0 * p0.x) + (w1 * p1.x)) + ((w2 * p2.x) + (w3 * p3.x)));
    float sy = (((w0 * p0.y) + (w1 * p1.y)) + ((w2 * p2.y) + (w3 * p3.y)));
    float sz = (((w0 * p0.z) + (w1 * p1.z)) + ((w2 * p2.z) + (w3 * p3.z)));
    float len2 = (((sx * sx) + (sy * sy)) + (sz * sz));
    float len = sqrt(len2);
    float scale = ((len > 0.000000) ? (n_c / len) : 1.000000);
    float om = (1.000000 - scale);
    float ex = (sx * om);
    float ey = (sy * om);
    float ez = (sz * om);
    float k = stiffness[c];
    float AdotE = (((Ax * ex) + (Ay * ey)) + (Az * ez));
    float AdotS = (((Ax * sx) + (Ay * sy)) + (Az * sz));
    float len3 = (len * len2);
    float factor = ((len > 0.000000) ? ((k * (n_c * AdotS)) / len3) : 0.000000);
    float kom = (k * om);
    float tx = ((kom * Ax) + (factor * sx));
    float ty = ((kom * Ay) + (factor * sy));
    float tz = ((kom * Az) + (factor * sz));
    vpx0 = (w0 * tx);
    vpy0 = (w0 * ty);
    vpz0 = (w0 * tz);
    vpx1 = (w1 * tx);
    vpy1 = (w1 * ty);
    vpz1 = (w1 * tz);
    vpx2 = (w2 * tx);
    vpy2 = (w2 * ty);
    vpz2 = (w2 * tz);
    vpx3 = (w3 * tx);
    vpy3 = (w3 * ty);
    vpz3 = (w3 * tz);
    vlx = Ax;
    vly = Ay;
    vlz = Az;
    vk = (AdotE + HW);
    vn = ((len > 0.000000) ? (0.000000 - ((k * AdotS) / len)) : 0.000000);
  }
  v_p[base] = float3(vpx0, vpy0, vpz0);
  v_p[(base + 1u)] = float3(vpx1, vpy1, vpz1);
  v_p[(base + 2u)] = float3(vpx2, vpy2, vpz2);
  v_p[(base + 3u)] = float3(vpx3, vpy3, vpz3);
  v_lambda[c] = float3(vlx, vly, vlz);
  v_stiffness[c] = vk;
  v_nTarget[c] = vn;
}
