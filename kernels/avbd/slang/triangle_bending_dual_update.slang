struct TriangleBendingDualUpdateParams {
  float beta;
  float penaltyMax;
  uint count;
};

[[vk::binding(0, 0)]]
StructuredBuffer<float3> positions;
[[vk::binding(1, 0)]]
StructuredBuffer<uint> idx;
[[vk::binding(2, 0)]]
StructuredBuffer<float> weight;
[[vk::binding(3, 0)]]
StructuredBuffer<float> nTarget;
[[vk::binding(4, 0)]]
RWStructuredBuffer<float> gamma;
[[vk::binding(5, 0)]]
RWStructuredBuffer<float3> lambda;
[[vk::binding(6, 0)]]
ConstantBuffer<TriangleBendingDualUpdateParams> params;

[shader("compute")] [numthreads(64, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
  uint c = tid.x;
  if ((c >= params.count)) {
    return;
  }
  float n_c = nTarget[c];
  uint base = (c * 4u);
  float w0 = weight[base];
  float w1 = weight[(base + 1u)];
  float w2 = weight[(base + 2u)];
  float w3 = weight[(base + 3u)];
  float ex = 0.000000;
  float ey = 0.000000;
  float ez = 0.000000;
  float g_eff = 0.000000;
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
    float len = sqrt((((sx * sx) + (sy * sy)) + (sz * sz)));
    float scale = (n_c / len);
    float om = (1.000000 - scale);
    ex = (sx * om);
    ey = (sy * om);
    ez = (sz * om);
    g_eff = gamma[c];
  }
  float3 lam = lambda[c];
  lambda[c] = float3((lam.x + (g_eff * ex)), (lam.y + (g_eff * ey)), (lam.z + (g_eff * ez)));
  float Cmag = sqrt((((ex * ex) + (ey * ey)) + (ez * ez)));
  gamma[c] = min((gamma[c] + (params.beta * Cmag)), params.penaltyMax);
}
