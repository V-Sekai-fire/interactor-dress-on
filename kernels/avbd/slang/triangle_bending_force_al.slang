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
RWStructuredBuffer<float3> grad;
[[vk::binding(7, 0)]]
RWStructuredBuffer<float> hessScalar;

[shader("compute")] [numthreads(64, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
  uint c = tid.x;
  float n_c = nTarget[c];
  uint base = (c * 4u);
  float w0 = weight[base];
  float w1 = weight[(base + 1u)];
  float w2 = weight[(base + 2u)];
  float w3 = weight[(base + 3u)];
  float ex = 0.000000;
  float ey = 0.000000;
  float ez = 0.000000;
  float k_eff = 0.000000;
  float lx = 0.000000;
  float ly = 0.000000;
  float lz = 0.000000;
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
    k_eff = stiffness[c];
    float3 lam = lambda[c];
    lx = lam.x;
    ly = lam.y;
    lz = lam.z;
  }
  grad[base] = float3((((k_eff * w0) * ex) + (w0 * lx)), (((k_eff * w0) * ey) + (w0 * ly)), (((k_eff * w0) * ez) + (w0 * lz)));
  grad[(base + 1u)] = float3((((k_eff * w1) * ex) + (w1 * lx)), (((k_eff * w1) * ey) + (w1 * ly)), (((k_eff * w1) * ez) + (w1 * lz)));
  grad[(base + 2u)] = float3((((k_eff * w2) * ex) + (w2 * lx)), (((k_eff * w2) * ey) + (w2 * ly)), (((k_eff * w2) * ez) + (w2 * lz)));
  grad[(base + 3u)] = float3((((k_eff * w3) * ex) + (w3 * lx)), (((k_eff * w3) * ey) + (w3 * ly)), (((k_eff * w3) * ez) + (w3 * lz)));
  hessScalar[base] = (k_eff * (w0 * w0));
  hessScalar[(base + 1u)] = (k_eff * (w1 * w1));
  hessScalar[(base + 2u)] = (k_eff * (w2 * w2));
  hessScalar[(base + 3u)] = (k_eff * (w3 * w3));
}
