struct GbParams {
  float3 tangent_a;
  float3 tangent_b;
  uint count;
};

[[vk::binding(0, 0)]]
ConstantBuffer<GbParams> params;
[[vk::binding(1, 0)]]
StructuredBuffer<float3> in_points;
[[vk::binding(2, 0)]]
StructuredBuffer<float> in_u;
[[vk::binding(3, 0)]]
RWStructuredBuffer<float3> out_ctrl;

[shader("compute")] [numthreads(1, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
  float3 p0 = in_points[0u];
  float3 p3 = in_points[(params.count - 1u)];
  float c00 = 0.000000;
  float c01 = 0.000000;
  float c11 = 0.000000;
  float x0 = 0.000000;
  float x1 = 0.000000;
  for (uint i = 0u; i < params.count; ++i) {
    float ui = in_u[i];
    float omu = (1.000000 - ui);
    float ka = (3.000000 * ((omu * omu) * ui));
    float kb = (3.000000 * ((ui * ui) * omu));
    float3 a1 = float3((ka * params.tangent_a.x), (ka * params.tangent_a.y), (ka * params.tangent_a.z));
    float3 a2 = float3((kb * params.tangent_b.x), (kb * params.tangent_b.y), (kb * params.tangent_b.z));
    c00 = (c00 + dot(a1, a1));
    c01 = (c01 + dot(a1, a2));
    c11 = (c11 + dot(a2, a2));
    float wA = ((omu * omu) * (1.000000 + (2.000000 * ui)));
    float wB = ((ui * ui) * (3.000000 - (2.000000 * ui)));
    float3 baseline = float3((float3((wA * p0.x), (wA * p0.y), (wA * p0.z)).x + float3((wB * p3.x), (wB * p3.y), (wB * p3.z)).x), (float3((wA * p0.x), (wA * p0.y), (wA * p0.z)).y + float3((wB * p3.x), (wB * p3.y), (wB * p3.z)).y), (float3((wA * p0.x), (wA * p0.y), (wA * p0.z)).z + float3((wB * p3.x), (wB * p3.y), (wB * p3.z)).z));
    float3 pi = in_points[i];
    float3 tmp = float3((pi.x - baseline.x), (pi.y - baseline.y), (pi.z - baseline.z));
    x0 = (x0 + dot(a1, tmp));
    x1 = (x1 + dot(a2, tmp));
  }
  float det = ((c00 * c11) - (c01 * c01));
  float det_x = ((c00 * x1) - (c01 * x0));
  float det_y = ((x0 * c11) - (x1 * c01));
  float3 chord = float3((p3.x - p0.x), (p3.y - p0.y), (p3.z - p0.z));
  float seg_length = length(chord);
  float epsilon = (0.000010 * seg_length);
  float alpha_a = 0.000000;
  float alpha_b = 0.000000;
  uint fallback = ((abs(det) < 0.000000) ? 1u : 0u);
  if ((fallback == 0u)) {
    alpha_a = (det_y / det);
    alpha_b = (det_x / det);
    if (((alpha_a < epsilon) || (alpha_b < epsilon))) {
      fallback = 1u;
    }
  }
  float3 p1 = p0;
  float3 p2 = p3;
  if ((fallback == 1u)) {
    float third = (seg_length / 3.000000);
    p1 = float3((p0.x + float3((third * params.tangent_a.x), (third * params.tangent_a.y), (third * params.tangent_a.z)).x), (p0.y + float3((third * params.tangent_a.x), (third * params.tangent_a.y), (third * params.tangent_a.z)).y), (p0.z + float3((third * params.tangent_a.x), (third * params.tangent_a.y), (third * params.tangent_a.z)).z));
    p2 = float3((p3.x + float3((third * params.tangent_b.x), (third * params.tangent_b.y), (third * params.tangent_b.z)).x), (p3.y + float3((third * params.tangent_b.x), (third * params.tangent_b.y), (third * params.tangent_b.z)).y), (p3.z + float3((third * params.tangent_b.x), (third * params.tangent_b.y), (third * params.tangent_b.z)).z));
  } else {
    p1 = float3((p0.x + float3((alpha_a * params.tangent_a.x), (alpha_a * params.tangent_a.y), (alpha_a * params.tangent_a.z)).x), (p0.y + float3((alpha_a * params.tangent_a.x), (alpha_a * params.tangent_a.y), (alpha_a * params.tangent_a.z)).y), (p0.z + float3((alpha_a * params.tangent_a.x), (alpha_a * params.tangent_a.y), (alpha_a * params.tangent_a.z)).z));
    p2 = float3((p3.x + float3((alpha_b * params.tangent_b.x), (alpha_b * params.tangent_b.y), (alpha_b * params.tangent_b.z)).x), (p3.y + float3((alpha_b * params.tangent_b.x), (alpha_b * params.tangent_b.y), (alpha_b * params.tangent_b.z)).y), (p3.z + float3((alpha_b * params.tangent_b.x), (alpha_b * params.tangent_b.y), (alpha_b * params.tangent_b.z)).z));
  }
  out_ctrl[0u] = p0;
  out_ctrl[1u] = p1;
  out_ctrl[2u] = p2;
  out_ctrl[3u] = p3;
  return;
}
