struct NewtonParams {
  float3 a;
  float3 b;
  float3 c;
  float3 d;
  uint count;
};

[[vk::binding(0, 0)]]
ConstantBuffer<NewtonParams> params;
[[vk::binding(1, 0)]]
StructuredBuffer<float3> in_points;
[[vk::binding(2, 0)]]
StructuredBuffer<float> in_u;
[[vk::binding(3, 0)]]
RWStructuredBuffer<float> out_u;

[shader("compute")] [numthreads(1, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
  for (uint i = 0u; i < params.count; ++i) {
    float u = in_u[i];
    float3 pt = in_points[i];
    float omu = (1.000000 - u);
    float3 ab = float3((params.b.x - params.a.x), (params.b.y - params.a.y), (params.b.z - params.a.z));
    float3 bc = float3((params.c.x - params.b.x), (params.c.y - params.b.y), (params.c.z - params.b.z));
    float3 cd = float3((params.d.x - params.c.x), (params.d.y - params.c.y), (params.d.z - params.c.z));
    float omu2 = (omu * omu);
    float u2 = (u * u);
    float twoOmU = (2.000000 * (omu * u));
    float3 q1 = float3((3.000000 * (((omu2 * ab.x) + (twoOmU * bc.x)) + (u2 * cd.x))), (3.000000 * (((omu2 * ab.y) + (twoOmU * bc.y)) + (u2 * cd.y))), (3.000000 * (((omu2 * ab.z) + (twoOmU * bc.z)) + (u2 * cd.z))));
    float3 dd1 = float3((bc.x - ab.x), (bc.y - ab.y), (bc.z - ab.z));
    float3 dd2 = float3((cd.x - bc.x), (cd.y - bc.y), (cd.z - bc.z));
    float3 q2 = float3((6.000000 * ((omu * dd1.x) + (u * dd2.x))), (6.000000 * ((omu * dd1.y) + (u * dd2.y))), (6.000000 * ((omu * dd1.z) + (u * dd2.z))));
    float3 qab = lerp(params.a, params.b, u);
    float3 qbc = lerp(params.b, params.c, u);
    float3 qcd = lerp(params.c, params.d, u);
    float3 qabc = lerp(qab, qbc, u);
    float3 qbcd = lerp(qbc, qcd, u);
    float3 qval = lerp(qabc, qbcd, u);
    float3 e = float3((qval.x - pt.x), (qval.y - pt.y), (qval.z - pt.z));
    float num = dot(e, q1);
    float den = (dot(q1, q1) + dot(e, q2));
    float abs_den = abs(den);
    float u_new = ((abs_den < 0.000000) ? u : (u - (num / den)));
    out_u[i] = u_new;
  }
  return;
}
