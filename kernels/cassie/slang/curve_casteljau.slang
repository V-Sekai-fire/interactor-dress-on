struct CasteljauParams {
  float3 a;
  float3 b;
  float3 c;
  float3 d;
  float u;
};

[[vk::binding(0, 0)]]
ConstantBuffer<CasteljauParams> params;
[[vk::binding(1, 0)]]
RWStructuredBuffer<float3> out;

[shader("compute")] [numthreads(1, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
  float3 ab = lerp(params.a, params.b, params.u);
  float3 bc = lerp(params.b, params.c, params.u);
  float3 cd = lerp(params.c, params.d, params.u);
  float3 abc = lerp(ab, bc, params.u);
  float3 bcd = lerp(bc, cd, params.u);
  float3 split = lerp(abc, bcd, params.u);
  out[0u] = params.a;
  out[1u] = ab;
  out[2u] = abc;
  out[3u] = split;
  out[4u] = split;
  out[5u] = bcd;
  out[6u] = cd;
  out[7u] = params.d;
}
