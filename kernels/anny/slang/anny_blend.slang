struct AnnyBlendParams {
  uint P;
  uint NB;
};

[[vk::binding(0, 0)]]
ConstantBuffer<AnnyBlendParams> params;
[[vk::binding(1, 0)]]
StructuredBuffer<float> templ;
[[vk::binding(2, 0)]]
StructuredBuffer<float> blend;
[[vk::binding(3, 0)]]
StructuredBuffer<float> coeffs;
[[vk::binding(4, 0)]]
RWStructuredBuffer<float> vanny;

[shader("compute")] [numthreads(64, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
  uint i = tid.x;
  if ((i >= params.P)) {
    return;
  }
  float x = templ[(i * 3u)];
  float y = templ[((i * 3u) + 1u)];
  float z = templ[((i * 3u) + 2u)];
  for (uint c = 0u; c < params.NB; ++c) {
    float w = coeffs[c];
    uint o = (((c * params.P) + i) * 3u);
    x = (x + (w * blend[o]));
    y = (y + (w * blend[(o + 1u)]));
    z = (z + (w * blend[(o + 2u)]));
  }
  vanny[(i * 3u)] = x;
  vanny[((i * 3u) + 1u)] = y;
  vanny[((i * 3u) + 2u)] = z;
}
