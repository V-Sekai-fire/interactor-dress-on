struct AnnyLbsParams {
  uint V;
  uint J;
};

[[vk::binding(0, 0)]]
ConstantBuffer<AnnyLbsParams> params;
[[vk::binding(1, 0)]]
StructuredBuffer<float> weights;
[[vk::binding(2, 0)]]
StructuredBuffer<float> bone;
[[vk::binding(3, 0)]]
StructuredBuffer<float> dverts;
[[vk::binding(4, 0)]]
RWStructuredBuffer<float> dbind;

[shader("compute")] [numthreads(64, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
  uint i = tid.x;
  if ((i >= params.V)) {
    return;
  }
  float gx = dverts[(i * 3u)];
  float gy = dverts[((i * 3u) + 1u)];
  float gz = dverts[((i * 3u) + 2u)];
  float x = 0.000000;
  float y = 0.000000;
  float z = 0.000000;
  for (uint j = 0u; j < params.J; ++j) {
    float w = weights[((i * params.J) + j)];
    if ((w != 0.000000)) {
      uint b = (j * 12u);
      x = (x + (w * (((bone[(b + 0u)] * gx) + (bone[(b + 4u)] * gy)) + (bone[(b + 8u)] * gz))));
      y = (y + (w * (((bone[(b + 1u)] * gx) + (bone[(b + 5u)] * gy)) + (bone[(b + 9u)] * gz))));
      z = (z + (w * (((bone[(b + 2u)] * gx) + (bone[(b + 6u)] * gy)) + (bone[(b + 10u)] * gz))));
    }
  }
  dbind[(i * 3u)] = x;
  dbind[((i * 3u) + 1u)] = y;
  dbind[((i * 3u) + 2u)] = z;
}
