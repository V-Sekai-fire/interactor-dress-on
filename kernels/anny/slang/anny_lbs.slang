struct AnnyLbsParams {
  uint V;
  uint J;
};

[[vk::binding(0, 0)]]
ConstantBuffer<AnnyLbsParams> params;
[[vk::binding(1, 0)]]
StructuredBuffer<float> bind;
[[vk::binding(2, 0)]]
StructuredBuffer<float> weights;
[[vk::binding(3, 0)]]
StructuredBuffer<float> bone;
[[vk::binding(4, 0)]]
RWStructuredBuffer<float> verts;

[shader("compute")] [numthreads(64, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
  uint i = tid.x;
  if ((i >= params.V)) {
    return;
  }
  float px = bind[(i * 3u)];
  float py = bind[((i * 3u) + 1u)];
  float pz = bind[((i * 3u) + 2u)];
  float x = 0.000000;
  float y = 0.000000;
  float z = 0.000000;
  for (uint j = 0u; j < params.J; ++j) {
    float w = weights[((i * params.J) + j)];
    if ((w != 0.000000)) {
      uint b = (j * 12u);
      x = (x + (w * ((((bone[(b + 0u)] * px) + (bone[(b + 1u)] * py)) + (bone[(b + 2u)] * pz)) + bone[(b + 3u)])));
      y = (y + (w * ((((bone[(b + 4u)] * px) + (bone[(b + 5u)] * py)) + (bone[(b + 6u)] * pz)) + bone[(b + 7u)])));
      z = (z + (w * ((((bone[(b + 8u)] * px) + (bone[(b + 9u)] * py)) + (bone[(b + 10u)] * pz)) + bone[(b + 11u)])));
    }
  }
  verts[(i * 3u)] = x;
  verts[((i * 3u) + 1u)] = y;
  verts[((i * 3u) + 2u)] = z;
}
