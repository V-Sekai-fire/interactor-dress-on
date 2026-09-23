struct AabbParams {
  uint n_v;
  uint n_e;
  uint n_f;
  float inflation;
  float margin;
};

[[vk::binding(0, 0)]]
StructuredBuffer<float> x0;
[[vk::binding(1, 0)]]
StructuredBuffer<float> x1;
[[vk::binding(2, 0)]]
StructuredBuffer<uint> edges;
[[vk::binding(3, 0)]]
StructuredBuffer<uint> faces;
[[vk::binding(4, 0)]]
RWStructuredBuffer<float> boxes;
[[vk::binding(5, 0)]]
ConstantBuffer<AabbParams> params;

[shader("compute")] [numthreads(64, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
  uint i = tid.x;
  if ((i >= ((params.n_v + params.n_e) + params.n_f))) {
    return;
  }
  uint vv[3];
  uint nv = 1u;
  vv[0u] = i;
  vv[1u] = 0u;
  vv[2u] = 0u;
  if ((i >= params.n_v)) {
    if ((i < (params.n_v + params.n_e))) {
      uint e = (i - params.n_v);
      nv = 2u;
      vv[0u] = edges[(2u * e)];
      vv[1u] = edges[((2u * e) + 1u)];
    } else {
      uint f = ((i - params.n_v) - params.n_e);
      nv = 3u;
      vv[0u] = faces[(3u * f)];
      vv[1u] = faces[((3u * f) + 1u)];
      vv[2u] = faces[((3u * f) + 2u)];
    }
  }
  float mn[3];
  float mx[3];
  for (uint d = 0u; d < 3u; ++d) {
    mn[d] = 3.0e38f;
    mx[d] = (-3.0e38f);
  }
  for (uint k = 0u; k < nv; ++k) {
    uint b = (3u * vv[k]);
    for (uint d = 0u; d < 3u; ++d) {
      float a0 = x0[(b + d)];
      float a1 = x1[(b + d)];
      mn[d] = min(mn[d], min(a0, a1));
      mx[d] = max(mx[d], max(a0, a1));
    }
  }
  uint ob = (6u * i);
  for (uint d = 0u; d < 3u; ++d) {
    float m = (params.margin * (1.0f + max(abs(mn[d]), abs(mx[d]))));
    boxes[(ob + d)] = ((mn[d] - params.inflation) - m);
    boxes[((ob + 3u) + d)] = ((mx[d] + params.inflation) + m);
  }
}
