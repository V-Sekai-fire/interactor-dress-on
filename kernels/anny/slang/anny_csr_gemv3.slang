struct AnnyCsrParams {
  uint rows;
  uint accumulate;
};

[[vk::binding(0, 0)]]
ConstantBuffer<AnnyCsrParams> params;
[[vk::binding(1, 0)]]
StructuredBuffer<uint> rowptr;
[[vk::binding(2, 0)]]
StructuredBuffer<uint> col;
[[vk::binding(3, 0)]]
StructuredBuffer<float> val;
[[vk::binding(4, 0)]]
StructuredBuffer<float> x;
[[vk::binding(5, 0)]]
RWStructuredBuffer<float> y;

[shader("compute")] [numthreads(64, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
  uint i = tid.x;
  if ((i >= params.rows)) {
    return;
  }
  bool acc = (params.accumulate != 0u);
  float a = (acc ? y[(i * 3u)] : 0.000000);
  float b = (acc ? y[((i * 3u) + 1u)] : 0.000000);
  float c = (acc ? y[((i * 3u) + 2u)] : 0.000000);
  for (uint k = rowptr[i]; k < rowptr[(i + 1u)]; ++k) {
    float w = val[k];
    uint o = (col[k] * 3u);
    a = (a + (w * x[o]));
    b = (b + (w * x[(o + 1u)]));
    c = (c + (w * x[(o + 2u)]));
  }
  y[(i * 3u)] = a;
  y[((i * 3u) + 1u)] = b;
  y[((i * 3u) + 2u)] = c;
}
