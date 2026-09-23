struct SpmvDf32Params {
  uint rows;
};

[[vk::binding(0, 0)]]
ConstantBuffer<SpmvDf32Params> params;
[[vk::binding(1, 0)]]
StructuredBuffer<int> rowPtr;
[[vk::binding(2, 0)]]
StructuredBuffer<int> colIdx;
[[vk::binding(3, 0)]]
StructuredBuffer<float> values;
[[vk::binding(4, 0)]]
StructuredBuffer<float> x;
[[vk::binding(5, 0)]]
RWStructuredBuffer<float> y;

void two_sum(float a, float b, out float hi, out float lo) {
  float h = (a + b);
  float bb = (h - a);
  float ah = (h - bb);
  float lo_a = (a - ah);
  float lo_b = (b - bb);
  hi = h;
  lo = (lo_a + lo_b);
  return;
}

void quick_two_sum(float a, float b, out float hi, out float lo) {
  float h = (a + b);
  float t = (h - a);
  hi = h;
  lo = (b - t);
  return;
}

void two_prod(float a, float b, out float hi, out float lo) {
  float h = (a * b);
  hi = h;
  lo = fma(a, b, (-h));
  return;
}

void df_add(float x_hi, float x_lo, float y_hi, float y_lo, out float z_hi, out float z_lo) {
  float sh;
  float sl;
  two_sum(x_hi, y_hi, sh, sl);
  float xy_lo = (x_lo + y_lo);
  float sl2 = (sl + xy_lo);
  quick_two_sum(sh, sl2, z_hi, z_lo);
  return;
}

[shader("compute")] [numthreads(256, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
  uint i = tid.x;
  if ((i >= params.rows)) {
    return;
  }
  uint rs = uint(rowPtr[i]);
  uint re = uint(rowPtr[(i + 1u)]);
  float s_hi = 0.000000;
  float s_lo = 0.000000;
  for (uint p = rs; p < re; ++p) {
    float v = values[p];
    float xc = x[uint(colIdx[p])];
    float t_hi;
    float t_lo;
    two_prod(v, xc, t_hi, t_lo);
    float n_hi;
    float n_lo;
    df_add(s_hi, s_lo, t_hi, t_lo, n_hi, n_lo);
    s_hi = n_hi;
    s_lo = n_lo;
  }
  y[i] = (s_hi + s_lo);
  return;
}
