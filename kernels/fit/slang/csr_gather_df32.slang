struct df {
  float hi;
  float lo;
};

struct GatherParams {
  uint nnz;
};

[[vk::binding(0, 0)]]
StructuredBuffer<uint> ptr;
[[vk::binding(1, 0)]]
StructuredBuffer<uint> src;
[[vk::binding(2, 0)]]
StructuredBuffer<df> blocks;
[[vk::binding(3, 0)]]
StructuredBuffer<df> weight;
[[vk::binding(4, 0)]]
RWStructuredBuffer<df> values;
[[vk::binding(5, 0)]]
ConstantBuffer<GatherParams> params;

df df_make(float h, float l) {
  df r;
  r.hi = h;
  r.lo = l;
  return r;
}

df df_f(float a) {
  return df_make(a, 0.0f);
}

df two_sum(float a, float b) {
  precise float s = (a + b);
  precise float bb = (s - a);
  precise float ea = (a - (s - bb));
  precise float eb = (b - bb);
  precise float e = (ea + eb);
  return df_make(s, e);
}

df quick_two_sum(float a, float b) {
  precise float s = (a + b);
  precise float e = (b - (s - a));
  return df_make(s, e);
}

df two_prod(float a, float b) {
  precise float p = (a * b);
  precise float e = fma(a, b, (-p));
  return df_make(p, e);
}

df df_add(df x, df y) {
  df s = two_sum(x.hi, y.hi);
  df t = two_sum(x.lo, y.lo);
  precise float sl = (s.lo + t.hi);
  df u = quick_two_sum(s.hi, sl);
  precise float ul = (u.lo + t.lo);
  return quick_two_sum(u.hi, ul);
}

df df_neg(df x) {
  return df_make((-x.hi), (-x.lo));
}

df df_sub(df x, df y) {
  return df_add(x, df_neg(y));
}

df df_mul(df x, df y) {
  df p = two_prod(x.hi, y.hi);
  precise float c = (x.hi * y.lo);
  precise float d = (x.lo * y.hi);
  precise float cd = (c + d);
  precise float pl = (p.lo + cd);
  return quick_two_sum(p.hi, pl);
}

df df_mulf(df x, float c) {
  df p = two_prod(x.hi, c);
  precise float d = (x.lo * c);
  precise float pl = (p.lo + d);
  return quick_two_sum(p.hi, pl);
}

df df_div(df x, df y) {
  float q1 = (x.hi / y.hi);
  df r1 = df_sub(x, df_mulf(y, q1));
  float q2 = (r1.hi / y.hi);
  df r2 = df_sub(r1, df_mulf(y, q2));
  float q3 = (r2.hi / y.hi);
  df q = quick_two_sum(q1, q2);
  return df_add(q, df_f(q3));
}

df df_sqrt(df x) {
  if ((x.hi <= 0.0f)) {
    return df_make(0.0f, 0.0f);
  }
  float r = sqrt(x.hi);
  df rr = two_prod(r, r);
  df d = df_sub(x, rr);
  precise float dd = (d.hi + d.lo);
  float corr = (dd / (2.0f * r));
  return quick_two_sum(r, corr);
}

float df_to_f(df x) {
  return (x.hi + x.lo);
}

bool df_lt(df x, df y) {
  return ((x.hi < y.hi) || ((x.hi == y.hi) && (x.lo < y.lo)));
}

df df_abs(df x) {
  if (df_lt(x, df_make(0.0f, 0.0f))) {
    return df_neg(x);
  }
  return x;
}

[shader("compute")] [numthreads(64, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
  uint i = tid.x;
  if ((i >= params.nnz)) {
    return;
  }
  df acc = df_make(0.0f, 0.0f);
  for (uint j = ptr[i]; j < ptr[(i + 1u)]; ++j) {
    uint s = src[j];
    uint h = (s / 144u);
    acc = df_add(acc, df_mul(weight[h], blocks[s]));
  }
  values[i] = acc;
}
