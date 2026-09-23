struct df {
  float hi;
  float lo;
};

struct PsdParams {
  uint count;
  uint enabled;
};

[[vk::binding(0, 0)]]
RWStructuredBuffer<df> blocks;
[[vk::binding(1, 0)]]
ConstantBuffer<PsdParams> params;

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
  uint k = tid.x;
  if ((k >= params.count)) {
    return;
  }
  if ((params.enabled == 0u)) {
    return;
  }
  uint ob = (144u * k);
  df A[144];
  df V[144];
  for (uint i = 0u; i < 144u; ++i) {
    A[i] = blocks[(ob + i)];
    V[i] = df_make(0.0f, 0.0f);
  }
  for (uint i = 0u; i < 12u; ++i) {
    V[(i * 13u)] = df_f(1.0f);
  }
  bool go = true;
  for (uint sweep = 0u; sweep < 16u; ++sweep) {
    if (go) {
      float off = 0.0f;
      float dg = 0.0f;
      for (uint p = 0u; p < 12u; ++p) {
        dg = (dg + (A[(p * 13u)].hi * A[(p * 13u)].hi));
        for (uint q = (p + 1u); q < 12u; ++q) {
          off = (off + (A[((p * 12u) + q)].hi * A[((p * 12u) + q)].hi));
        }
      }
      if ((off <= (dg * 1.0e-30f))) {
        go = false;
      } else {
        for (uint p = 0u; p < 12u; ++p) {
          for (uint q = (p + 1u); q < 12u; ++q) {
            df apq = A[((p * 12u) + q)];
            if (((apq.hi != 0.0f) || (apq.lo != 0.0f))) {
              df theta = df_div(df_sub(A[(q * 13u)], A[(p * 13u)]), df_mulf(apq, 2.0f));
              df at = df_abs(theta);
              df t = df_div(df_f(1.0f), df_add(at, df_sqrt(df_add(df_mul(at, at), df_f(1.0f)))));
              if ((theta.hi < 0.0f)) {
                t = df_neg(t);
              }
              df c = df_div(df_f(1.0f), df_sqrt(df_add(df_mul(t, t), df_f(1.0f))));
              df s = df_mul(t, c);
              for (uint r = 0u; r < 12u; ++r) {
                df acp = A[((r * 12u) + p)];
                df acq = A[((r * 12u) + q)];
                A[((r * 12u) + p)] = df_sub(df_mul(c, acp), df_mul(s, acq));
                A[((r * 12u) + q)] = df_add(df_mul(s, acp), df_mul(c, acq));
              }
              for (uint r = 0u; r < 12u; ++r) {
                df arp = A[((p * 12u) + r)];
                df arq = A[((q * 12u) + r)];
                A[((p * 12u) + r)] = df_sub(df_mul(c, arp), df_mul(s, arq));
                A[((q * 12u) + r)] = df_add(df_mul(s, arp), df_mul(c, arq));
              }
              for (uint r = 0u; r < 12u; ++r) {
                df vcp = V[((r * 12u) + p)];
                df vcq = V[((r * 12u) + q)];
                V[((r * 12u) + p)] = df_sub(df_mul(c, vcp), df_mul(s, vcq));
                V[((r * 12u) + q)] = df_add(df_mul(s, vcp), df_mul(c, vcq));
              }
            }
          }
        }
      }
    }
  }
  bool negative = false;
  for (uint i = 0u; i < 12u; ++i) {
    if (df_lt(A[(i * 13u)], df_make(0.0f, 0.0f))) {
      negative = true;
    }
  }
  if ((!negative)) {
    return;
  }
  for (uint i = 0u; i < 12u; ++i) {
    for (uint j = 0u; j < 12u; ++j) {
      df acc = df_make(0.0f, 0.0f);
      for (uint m = 0u; m < 12u; ++m) {
        df lam = A[(m * 13u)];
        if (df_lt(df_make(0.0f, 0.0f), lam)) {
          acc = df_add(acc, df_mul(lam, df_mul(V[((i * 12u) + m)], V[((j * 12u) + m)])));
        }
      }
      blocks[(ob + ((i * 12u) + j))] = acc;
    }
  }
}
