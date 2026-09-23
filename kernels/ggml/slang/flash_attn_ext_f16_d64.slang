struct Slot {
  uint base;
  uint pad0;
  uint pad1;
  uint pad2;
};

groupshared float Qs[16][65];
groupshared float Ks[32][65];
groupshared float Vs[32][64];
groupshared float Ss[16][33];

[[vk::binding(0, 0)]]
StructuredBuffer<uint> params;
[[vk::binding(1, 0)]]
RWStructuredBuffer<float> s0;
[[vk::binding(2, 0)]]
RWStructuredBuffer<uint> s1;
[[vk::binding(3, 0)]]
RWStructuredBuffer<uint> s2;
[[vk::binding(4, 0)]]
RWStructuredBuffer<float> dst;
[[vk::binding(0, 1)]]
ConstantBuffer<Slot> slot;

uint pw(uint k) {
  return params[(slot.base + k)];
}

float pf(uint k) {
  return asfloat(pw(k));
}

uint off4(uint w, uint i0, uint i1, uint i2, uint i3) {
  return ((((pw((w + 8u)) + (i0 * pw((w + 4u)))) + (i1 * pw((w + 5u)))) + (i2 * pw((w + 6u)))) + (i3 * pw((w + 7u))));
}

float ld_f16_s1(uint e) {
  uint w = s1[(e >> 1u)];
  uint h = (((e & 1u) != 0u) ? (w >> 16u) : (w & 65535u));
  return f16tof32(h);
}

float ld_f16_s2(uint e) {
  uint w = s2[(e >> 1u)];
  uint h = (((e & 1u) != 0u) ? (w >> 16u) : (w & 65535u));
  return f16tof32(h);
}

[shader("compute")] [numthreads(128, 1, 1)]
void main(uint3 gid : SV_GroupID, uint3 lid : SV_GroupThreadID) {
  uint t = lid.x;
  uint r = (t / 8u);
  uint l = (t % 8u);
  uint iq = ((gid.x * 16u) + r);
  uint h = gid.y;
  uint b = gid.z;
  uint lk = pw(20u);
  uint hk = (h / pw(55u));
  uint bk = (b / pw(56u));
  uint hv = (h / pw(57u));
  uint bv = (b / pw(58u));
  float scale = pf(59u);
  float cap = pf(60u);
  uint qb = off4(10u, 0u, iq, h, b);
  uint kb = off4(19u, 0u, 0u, hk, bk);
  uint kn = pw(24u);
  uint vb = off4(28u, 0u, 0u, hv, bv);
  uint vn = pw(33u);
  bool valid = (iq < pw(11u));
  for (uint i = 0u; i < 8u; ++i) {
    uint c = (l + (i * 8u));
    float x = 0.000000;
    if (valid) {
      x = f16tof32(f32tof16(s0[(qb + c)]));
    }
    Qs[r][c] = x;
  }
  float m = asfloat(4286578688u);
  float ls = 0.000000;
  float acc0 = 0.000000;
  float acc1 = 0.000000;
  float acc2 = 0.000000;
  float acc3 = 0.000000;
  float acc4 = 0.000000;
  float acc5 = 0.000000;
  float acc6 = 0.000000;
  float acc7 = 0.000000;
  uint nt = ((lk + 31u) / 32u);
  for (uint tt = 0u; tt < nt; ++tt) {
    uint k0 = (tt * 32u);
    GroupMemoryBarrierWithGroupSync();
    for (uint i = 0u; i < 16u; ++i) {
      uint f = (t + (i * 128u));
      uint j = (f / 64u);
      uint c = (f % 64u);
      uint key = (k0 + j);
      float kx = 0.000000;
      float vx = 0.000000;
      if ((key < lk)) {
        kx = ld_f16_s1(((kb + (key * kn)) + c));
        vx = ld_f16_s2(((vb + (key * vn)) + c));
      }
      Ks[j][c] = kx;
      Vs[j][c] = vx;
    }
    GroupMemoryBarrierWithGroupSync();
    float d0 = 0.000000;
    float d1 = 0.000000;
    float d2 = 0.000000;
    float d3 = 0.000000;
    for (uint c = 0u; c < 64u; ++c) {
      float qc = Qs[r][c];
      d0 = (d0 + (qc * Ks[l][c]));
      d1 = (d1 + (qc * Ks[(l + 8u)][c]));
      d2 = (d2 + (qc * Ks[(l + 16u)][c]));
      d3 = (d3 + (qc * Ks[(l + 24u)][c]));
    }
    d0 = (d0 * scale);
    if ((cap != 0.000000)) {
      d0 = (cap * tanh(d0));
    }
    if (((k0 + l) >= lk)) {
      d0 = asfloat(4286578688u);
    }
    Ss[r][l] = d0;
    d1 = (d1 * scale);
    if ((cap != 0.000000)) {
      d1 = (cap * tanh(d1));
    }
    if (((k0 + (l + 8u)) >= lk)) {
      d1 = asfloat(4286578688u);
    }
    Ss[r][(l + 8u)] = d1;
    d2 = (d2 * scale);
    if ((cap != 0.000000)) {
      d2 = (cap * tanh(d2));
    }
    if (((k0 + (l + 16u)) >= lk)) {
      d2 = asfloat(4286578688u);
    }
    Ss[r][(l + 16u)] = d2;
    d3 = (d3 * scale);
    if ((cap != 0.000000)) {
      d3 = (cap * tanh(d3));
    }
    if (((k0 + (l + 24u)) >= lk)) {
      d3 = asfloat(4286578688u);
    }
    Ss[r][(l + 24u)] = d3;
    GroupMemoryBarrierWithGroupSync();
    float mt = m;
    for (uint j = 0u; j < 32u; ++j) {
      mt = max(mt, Ss[r][j]);
    }
    float corr = exp((m - mt));
    acc0 = (acc0 * corr);
    acc1 = (acc1 * corr);
    acc2 = (acc2 * corr);
    acc3 = (acc3 * corr);
    acc4 = (acc4 * corr);
    acc5 = (acc5 * corr);
    acc6 = (acc6 * corr);
    acc7 = (acc7 * corr);
    float ps = 0.000000;
    for (uint j = 0u; j < 32u; ++j) {
      float p = exp((Ss[r][j] - mt));
      ps = (ps + p);
      acc0 = (acc0 + (p * Vs[j][l]));
      acc1 = (acc1 + (p * Vs[j][(l + 8u)]));
      acc2 = (acc2 + (p * Vs[j][(l + 16u)]));
      acc3 = (acc3 + (p * Vs[j][(l + 24u)]));
      acc4 = (acc4 + (p * Vs[j][(l + 32u)]));
      acc5 = (acc5 + (p * Vs[j][(l + 40u)]));
      acc6 = (acc6 + (p * Vs[j][(l + 48u)]));
      acc7 = (acc7 + (p * Vs[j][(l + 56u)]));
    }
    ls = ((ls * corr) + ps);
    m = mt;
  }
  if (valid) {
    float inv = ((ls == 0.000000) ? 0.000000 : (1.000000 / ls));
    dst[off4(1u, l, h, iq, b)] = (acc0 * inv);
    dst[off4(1u, (l + 8u), h, iq, b)] = (acc1 * inv);
    dst[off4(1u, (l + 16u), h, iq, b)] = (acc2 * inv);
    dst[off4(1u, (l + 24u), h, iq, b)] = (acc3 * inv);
    dst[off4(1u, (l + 32u), h, iq, b)] = (acc4 * inv);
    dst[off4(1u, (l + 40u), h, iq, b)] = (acc5 * inv);
    dst[off4(1u, (l + 48u), h, iq, b)] = (acc6 * inv);
    dst[off4(1u, (l + 56u), h, iq, b)] = (acc7 * inv);
  }
}
