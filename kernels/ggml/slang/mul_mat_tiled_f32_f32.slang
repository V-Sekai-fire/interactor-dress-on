struct Slot {
  uint base;
  uint pad0;
  uint pad1;
  uint pad2;
};

groupshared float As[16][65];
groupshared float Bs[16][65];

[[vk::binding(0, 0)]]
StructuredBuffer<uint> params;
[[vk::binding(1, 0)]]
RWStructuredBuffer<float> s0;
[[vk::binding(2, 0)]]
RWStructuredBuffer<float> s1;
[[vk::binding(3, 0)]]
RWStructuredBuffer<uint> s2;
[[vk::binding(4, 0)]]
RWStructuredBuffer<float> dst;
[[vk::binding(0, 1)]]
ConstantBuffer<Slot> slot;

uint pw(uint k) {
  return params[(slot.base + k)];
}

float lda(uint e) {
  return s0[e];
}

float ldb(uint e) {
  return s1[e];
}

[shader("compute")] [numthreads(16, 16, 1)]
void main(uint3 gid : SV_GroupID, uint3 lid : SV_GroupThreadID) {
  uint K = pw(10u);
  uint M = pw(11u);
  uint N = pw(20u);
  uint ne12 = pw(21u);
  uint b2 = (gid.z % ne12);
  uint b3 = (gid.z / ne12);
  uint aBase = ((pw(18u) + ((b2 / pw(55u)) * pw(16u))) + ((b3 / pw(56u)) * pw(17u)));
  uint bBase = ((pw(27u) + (b2 * pw(25u))) + (b3 * pw(26u)));
  uint dBase = ((pw(9u) + (b2 * pw(7u))) + (b3 * pw(8u)));
  uint ak = pw(14u);
  uint am = pw(15u);
  uint bk = pw(23u);
  uint bn = pw(24u);
  uint dm = pw(5u);
  uint dn = pw(6u);
  uint m0 = (gid.x * 64u);
  uint n0 = (gid.y * 64u);
  uint t = ((lid.y * 16u) + lid.x);
  float c00 = 0.0f;
  float c01 = 0.0f;
  float c02 = 0.0f;
  float c03 = 0.0f;
  float c10 = 0.0f;
  float c11 = 0.0f;
  float c12 = 0.0f;
  float c13 = 0.0f;
  float c20 = 0.0f;
  float c21 = 0.0f;
  float c22 = 0.0f;
  float c23 = 0.0f;
  float c30 = 0.0f;
  float c31 = 0.0f;
  float c32 = 0.0f;
  float c33 = 0.0f;
  uint nt = ((K + 15u) / 16u);
  for (uint kt = 0u; kt < nt; ++kt) {
    uint k0 = (kt * 16u);
    for (uint l = 0u; l < 4u; ++l) {
      uint idx = (t + (l * 256u));
      uint kk = (idx % 16u);
      uint rr = (idx / 16u);
      uint gk = (k0 + kk);
      uint gm = (m0 + rr);
      float av = 0.0f;
      if (((gk < K) && (gm < M))) {
        av = lda(((aBase + (gk * ak)) + (gm * am)));
      }
      As[kk][rr] = av;
      uint gn = (n0 + rr);
      float bv = 0.0f;
      if (((gk < K) && (gn < N))) {
        bv = ldb(((bBase + (gk * bk)) + (gn * bn)));
      }
      Bs[kk][rr] = bv;
    }
    GroupMemoryBarrierWithGroupSync();
    for (uint q = 0u; q < 16u; ++q) {
      float x0 = As[q][lid.x];
      float x1 = As[q][(lid.x + 16u)];
      float x2 = As[q][(lid.x + 32u)];
      float x3 = As[q][(lid.x + 48u)];
      float y0 = Bs[q][lid.y];
      float y1 = Bs[q][(lid.y + 16u)];
      float y2 = Bs[q][(lid.y + 32u)];
      float y3 = Bs[q][(lid.y + 48u)];
      c00 = (c00 + (x0 * y0));
      c01 = (c01 + (x0 * y1));
      c02 = (c02 + (x0 * y2));
      c03 = (c03 + (x0 * y3));
      c10 = (c10 + (x1 * y0));
      c11 = (c11 + (x1 * y1));
      c12 = (c12 + (x1 * y2));
      c13 = (c13 + (x1 * y3));
      c20 = (c20 + (x2 * y0));
      c21 = (c21 + (x2 * y1));
      c22 = (c22 + (x2 * y2));
      c23 = (c23 + (x2 * y3));
      c30 = (c30 + (x3 * y0));
      c31 = (c31 + (x3 * y1));
      c32 = (c32 + (x3 * y2));
      c33 = (c33 + (x3 * y3));
    }
    GroupMemoryBarrierWithGroupSync();
  }
  uint i0 = (m0 + lid.x);
  uint i1 = (m0 + (lid.x + 16u));
  uint i2 = (m0 + (lid.x + 32u));
  uint i3 = (m0 + (lid.x + 48u));
  uint j0 = (n0 + lid.y);
  uint j1 = (n0 + (lid.y + 16u));
  uint j2 = (n0 + (lid.y + 32u));
  uint j3 = (n0 + (lid.y + 48u));
  if (((i0 < M) && (j0 < N))) {
    dst[((dBase + (i0 * dm)) + (j0 * dn))] = c00;
  }
  if (((i0 < M) && (j1 < N))) {
    dst[((dBase + (i0 * dm)) + (j1 * dn))] = c01;
  }
  if (((i0 < M) && (j2 < N))) {
    dst[((dBase + (i0 * dm)) + (j2 * dn))] = c02;
  }
  if (((i0 < M) && (j3 < N))) {
    dst[((dBase + (i0 * dm)) + (j3 * dn))] = c03;
  }
  if (((i1 < M) && (j0 < N))) {
    dst[((dBase + (i1 * dm)) + (j0 * dn))] = c10;
  }
  if (((i1 < M) && (j1 < N))) {
    dst[((dBase + (i1 * dm)) + (j1 * dn))] = c11;
  }
  if (((i1 < M) && (j2 < N))) {
    dst[((dBase + (i1 * dm)) + (j2 * dn))] = c12;
  }
  if (((i1 < M) && (j3 < N))) {
    dst[((dBase + (i1 * dm)) + (j3 * dn))] = c13;
  }
  if (((i2 < M) && (j0 < N))) {
    dst[((dBase + (i2 * dm)) + (j0 * dn))] = c20;
  }
  if (((i2 < M) && (j1 < N))) {
    dst[((dBase + (i2 * dm)) + (j1 * dn))] = c21;
  }
  if (((i2 < M) && (j2 < N))) {
    dst[((dBase + (i2 * dm)) + (j2 * dn))] = c22;
  }
  if (((i2 < M) && (j3 < N))) {
    dst[((dBase + (i2 * dm)) + (j3 * dn))] = c23;
  }
  if (((i3 < M) && (j0 < N))) {
    dst[((dBase + (i3 * dm)) + (j0 * dn))] = c30;
  }
  if (((i3 < M) && (j1 < N))) {
    dst[((dBase + (i3 * dm)) + (j1 * dn))] = c31;
  }
  if (((i3 < M) && (j2 < N))) {
    dst[((dBase + (i3 * dm)) + (j2 * dn))] = c32;
  }
  if (((i3 < M) && (j3 < N))) {
    dst[((dBase + (i3 * dm)) + (j3 * dn))] = c33;
  }
}
