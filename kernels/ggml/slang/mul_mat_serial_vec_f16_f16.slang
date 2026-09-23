struct Slot {
  uint base;
  uint pad0;
  uint pad1;
  uint pad2;
};

[[vk::binding(0, 0)]]
StructuredBuffer<uint> params;
[[vk::binding(1, 0)]]
RWStructuredBuffer<uint> s0;
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

float lda(uint e) {
  uint w = s0[(e >> 1u)];
  uint h = (((e & 1u) != 0u) ? (w >> 16u) : (w & 65535u));
  return f16tof32(h);
}

float ldb(uint e) {
  uint w = s1[(e >> 1u)];
  uint h = (((e & 1u) != 0u) ? (w >> 16u) : (w & 65535u));
  return f16tof32(h);
}

[shader("compute")] [numthreads(32, 8, 1)]
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
  uint row = ((((gid.y * pw(57u)) + gid.x) * 8u) + lid.y);
  float p0 = 0.0f;
  float p1 = 0.0f;
  float p2 = 0.0f;
  float p3 = 0.0f;
  uint lane = lid.x;
  if (((lane != 0u) || (row >= M))) {
    return;
  }
  float q0[32];
  float q1[32];
  float q2[32];
  float q3[32];
  for (uint l = 0u; l < 32u; ++l) {
    p0 = 0.0f;
    p1 = 0.0f;
    p2 = 0.0f;
    p3 = 0.0f;
    uint k = l;
    while ((k < K)) {
      float a = lda(((aBase + (k * ak)) + (row * am)));
      uint bo = (bBase + (k * bk));
      p0 = (p0 + (a * ldb(bo)));
      if ((N > 1u)) {
        p1 = (p1 + (a * ldb((bo + bn))));
      }
      if ((N > 2u)) {
        p2 = (p2 + (a * ldb((bo + (2u * bn)))));
      }
      if ((N > 3u)) {
        p3 = (p3 + (a * ldb((bo + (3u * bn)))));
      }
      k = (k + 32u);
    }
    q0[l] = p0;
    q1[l] = p1;
    q2[l] = p2;
    q3[l] = p3;
  }
  for (uint l = 0u; l < 16u; ++l) {
    q0[l] = (q0[l] + q0[(l + 16u)]);
    q1[l] = (q1[l] + q1[(l + 16u)]);
    q2[l] = (q2[l] + q2[(l + 16u)]);
    q3[l] = (q3[l] + q3[(l + 16u)]);
  }
  for (uint l = 0u; l < 8u; ++l) {
    q0[l] = (q0[l] + q0[(l + 8u)]);
    q1[l] = (q1[l] + q1[(l + 8u)]);
    q2[l] = (q2[l] + q2[(l + 8u)]);
    q3[l] = (q3[l] + q3[(l + 8u)]);
  }
  for (uint l = 0u; l < 4u; ++l) {
    q0[l] = (q0[l] + q0[(l + 4u)]);
    q1[l] = (q1[l] + q1[(l + 4u)]);
    q2[l] = (q2[l] + q2[(l + 4u)]);
    q3[l] = (q3[l] + q3[(l + 4u)]);
  }
  for (uint l = 0u; l < 2u; ++l) {
    q0[l] = (q0[l] + q0[(l + 2u)]);
    q1[l] = (q1[l] + q1[(l + 2u)]);
    q2[l] = (q2[l] + q2[(l + 2u)]);
    q3[l] = (q3[l] + q3[(l + 2u)]);
  }
  for (uint l = 0u; l < 1u; ++l) {
    q0[l] = (q0[l] + q0[(l + 1u)]);
    q1[l] = (q1[l] + q1[(l + 1u)]);
    q2[l] = (q2[l] + q2[(l + 1u)]);
    q3[l] = (q3[l] + q3[(l + 1u)]);
  }
  if ((N > 0u)) {
    dst[((dBase + (row * dm)) + (0u * dn))] = q0[0u];
  }
  if ((N > 1u)) {
    dst[((dBase + (row * dm)) + (1u * dn))] = q1[0u];
  }
  if ((N > 2u)) {
    dst[((dBase + (row * dm)) + (2u * dn))] = q2[0u];
  }
  if ((N > 3u)) {
    dst[((dBase + (row * dm)) + (3u * dn))] = q3[0u];
  }
}
