struct Slot {
  uint base;
  uint pad0;
  uint pad1;
  uint pad2;
};

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

[shader("compute")] [numthreads(1, 1, 1)]
void main(uint3 gid : SV_GroupID, uint3 lid : SV_GroupThreadID) {
  uint row = ((gid.y * pw(54u)) + gid.x);
  if ((row >= pw(55u))) {
    return;
  }
  float sh[256];
  uint n = pw(10u);
  uint r1 = (row % pw(11u));
  uint rr = (row / pw(11u));
  uint r2 = (rr % pw(12u));
  uint r3 = (rr / pw(12u));
  uint xb = off4(10u, 0u, r1, r2, r3);
  uint xs = pw(14u);
  uint yb = off4(1u, 0u, r1, r2, r3);
  uint ys = pw(5u);
  float eps = pf(37u);
  for (uint t = 0u; t < 256u; ++t) {
    float acc_sumsq = 0.0f;
    uint i_sumsq = t;
    while ((i_sumsq < n)) {
      float x = s0[(xb + (i_sumsq * xs))];
      acc_sumsq = (acc_sumsq + (x * x));
      i_sumsq = (i_sumsq + 256u);
    }
    sh[t] = acc_sumsq;
  }
  for (uint t = 0u; t < 128u; ++t) {
    sh[t] = (sh[t] + sh[(t + 128u)]);
  }
  for (uint t = 0u; t < 64u; ++t) {
    sh[t] = (sh[t] + sh[(t + 64u)]);
  }
  for (uint t = 0u; t < 32u; ++t) {
    sh[t] = (sh[t] + sh[(t + 32u)]);
  }
  for (uint t = 0u; t < 16u; ++t) {
    sh[t] = (sh[t] + sh[(t + 16u)]);
  }
  for (uint t = 0u; t < 8u; ++t) {
    sh[t] = (sh[t] + sh[(t + 8u)]);
  }
  for (uint t = 0u; t < 4u; ++t) {
    sh[t] = (sh[t] + sh[(t + 4u)]);
  }
  for (uint t = 0u; t < 2u; ++t) {
    sh[t] = (sh[t] + sh[(t + 2u)]);
  }
  for (uint t = 0u; t < 1u; ++t) {
    sh[t] = (sh[t] + sh[(t + 1u)]);
  }
  float sumsq = sh[0u];
  float ms = (sumsq / float(n));
  float scale = (1.0f / sqrt((ms + eps)));
  for (uint j = 0u; j < n; ++j) {
    float x = s0[(xb + (j * xs))];
    dst[(yb + (j * ys))] = (x * scale);
  }
}
