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

void unravel4(uint e, uint w, out uint i0, out uint i1, out uint i2, out uint i3) {
  uint n0 = pw(w);
  uint n1 = pw((w + 1u));
  uint n2 = pw((w + 2u));
  i0 = (e % n0);
  uint r = (e / n0);
  i1 = (r % n1);
  r = (r / n1);
  i2 = (r % n2);
  i3 = (r / n2);
}

uint off4(uint w, uint i0, uint i1, uint i2, uint i3) {
  return ((((pw((w + 8u)) + (i0 * pw((w + 4u)))) + (i1 * pw((w + 5u)))) + (i2 * pw((w + 6u)))) + (i3 * pw((w + 7u))));
}

[shader("compute")] [numthreads(1, 1, 1)]
void main(uint3 gid : SV_GroupID, uint3 lid : SV_GroupThreadID) {
  uint e = ((((gid.y * pw(54u)) + gid.x) * 1u) + lid.x);
  if ((e >= pw(53u))) {
    return;
  }
  uint n = (((pw(10u) * pw(11u)) * pw(12u)) * pw(13u));
  float sh[256];
  for (uint t = 0u; t < 256u; ++t) {
    float acc = 0.0f;
    uint i = t;
    while ((i < n)) {
      uint j0;
      uint j1;
      uint j2;
      uint j3;
      unravel4(i, 10u, j0, j1, j2, j3);
      float x = s0[off4(10u, j0, j1, j2, j3)];
      acc = (acc + x);
      i = (i + 256u);
    }
    sh[t] = acc;
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
  float total = sh[0u];
  dst[pw(9u)] = total;
}
