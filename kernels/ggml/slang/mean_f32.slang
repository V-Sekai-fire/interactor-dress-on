struct Slot {
  uint base;
  uint pad0;
  uint pad1;
  uint pad2;
};

groupshared float sh[256];

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

uint off4(uint w, uint i0, uint i1, uint i2, uint i3) {
  return ((((pw((w + 8u)) + (i0 * pw((w + 4u)))) + (i1 * pw((w + 5u)))) + (i2 * pw((w + 6u)))) + (i3 * pw((w + 7u))));
}

[shader("compute")] [numthreads(256, 1, 1)]
void main(uint3 gid : SV_GroupID, uint3 lid : SV_GroupThreadID) {
  uint row = ((gid.y * pw(54u)) + gid.x);
  if ((row >= pw(55u))) {
    return;
  }
  uint t = lid.x;
  uint n = pw(10u);
  uint r1 = (row % pw(11u));
  uint rr = (row / pw(11u));
  uint r2 = (rr % pw(12u));
  uint r3 = (rr / pw(12u));
  uint xb = off4(10u, 0u, r1, r2, r3);
  uint xs = pw(14u);
  uint yb = off4(1u, 0u, r1, r2, r3);
  uint ys = pw(5u);
  float acc_sumx = 0.0f;
  uint i_sumx = t;
  while ((i_sumx < n)) {
    float x = s0[(xb + (i_sumx * xs))];
    acc_sumx = (acc_sumx + x);
    i_sumx = (i_sumx + 256u);
  }
  sh[t] = acc_sumx;
  GroupMemoryBarrierWithGroupSync();
  if ((t < 128u)) {
    sh[t] = (sh[t] + sh[(t + 128u)]);
  }
  GroupMemoryBarrierWithGroupSync();
  if ((t < 64u)) {
    sh[t] = (sh[t] + sh[(t + 64u)]);
  }
  GroupMemoryBarrierWithGroupSync();
  if ((t < 32u)) {
    sh[t] = (sh[t] + sh[(t + 32u)]);
  }
  GroupMemoryBarrierWithGroupSync();
  if ((t < 16u)) {
    sh[t] = (sh[t] + sh[(t + 16u)]);
  }
  GroupMemoryBarrierWithGroupSync();
  if ((t < 8u)) {
    sh[t] = (sh[t] + sh[(t + 8u)]);
  }
  GroupMemoryBarrierWithGroupSync();
  if ((t < 4u)) {
    sh[t] = (sh[t] + sh[(t + 4u)]);
  }
  GroupMemoryBarrierWithGroupSync();
  if ((t < 2u)) {
    sh[t] = (sh[t] + sh[(t + 2u)]);
  }
  GroupMemoryBarrierWithGroupSync();
  if ((t < 1u)) {
    sh[t] = (sh[t] + sh[(t + 1u)]);
  }
  GroupMemoryBarrierWithGroupSync();
  float sumx = sh[0u];
  GroupMemoryBarrierWithGroupSync();
  if ((t == 0u)) {
    dst[yb] = (sumx / float(n));
  }
}
