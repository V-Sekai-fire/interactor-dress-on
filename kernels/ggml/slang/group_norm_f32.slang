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

float pf(uint k) {
  return asfloat(pw(k));
}

uint off4(uint w, uint i0, uint i1, uint i2, uint i3) {
  return ((((pw((w + 8u)) + (i0 * pw((w + 4u)))) + (i1 * pw((w + 5u)))) + (i2 * pw((w + 6u)))) + (i3 * pw((w + 7u))));
}

uint gidx(uint w, uint e, uint n0, uint n1, uint c0, uint i3) {
  uint r = (e / n0);
  return off4(w, (e % n0), (r % n1), (c0 + (r / n1)), i3);
}

[shader("compute")] [numthreads(256, 1, 1)]
void main(uint3 gid : SV_GroupID, uint3 lid : SV_GroupThreadID) {
  uint row = ((gid.y * pw(54u)) + gid.x);
  if ((row >= pw(55u))) {
    return;
  }
  uint t = lid.x;
  uint ng = pw(37u);
  uint g = (row % ng);
  uint i3 = (row / ng);
  uint ne0 = pw(10u);
  uint ne1 = pw(11u);
  uint ne2 = pw(12u);
  uint cpg = (((ne2 + ng) - 1u) / ng);
  uint start = (g * cpg);
  uint stop = (((start + cpg) < ne2) ? (start + cpg) : ne2);
  uint n = ((start < stop) ? ((ne0 * ne1) * (stop - start)) : 0u);
  float eps = pf(38u);
  float acc_sumx = 0.0f;
  uint i_sumx = t;
  while ((i_sumx < n)) {
    float x = s0[gidx(10u, i_sumx, ne0, ne1, start, i3)];
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
  float mean = (sumx / float(n));
  float acc_sumd = 0.0f;
  uint i_sumd = t;
  while ((i_sumd < n)) {
    float x = s0[gidx(10u, i_sumd, ne0, ne1, start, i3)];
    acc_sumd = (acc_sumd + ((x - mean) * (x - mean)));
    i_sumd = (i_sumd + 256u);
  }
  sh[t] = acc_sumd;
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
  float sumd = sh[0u];
  GroupMemoryBarrierWithGroupSync();
  float variance = (sumd / float(n));
  float scale = (1.0f / sqrt((variance + eps)));
  uint j = t;
  while ((j < n)) {
    float x = s0[gidx(10u, j, ne0, ne1, start, i3)];
    dst[gidx(1u, j, ne0, ne1, start, i3)] = ((x - mean) * scale);
    j = (j + 256u);
  }
}
