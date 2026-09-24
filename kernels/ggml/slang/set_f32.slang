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

[shader("compute")] [numthreads(256, 1, 1)]
void main(uint3 gid : SV_GroupID, uint3 lid : SV_GroupThreadID) {
  uint e = ((((gid.y * pw(54u)) + gid.x) * 256u) + lid.x);
  if ((e >= pw(53u))) {
    return;
  }
  uint i0;
  uint i1;
  uint i2;
  uint i3;
  unravel4(e, 1u, i0, i1, i2, i3);
  uint d = off4(1u, i0, i1, i2, i3);
  uint l = (((i0 + (i1 * pw(1u))) + (i2 * (pw(1u) * pw(2u)))) + (i3 * ((pw(1u) * pw(2u)) * pw(3u))));
  float x = s0[off4(10u, i0, i1, i2, i3)];
  uint o = pw(58u);
  if ((l >= o)) {
    uint r = (l - o);
    uint j3 = (r / pw(57u));
    r = (r % pw(57u));
    uint j2 = (r / pw(56u));
    r = (r % pw(56u));
    uint j1 = (r / pw(55u));
    uint j0 = (r % pw(55u));
    if ((((j0 < pw(19u)) && (j1 < pw(20u))) && ((j2 < pw(21u)) && (j3 < pw(22u))))) {
      x = s1[off4(19u, j0, j1, j2, j3)];
    }
  }
  dst[d] = x;
}
