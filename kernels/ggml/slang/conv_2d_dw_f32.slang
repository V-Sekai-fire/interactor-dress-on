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
  uint kw = pw(10u);
  uint kh = pw(11u);
  uint iw = pw(19u);
  uint ih = pw(20u);
  uint st0 = pw(37u);
  uint st1 = pw(38u);
  uint p0 = pw(39u);
  uint p1 = pw(40u);
  uint d0 = pw(41u);
  uint d1 = pw(42u);
  float sum = 0.0f;
  for (uint ky = 0u; ky < kh; ++ky) {
    uint qy = ((i1 * st1) + (ky * d1));
    if (((qy >= p1) && ((qy - p1) < ih))) {
      for (uint kx = 0u; kx < kw; ++kx) {
        uint qx = ((i0 * st0) + (kx * d0));
        if (((qx >= p0) && ((qx - p0) < iw))) {
          sum = (sum + (s0[off4(10u, kx, ky, 0u, i2)] * s1[off4(19u, (qx - p0), (qy - p1), i2, i3)]));
        }
      }
    }
  }
  dst[d] = sum;
}
