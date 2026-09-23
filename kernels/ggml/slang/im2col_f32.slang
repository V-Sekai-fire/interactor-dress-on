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

float im2col_at(uint i0, uint i1, uint i2, uint i3) {
  bool is2d = (pw(43u) == 1u);
  uint kw = pw(10u);
  uint kh = (is2d ? pw(11u) : 1u);
  uint iw = pw(19u);
  uint ih = (is2d ? pw(20u) : 1u);
  uint kk = (kh * kw);
  uint iic = (i0 / kk);
  uint r = (i0 % kk);
  uint ikh = (r / kw);
  uint ikw = (r % kw);
  uint ioh = (is2d ? i2 : 0u);
  uint inn = (is2d ? i3 : i2);
  uint x = ((i1 * pw(37u)) + (ikw * pw(41u)));
  uint y = ((ioh * pw(38u)) + (ikh * pw(42u)));
  if ((!(((x >= pw(39u)) && ((x - pw(39u)) < iw)) && ((y >= pw(40u)) && ((y - pw(40u)) < ih))))) {
    return 0.000000;
  }
  uint xx = (x - pw(39u));
  uint yy = (y - pw(40u));
  return s1[off4(19u, xx, (is2d ? yy : iic), (is2d ? iic : inn), (is2d ? inn : 0u))];
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
  dst[off4(1u, i0, i1, i2, i3)] = im2col_at(i0, i1, i2, i3);
}
