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
  uint ox;
  uint oy;
  uint oc;
  uint i3;
  unravel4(e, 1u, ox, oy, oc, i3);
  uint st = pw(37u);
  uint kw = pw(10u);
  uint kh = pw(11u);
  uint c = pw(13u);
  uint sw = pw(19u);
  uint sh = pw(20u);
  uint kn0 = pw(14u);
  uint kn1 = pw(15u);
  uint kn2 = pw(16u);
  uint kn3 = pw(17u);
  uint in0 = pw(23u);
  uint in1 = pw(24u);
  uint in2 = pw(25u);
  uint koc = (pw(18u) + (oc * kn2));
  float acc = 0.000000;
  for (uint i11 = 0u; i11 < sh; ++i11) {
    uint y = (i11 * st);
    if (((oy >= y) && ((oy - y) < kh))) {
      uint ky = (oy - y);
      for (uint i10 = 0u; i10 < sw; ++i10) {
        uint x = (i10 * st);
        if (((ox >= x) && ((ox - x) < kw))) {
          uint kx = (ox - x);
          uint ib = (pw(27u) + ((i10 * in0) + (i11 * in1)));
          uint kb = (koc + ((kx * kn0) + (ky * kn1)));
          float vdot = 0.000000;
          for (uint ic = 0u; ic < c; ++ic) {
            vdot = (vdot + (s1[(ib + (ic * in2))] * s0[(kb + (ic * kn3))]));
          }
          acc = (acc + vdot);
        }
      }
    }
  }
  dst[off4(1u, ox, oy, oc, i3)] = acc;
}
