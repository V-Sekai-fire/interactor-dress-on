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
  uint oz;
  uint ocn;
  unravel4(e, 1u, ox, oy, oz, ocn);
  uint c = pw(46u);
  uint oc = pw(48u);
  uint sx = pw(37u);
  uint sy = pw(38u);
  uint sz = pw(39u);
  uint px = pw(40u);
  uint py = pw(41u);
  uint pz = pw(42u);
  uint dx = pw(43u);
  uint dy = pw(44u);
  uint dz = pw(45u);
  uint kw = pw(10u);
  uint kh = pw(11u);
  uint kd = pw(12u);
  uint iw = pw(19u);
  uint ih = pw(20u);
  uint idd = pw(21u);
  uint kn0 = pw(14u);
  uint kn1 = pw(15u);
  uint kn2 = pw(16u);
  uint kn3 = pw(17u);
  uint in0 = pw(23u);
  uint in1 = pw(24u);
  uint in2 = pw(25u);
  uint in3 = pw(26u);
  uint ib = (pw(27u) + (((ocn / oc) * c) * in3));
  uint kb = (pw(18u) + (((ocn % oc) * c) * kn3));
  float acc = 0.000000;
  for (uint ic = 0u; ic < c; ++ic) {
    uint ibc = (ib + (ic * in3));
    uint kbc = (kb + (ic * kn3));
    for (uint kz = 0u; kz < kd; ++kz) {
      uint z = ((oz * sz) + (kz * dz));
      if (((z >= pz) && ((z - pz) < idd))) {
        uint ibz = (ibc + ((z - pz) * in2));
        uint kbz = (kbc + (kz * kn2));
        for (uint ky = 0u; ky < kh; ++ky) {
          uint y = ((oy * sy) + (ky * dy));
          if (((y >= py) && ((y - py) < ih))) {
            uint iby = (ibz + ((y - py) * in1));
            uint kby = (kbz + (ky * kn1));
            for (uint kx = 0u; kx < kw; ++kx) {
              uint x = ((ox * sx) + (kx * dx));
              if (((x >= px) && ((x - px) < iw))) {
                acc = (acc + (s1[(iby + ((x - px) * in0))] * s0[(kby + (kx * kn0))]));
              }
            }
          }
        }
      }
    }
  }
  dst[off4(1u, ox, oy, oz, ocn)] = acc;
}
