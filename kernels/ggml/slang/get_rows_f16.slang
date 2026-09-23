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
RWStructuredBuffer<uint> dst;
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

uint ld16_s0(uint e) {
  return ((s0[(e >> 1u)] >> ((e & 1u) * 16u)) & 65535u);
}

uint f16_to_f32(uint h) {
  uint s = ((h & 32768u) << 16u);
  uint x = ((h >> 10u) & 31u);
  uint m = (h & 1023u);
  if ((x == 31u)) {
    return ((s | 2139095040u) | (m << 13u));
  }
  if ((x != 0u)) {
    return ((s | ((x + 112u) << 23u)) | (m << 13u));
  }
  if ((m == 0u)) {
    return s;
  }
  uint x2 = 113u;
  while (((m & 1024u) == 0u)) {
    m = (m << 1u);
    x2 = (x2 - 1u);
  }
  return ((s | (x2 << 23u)) | ((m & 1023u) << 13u));
}

uint val(uint i0, uint i1, uint i2, uint i3) {
  uint r = s1[off4(19u, i1, i2, i3, 0u)];
  return f16_to_f32(ld16_s0(off4(10u, i0, r, i2, i3)));
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
  dst[off4(1u, i0, i1, i2, i3)] = val(i0, i1, i2, i3);
}
