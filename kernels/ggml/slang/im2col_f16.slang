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

uint f32_to_f16(float f) {
  uint u = asuint(f);
  uint s = ((u >> 16u) & 32768u);
  uint a = (u & 2147483647u);
  if ((a > 2139095040u)) {
    return (s | 32256u);
  }
  if ((a >= 1199566848u)) {
    return (s | 31744u);
  }
  if ((a >= 947912704u)) {
    uint h = ((a - 939524096u) >> 13u);
    uint r = (a & 8191u);
    if (((r > 4096u) || ((r == 4096u) && ((h & 1u) != 0u)))) {
      h = (h + 1u);
    }
    return (s | h);
  }
  if ((a < 855638017u)) {
    return s;
  }
  uint m = ((a & 8388607u) | 8388608u);
  uint sh = (126u - (a >> 23u));
  uint h = (m >> sh);
  uint r = (m & ((1u << sh) - 1u));
  uint hw = (1u << (sh - 1u));
  if (((r > hw) || ((r == hw) && ((h & 1u) != 0u)))) {
    h = (h + 1u);
  }
  return (s | h);
}

uint im2col_h(uint h, uint off, uint n, uint old) {
  if ((!((h >= off) && ((h - off) < n)))) {
    return old;
  }
  uint i0;
  uint i1;
  uint i2;
  uint i3;
  unravel4((h - off), 1u, i0, i1, i2, i3);
  return f32_to_f16(im2col_at(i0, i1, i2, i3));
}

[shader("compute")] [numthreads(256, 1, 1)]
void main(uint3 gid : SV_GroupID, uint3 lid : SV_GroupThreadID) {
  uint e = ((((gid.y * pw(54u)) + gid.x) * 256u) + lid.x);
  if ((e >= pw(53u))) {
    return;
  }
  uint off = pw(9u);
  uint n = ((pw(1u) * pw(2u)) * (pw(3u) * pw(4u)));
  uint wd = ((off >> 1u) + e);
  uint h0 = (wd * 2u);
  uint old = 0u;
  if ((!(((h0 >= off) && ((h0 - off) < n)) && (((h0 + 1u) >= off) && (((h0 + 1u) - off) < n))))) {
    old = dst[wd];
  }
  uint lo = im2col_h(h0, off, n, (old & 65535u));
  uint hi = im2col_h((h0 + 1u), off, n, (old >> 16u));
  dst[wd] = (lo | (hi << 16u));
}
