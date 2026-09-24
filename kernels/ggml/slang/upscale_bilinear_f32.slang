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

uint src_idx(uint i, uint nd, uint ns) {
  return ((i * ns) / nd);
}

int clamp_idx(int i, int hi) {
  int m = ((hi < i) ? hi : i);
  return ((m < 0) ? 0 : m);
}

float clamp01(float t) {
  float m = ((1.0f < t) ? 1.0f : t);
  return ((0.0f < m) ? m : 0.0f);
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
  bool ac = ((pw(37u) & 256u) != 0u);
  float po = (ac ? 0.0f : 0.5f);
  float sf0 = ((ac && ((pw(1u) > 1u) && (pw(10u) > 1u))) ? (float((pw(1u) - 1u)) / float((pw(10u) - 1u))) : (float(pw(1u)) / float(pw(10u))));
  float sf1 = ((ac && ((pw(2u) > 1u) && (pw(11u) > 1u))) ? (float((pw(2u) - 1u)) / float((pw(11u) - 1u))) : (float(pw(2u)) / float(pw(11u))));
  float yc = (((float(i1) + po) / sf1) - po);
  int y0 = clamp_idx(int(floor(yc)), int((pw(11u) - 1u)));
  int y1 = clamp_idx((int(floor(yc)) + 1), int((pw(11u) - 1u)));
  float dy = clamp01((yc - float(y0)));
  float xc = (((float(i0) + po) / sf0) - po);
  int x0 = clamp_idx(int(floor(xc)), int((pw(10u) - 1u)));
  int x1 = clamp_idx((int(floor(xc)) + 1), int((pw(10u) - 1u)));
  float dx = clamp01((xc - float(x0)));
  uint i02 = src_idx(i2, pw(3u), pw(12u));
  uint i03 = src_idx(i3, pw(4u), pw(13u));
  float a = s0[off4(10u, uint(x0), uint(y0), i02, i03)];
  float b = s0[off4(10u, uint(x1), uint(y0), i02, i03)];
  float c = s0[off4(10u, uint(x0), uint(y1), i02, i03)];
  float dd = s0[off4(10u, uint(x1), uint(y1), i02, i03)];
  dst[d] = (((((a * (1.0f - dx)) * (1.0f - dy)) + ((b * dx) * (1.0f - dy))) + ((c * (1.0f - dx)) * dy)) + ((dd * dx) * dy));
}
