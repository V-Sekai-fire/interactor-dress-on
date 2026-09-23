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
RWStructuredBuffer<int> s1;
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

[shader("compute")] [numthreads(256, 1, 1)]
void main(uint3 gid : SV_GroupID, uint3 lid : SV_GroupThreadID) {
  uint e = ((((gid.y * pw(54u)) + gid.x) * 256u) + lid.x);
  if ((e >= pw(53u))) {
    return;
  }
  uint np = pw(55u);
  uint j = (e % np);
  uint r = (e / np);
  uint i1 = (r % pw(2u));
  r = (r / pw(2u));
  uint i2 = (r % pw(3u));
  uint i3 = (r / pw(3u));
  uint nh = pw(56u);
  if ((j >= nh)) {
    uint c0 = (2u * j);
    float y0 = s0[off4(10u, c0, i1, i2, i3)];
    float y1 = s0[off4(10u, (c0 + 1u), i1, i2, i3)];
    dst[off4(1u, c0, i1, i2, i3)] = y0;
    dst[off4(1u, (c0 + 1u), i1, i2, i3)] = y1;
    return;
  }
  float te = float(s1[(pw(27u) + (i2 * pw(23u)))]);
  float ts = pf(57u);
  for (uint k = 0u; k < j; ++k) {
    te = (te * ts);
  }
  float ext = pf(44u);
  float ti = (pf(43u) * te);
  float th = ti;
  if ((ext != 0.0f)) {
    float lo = pf(58u);
    float y = ((float(j) - lo) / max(0.001f, (pf(59u) - lo)));
    float rm = ((1.0f - min(1.0f, max(0.0f, y))) * ext);
    th = ((ti * (1.0f - rm)) + (te * rm));
  }
  float ms = pf(60u);
  float ct = (cos(th) * ms);
  float st = (sin(th) * ms);
  uint j1 = (j + nh);
  float x0 = s0[off4(10u, j, i1, i2, i3)];
  float x1 = s0[off4(10u, j1, i1, i2, i3)];
  dst[off4(1u, j, i1, i2, i3)] = ((x0 * ct) - (x1 * st));
  dst[off4(1u, j1, i1, i2, i3)] = ((x0 * st) + (x1 * ct));
}
