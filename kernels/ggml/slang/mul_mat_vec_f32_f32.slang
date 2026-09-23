struct Slot {
  uint base;
  uint pad0;
  uint pad1;
  uint pad2;
};

groupshared float red[32][32];

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

float lda(uint e) {
  return s0[e];
}

float ldb(uint e) {
  return s1[e];
}

[shader("compute")] [numthreads(32, 8, 1)]
void main(uint3 gid : SV_GroupID, uint3 lid : SV_GroupThreadID) {
  uint K = pw(10u);
  uint M = pw(11u);
  uint N = pw(20u);
  uint ne12 = pw(21u);
  uint b2 = (gid.z % ne12);
  uint b3 = (gid.z / ne12);
  uint aBase = ((pw(18u) + ((b2 / pw(55u)) * pw(16u))) + ((b3 / pw(56u)) * pw(17u)));
  uint bBase = ((pw(27u) + (b2 * pw(25u))) + (b3 * pw(26u)));
  uint dBase = ((pw(9u) + (b2 * pw(7u))) + (b3 * pw(8u)));
  uint ak = pw(14u);
  uint am = pw(15u);
  uint bk = pw(23u);
  uint bn = pw(24u);
  uint dm = pw(5u);
  uint dn = pw(6u);
  uint row = ((((gid.y * pw(57u)) + gid.x) * 8u) + lid.y);
  float p0 = 0.0f;
  float p1 = 0.0f;
  float p2 = 0.0f;
  float p3 = 0.0f;
  uint lane = lid.x;
  uint ry = (lid.y * 4u);
  if ((row < M)) {
    uint k = lane;
    while ((k < K)) {
      float a = lda(((aBase + (k * ak)) + (row * am)));
      uint bo = (bBase + (k * bk));
      p0 = (p0 + (a * ldb(bo)));
      if ((N > 1u)) {
        p1 = (p1 + (a * ldb((bo + bn))));
      }
      if ((N > 2u)) {
        p2 = (p2 + (a * ldb((bo + (2u * bn)))));
      }
      if ((N > 3u)) {
        p3 = (p3 + (a * ldb((bo + (3u * bn)))));
      }
      k = (k + 32u);
    }
  }
  red[ry][lane] = p0;
  red[(ry + 1u)][lane] = p1;
  red[(ry + 2u)][lane] = p2;
  red[(ry + 3u)][lane] = p3;
  GroupMemoryBarrierWithGroupSync();
  if ((lane < 16u)) {
    red[ry][lane] = (red[ry][lane] + red[ry][(lane + 16u)]);
    red[(ry + 1u)][lane] = (red[(ry + 1u)][lane] + red[(ry + 1u)][(lane + 16u)]);
    red[(ry + 2u)][lane] = (red[(ry + 2u)][lane] + red[(ry + 2u)][(lane + 16u)]);
    red[(ry + 3u)][lane] = (red[(ry + 3u)][lane] + red[(ry + 3u)][(lane + 16u)]);
  }
  GroupMemoryBarrierWithGroupSync();
  if ((lane < 8u)) {
    red[ry][lane] = (red[ry][lane] + red[ry][(lane + 8u)]);
    red[(ry + 1u)][lane] = (red[(ry + 1u)][lane] + red[(ry + 1u)][(lane + 8u)]);
    red[(ry + 2u)][lane] = (red[(ry + 2u)][lane] + red[(ry + 2u)][(lane + 8u)]);
    red[(ry + 3u)][lane] = (red[(ry + 3u)][lane] + red[(ry + 3u)][(lane + 8u)]);
  }
  GroupMemoryBarrierWithGroupSync();
  if ((lane < 4u)) {
    red[ry][lane] = (red[ry][lane] + red[ry][(lane + 4u)]);
    red[(ry + 1u)][lane] = (red[(ry + 1u)][lane] + red[(ry + 1u)][(lane + 4u)]);
    red[(ry + 2u)][lane] = (red[(ry + 2u)][lane] + red[(ry + 2u)][(lane + 4u)]);
    red[(ry + 3u)][lane] = (red[(ry + 3u)][lane] + red[(ry + 3u)][(lane + 4u)]);
  }
  GroupMemoryBarrierWithGroupSync();
  if ((lane < 2u)) {
    red[ry][lane] = (red[ry][lane] + red[ry][(lane + 2u)]);
    red[(ry + 1u)][lane] = (red[(ry + 1u)][lane] + red[(ry + 1u)][(lane + 2u)]);
    red[(ry + 2u)][lane] = (red[(ry + 2u)][lane] + red[(ry + 2u)][(lane + 2u)]);
    red[(ry + 3u)][lane] = (red[(ry + 3u)][lane] + red[(ry + 3u)][(lane + 2u)]);
  }
  GroupMemoryBarrierWithGroupSync();
  if ((lane < 1u)) {
    red[ry][lane] = (red[ry][lane] + red[ry][(lane + 1u)]);
    red[(ry + 1u)][lane] = (red[(ry + 1u)][lane] + red[(ry + 1u)][(lane + 1u)]);
    red[(ry + 2u)][lane] = (red[(ry + 2u)][lane] + red[(ry + 2u)][(lane + 1u)]);
    red[(ry + 3u)][lane] = (red[(ry + 3u)][lane] + red[(ry + 3u)][(lane + 1u)]);
  }
  GroupMemoryBarrierWithGroupSync();
  if (((lane < N) && (row < M))) {
    dst[((dBase + (row * dm)) + (lane * dn))] = red[(ry + lane)][0u];
  }
}
