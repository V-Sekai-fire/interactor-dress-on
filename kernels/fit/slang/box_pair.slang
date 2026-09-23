struct PairParams {
  uint n_queries;
  uint n_v;
  uint n_e;
  uint n_f;
  uint self_collision;
  uint write;
};

[[vk::binding(0, 0)]]
StructuredBuffer<uint> queries;
[[vk::binding(1, 0)]]
StructuredBuffer<float> boxes;
[[vk::binding(2, 0)]]
StructuredBuffer<uint> edges;
[[vk::binding(3, 0)]]
StructuredBuffer<uint> faces;
[[vk::binding(4, 0)]]
StructuredBuffer<uint> garment;
[[vk::binding(5, 0)]]
RWStructuredBuffer<uint> counts;
[[vk::binding(6, 0)]]
RWStructuredBuffer<uint> pairs;
[[vk::binding(7, 0)]]
ConstantBuffer<PairParams> params;

bool can_collide(uint a, uint b) {
  bool ga = (garment[a] != 0u);
  bool gb = (garment[b] != 0u);
  if ((params.self_collision != 0u)) {
    return (ga || gb);
  }
  return (ga != gb);
}

bool overlaps(uint a, uint b) {
  uint oa = (6u * a);
  uint ob = (6u * b);
  for (uint d = 0u; d < 3u; ++d) {
    if ((boxes[(oa + d)] > boxes[((ob + 3u) + d)])) {
      return false;
    }
    if ((boxes[(ob + d)] > boxes[((oa + 3u) + d)])) {
      return false;
    }
  }
  return true;
}

bool garment_face(uint f) {
  uint b = (3u * f);
  return (((garment[faces[b]] != 0u) || (garment[faces[(b + 1u)]] != 0u)) || (garment[faces[(b + 2u)]] != 0u));
}

[shader("compute")] [numthreads(64, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
  uint q = tid.x;
  if ((q >= params.n_queries)) {
    return;
  }
  uint code = queries[q];
  uint kind = (code >> 30u);
  uint id = (code & 1073741823u);
  uint c = 0u;
  uint o = ((params.write != 0u) ? counts[q] : 0u);
  if ((kind == 0u)) {
    uint a0 = edges[(2u * id)];
    uint a1 = edges[((2u * id) + 1u)];
    for (uint j = 0u; j < params.n_e; ++j) {
      if ((j != id)) {
        uint b0 = edges[(2u * j)];
        uint b1 = edges[((2u * j) + 1u)];
        bool share = (((a0 == b0) || (a0 == b1)) || ((a1 == b0) || (a1 == b1)));
        bool target_is_query = ((garment[b0] != 0u) || (garment[b1] != 0u));
        bool mine = ((!target_is_query) || (j > id));
        if ((((!share) && mine) && ((can_collide(a0, b0) || can_collide(a0, b1)) || (can_collide(a1, b0) || can_collide(a1, b1))))) {
          if (overlaps((params.n_v + id), (params.n_v + j))) {
            if ((params.write != 0u)) {
              pairs[(2u * (o + c))] = min(id, j);
              pairs[((2u * (o + c)) + 1u)] = max(id, j);
            }
            c = (c + 1u);
          }
        }
      }
    }
  } else {
    if ((kind == 1u)) {
      for (uint f = 0u; f < params.n_f; ++f) {
        uint b = (3u * f);
        uint f0 = faces[b];
        uint f1 = faces[(b + 1u)];
        uint f2 = faces[(b + 2u)];
        bool inface = (((id == f0) || (id == f1)) || (id == f2));
        if ((((!inface) && (!garment_face(f))) && ((can_collide(id, f0) || can_collide(id, f1)) || can_collide(id, f2)))) {
          if (overlaps(id, ((params.n_v + params.n_e) + f))) {
            if ((params.write != 0u)) {
              pairs[(2u * (o + c))] = f;
              pairs[((2u * (o + c)) + 1u)] = id;
            }
            c = (c + 1u);
          }
        }
      }
    } else {
      uint b = (3u * id);
      uint f0 = faces[b];
      uint f1 = faces[(b + 1u)];
      uint f2 = faces[(b + 2u)];
      for (uint w = 0u; w < params.n_v; ++w) {
        bool inface = (((w == f0) || (w == f1)) || (w == f2));
        if (((!inface) && ((can_collide(w, f0) || can_collide(w, f1)) || can_collide(w, f2)))) {
          if (overlaps(w, ((params.n_v + params.n_e) + id))) {
            if ((params.write != 0u)) {
              pairs[(2u * (o + c))] = id;
              pairs[((2u * (o + c)) + 1u)] = w;
            }
            c = (c + 1u);
          }
        }
      }
    }
  }
  if ((params.write == 0u)) {
    counts[q] = c;
  }
}
