struct RdpParams {
  uint in_count;
  float tolerance;
};

[[vk::binding(0, 0)]]
ConstantBuffer<RdpParams> params;
[[vk::binding(1, 0)]]
StructuredBuffer<float3> in_points;
[[vk::binding(2, 0)]]
RWStructuredBuffer<uint> out_keep;
[[vk::binding(3, 0)]]
RWStructuredBuffer<uint> out_count;

[shader("compute")] [numthreads(1, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
  uint stack[512];
  uint stack_top = 0u;
  uint kept = 0u;
  for (uint i = 0u; i < params.in_count; ++i) {
    out_keep[i] = 0u;
  }
  if ((params.in_count >= 1u)) {
    out_keep[0u] = 1u;
  }
  if ((params.in_count >= 2u)) {
    out_keep[(params.in_count - 1u)] = 1u;
  }
  if ((params.in_count >= 3u)) {
    stack[0u] = 0u;
    stack[1u] = (params.in_count - 1u);
    stack_top = 1u;
  }
  while ((stack_top > 0u)) {
    stack_top = (stack_top - 1u);
    uint first = stack[(stack_top * 2u)];
    uint last = stack[((stack_top * 2u) + 1u)];
    if (((last - first) > 1u)) {
      float3 a = in_points[first];
      float3 b = in_points[last];
      float3 ab = float3((b.x - a.x), (b.y - a.y), (b.z - a.z));
      float ab_len2 = dot(ab, ab);
      float max_dist = params.tolerance;
      uint split = 0u;
      for (uint j = (first + 1u); j < last; ++j) {
        float3 p = in_points[j];
        float3 ap = float3((p.x - a.x), (p.y - a.y), (p.z - a.z));
        float3 bp = float3((p.x - b.x), (p.y - b.y), (p.z - b.z));
        float d = ((ab_len2 <= 0.000000) ? length(ap) : sqrt((dot(cross(ap, bp), cross(ap, bp)) / ab_len2)));
        if ((d > max_dist)) {
          max_dist = d;
          split = j;
        }
      }
      if ((split != 0u)) {
        out_keep[split] = 1u;
        if ((stack_top < 255u)) {
          stack[(stack_top * 2u)] = first;
          stack[((stack_top * 2u) + 1u)] = split;
          stack_top = (stack_top + 1u);
        }
        if ((stack_top < 255u)) {
          stack[(stack_top * 2u)] = split;
          stack[((stack_top * 2u) + 1u)] = last;
          stack_top = (stack_top + 1u);
        }
      }
    }
  }
  for (uint i = 0u; i < params.in_count; ++i) {
    if ((out_keep[i] == 1u)) {
      kept = (kept + 1u);
    }
  }
  out_count[0u] = kept;
  return;
}
