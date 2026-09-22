struct VbdInitBackwardParams {
  float invHSquared;
};

[[vk::binding(0, 0)]]
StructuredBuffer<float3> positions;
[[vk::binding(1, 0)]]
StructuredBuffer<float3> predicted;
[[vk::binding(2, 0)]]
StructuredBuffer<float> mass;
[[vk::binding(3, 0)]]
StructuredBuffer<float3> v_g;
[[vk::binding(4, 0)]]
StructuredBuffer<float> v_H;
[[vk::binding(5, 0)]]
RWStructuredBuffer<float3> v_x;
[[vk::binding(6, 0)]]
RWStructuredBuffer<float3> v_y;
[[vk::binding(7, 0)]]
RWStructuredBuffer<float> v_mass;
[[vk::binding(8, 0)]]
ConstantBuffer<VbdInitBackwardParams> params;

[shader("compute")] [numthreads(64, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
  uint v = tid.x;
  uint hb = (6u * v);
  float3 x = positions[v];
  float3 y = predicted[v];
  float m = mass[v];
  float3 vg = v_g[v];
  float w = (m * params.invHSquared);
  v_x[v] = float3((w * vg.x), (w * vg.y), (w * vg.z));
  v_y[v] = float3(((0.000000 - w) * vg.x), ((0.000000 - w) * vg.y), ((0.000000 - w) * vg.z));
  float dot_dx_vg = ((((x.x - y.x) * vg.x) + ((x.y - y.y) * vg.y)) + ((x.z - y.z) * vg.z));
  float vHxx = v_H[hb];
  float vHyy = v_H[(hb + 3u)];
  float vHzz = v_H[(hb + 5u)];
  float trH = ((vHxx + vHyy) + vHzz);
  v_mass[v] = (params.invHSquared * (dot_dx_vg + trH));
}
