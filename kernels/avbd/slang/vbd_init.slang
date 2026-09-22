struct VbdInitParams {
  float invHSquared;
  uint colorOffset;
  uint count;
};

[[vk::binding(0, 0)]]
ConstantBuffer<VbdInitParams> params;
[[vk::binding(1, 0)]]
StructuredBuffer<float3> positions;
[[vk::binding(2, 0)]]
StructuredBuffer<float3> predicted;
[[vk::binding(3, 0)]]
StructuredBuffer<float> mass;
[[vk::binding(4, 0)]]
RWStructuredBuffer<float3> gScratch;
[[vk::binding(5, 0)]]
RWStructuredBuffer<float> hScratch;
[[vk::binding(6, 0)]]
StructuredBuffer<uint> vertPerm;

[shader("compute")] [numthreads(64, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
  uint lane = tid.x;
  if ((lane >= params.count)) {
    return;
  }
  uint v = vertPerm[(lane + params.colorOffset)];
  float3 x = positions[v];
  float3 px = predicted[v];
  float m = mass[v];
  float w = (m * params.invHSquared);
  gScratch[v] = float3((w * (x.x - px.x)), (w * (x.y - px.y)), (w * (x.z - px.z)));
  uint hb = (6u * v);
  hScratch[hb] = w;
  hScratch[(hb + 1u)] = 0.000000;
  hScratch[(hb + 2u)] = 0.000000;
  hScratch[(hb + 3u)] = w;
  hScratch[(hb + 4u)] = 0.000000;
  hScratch[(hb + 5u)] = w;
}
