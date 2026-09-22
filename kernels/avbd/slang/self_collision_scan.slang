struct SelfCollisionScanParams {
  uint nVerts;
  uint K;
};

[[vk::binding(0, 0)]]
StructuredBuffer<float3> positions;
[[vk::binding(1, 0)]]
StructuredBuffer<float> radii;
[[vk::binding(2, 0)]]
RWStructuredBuffer<uint> neighbors;
[[vk::binding(3, 0)]]
ConstantBuffer<SelfCollisionScanParams> params;

[shader("compute")] [numthreads(64, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
  uint i = tid.x;
  if ((i >= params.nVerts)) {
    return;
  }
  for (uint k = 0u; k < params.K; ++k) {
    neighbors[((i * params.K) + k)] = (0u - 1u);
  }
  float3 pi = positions[i];
  float ri = radii[i];
  uint count = 0u;
  for (uint j = 0u; j < params.nVerts; ++j) {
    float3 dj = positions[j];
    float dx = (pi.x - dj.x);
    float dy = (pi.y - dj.y);
    float dz = (pi.z - dj.z);
    float d2 = (((dx * dx) + (dy * dy)) + (dz * dz));
    float rj = radii[j];
    float tr = (ri + rj);
    float t2 = (tr * tr);
    if ((((j != i) && (count < params.K)) && (d2 < t2))) {
      neighbors[((i * params.K) + count)] = j;
      count = (count + 1u);
    }
  }
}
