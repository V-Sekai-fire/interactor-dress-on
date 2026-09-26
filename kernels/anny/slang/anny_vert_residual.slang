struct AnnyResidualParams {
  uint n;
};

[[vk::binding(0, 0)]]
ConstantBuffer<AnnyResidualParams> params;
[[vk::binding(1, 0)]]
StructuredBuffer<float> verts;
[[vk::binding(2, 0)]]
StructuredBuffer<float> target;
[[vk::binding(3, 0)]]
RWStructuredBuffer<float> dverts;

[shader("compute")] [numthreads(256, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
  uint i = tid.x;
  if ((i >= params.n)) {
    return;
  }
  dverts[i] = (verts[i] - target[i]);
}
