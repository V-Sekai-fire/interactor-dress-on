struct ProbeParams {
  uint n;
  float alpha;
};

[[vk::binding(0, 0)]]
ConstantBuffer<ProbeParams> params;
[[vk::binding(1, 0)]]
StructuredBuffer<float> src0;
[[vk::binding(2, 0)]]
StructuredBuffer<float> src1;
[[vk::binding(3, 0)]]
StructuredBuffer<float> src2;
[[vk::binding(4, 0)]]
RWStructuredBuffer<float> dst;

[shader("compute")] [numthreads(64, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
  uint i = tid.x;
  if ((i >= params.n)) {
    return;
  }
  dst[i] = (src0[i] + src1[i]);
}
