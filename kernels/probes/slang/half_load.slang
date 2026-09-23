[[vk::binding(0, 0)]]
StructuredBuffer<half> w;
[[vk::binding(1, 0)]]
RWStructuredBuffer<float> o;

[shader("compute")] [numthreads(64, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
  uint i = tid.x;
  o[i] = (float(w[i]) * 2.000000);
  return;
}
