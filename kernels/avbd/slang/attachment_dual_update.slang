struct AttachmentDualUpdateParams {
  float beta;
  float penaltyMax;
  uint count;
};

[[vk::binding(0, 0)]]
StructuredBuffer<float3> positions;
[[vk::binding(1, 0)]]
StructuredBuffer<uint> vertIdx;
[[vk::binding(2, 0)]]
StructuredBuffer<float3> fixedPos;
[[vk::binding(3, 0)]]
RWStructuredBuffer<float> gamma;
[[vk::binding(4, 0)]]
RWStructuredBuffer<float3> lambda;
[[vk::binding(5, 0)]]
ConstantBuffer<AttachmentDualUpdateParams> params;

[shader("compute")] [numthreads(64, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
  uint c = tid.x;
  if ((c >= params.count)) {
    return;
  }
  uint v = vertIdx[c];
  float3 p = positions[v];
  float3 fp = fixedPos[c];
  float g = gamma[c];
  float3 lam = lambda[c];
  float Cx = (p.x - fp.x);
  float Cy = (p.y - fp.y);
  float Cz = (p.z - fp.z);
  lambda[c] = float3((lam.x + (g * Cx)), (lam.y + (g * Cy)), (lam.z + (g * Cz)));
  float Cmag = sqrt((((Cx * Cx) + (Cy * Cy)) + (Cz * Cz)));
  gamma[c] = min((g + (params.beta * Cmag)), params.penaltyMax);
}
