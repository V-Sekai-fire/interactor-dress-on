// A dispatch that leaves a count behind. Each run adds one to outBuf[0], so
// after nDispatch dispatches in nSubmit submissions the buffer must read
// exactly nDispatch * nSubmit -- any lost or reordered dispatch shows as a
// wrong number rather than as a plausible one.
[[vk::binding(0, 0)]]
RWStructuredBuffer<uint> outBuf;

[shader("compute")]
[numthreads(1, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
	outBuf[0] = outBuf[0] + 1u;
}
