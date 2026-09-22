// The smallest possible proof that a compute dispatch reached the GPU.
//
// Writes a recognisable constant. Not zero, and not a small integer that
// could plausibly be uninitialised memory or an off-by-one: if the readback
// says 0xC0FFEE the kernel ran, and if it says anything else it did not.
[[vk::binding(0, 0)]]
RWStructuredBuffer<uint> outBuf;

[shader("compute")]
[numthreads(1, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
	outBuf[0] = 0x00C0FFEEu;
}
