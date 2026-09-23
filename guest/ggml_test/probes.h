// ggml-rd probes for Gate 3, each a pump job that prints its numbers and a
// final "RESULT: PASS" or "RESULT: FAIL" line:
//
//   chain <n>        n in-place ADDs (x += 1) on one tensor in one ggml
//                    buffer, one graph: every dispatch reads what the one
//                    before wrote, in the same RD buffer. x must be exactly n
//                    and the graph must have n - 1 barriers. The ordering
//                    Gate 0F lost (80% of increments) with read-only sources.
//   independent <n>  n ADDs into n separate outputs of one buffer, one graph:
//                    no dispatch touches another's bytes, so barrier elision
//                    records 0 barriers (n - 1 under GGML_RD_BARRIER_ALL=1),
//                    and every output must be exact.
//   files <path>     the READ and UPLOAD requests: READ a host file of f32s,
//                    UPLOAD the same file into a tensor's RD buffer, read the
//                    tensor back, and compute x + x on the GPU: both exact.
//   alias <rw|ro>    the render-graph aliasing hazard (Gate 0F finding 4), as
//                    an A/B on one recording: x += 1 in place, 1000 times,
//                    4096 elements, one compute list with a barrier between
//                    every two dispatches, x bound at b1 and b4 of one set
//                    (one ggml buffer is one RD buffer). The words come from
//                    ggml-rd's own packer. rw uses add_f32, whose sources are
//                    read-write: every element must end at exactly 1000. ro
//                    uses the control ctl_add_f32_rosrc, identical but for
//                    read-only sources: Godot records each span's usage of
//                    the buffer from its first binding (a read), does not
//                    order the spans, and increments are lost; the probe
//                    PASSes when they are (the hazard is still there, and
//                    read-write sources are what avoid it).
//   fa_perf <lq,lk,reps>  FLASH_ATTN_EXT at a census shape (D = 128, 12
//                    heads, f32, no mask): a one-node graph computed reps
//                    times, one submit and one WAIT_GPU each (the host times
//                    the period), then 8 sampled query rows checked against
//                    a double reference.
#pragma once

#include <string>

namespace rdc {
class Device;
}

namespace probes {

// The device the alias probe records on directly (the one ggml-rd uses).
void set_device(rdc::Device *dev);
bool start(const std::string &name, const std::string &arg, std::string &err);

} // namespace probes
