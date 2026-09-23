// SPDX-License-Identifier: MIT
//
// av1mkv: a CineForm (CFHD in AVI, as entities-godot-cineform's MovieWriter writes it) to
// AV1-in-Matroska transcoder, and an mkvparser-based inspector for the result.
//
//   av1mkv encode <in.cfhd> <out.mkv> [--cq 22] [--gop 60] [--gpu "RTX 4090"] [--frames N]
//   av1mkv info <file.mkv>
//   av1mkv dump-frame <in.cfhd> <index> <out.ppm>
//
// Decode: V-Sekai-fire/cineform-sdk, CFHD_OpenDecoder / CFHD_PrepareToDecode /
// CFHD_DecodeSample to 8-bit BGRA (the 12-bit 4:4:4 source is rounded to 8 bits here; the
// encoder is 8-bit 4:2:0). Encode: NVENC AV1 through the NVIDIA driver (nvenc_av1.cpp).
// Mux: V-Sekai-fire/libwebm mkvmuxer, V_AV1 with an av1C CodecPrivate built from the
// sequence header OBU of the first key frame (av1c.h), plus the recording's PCM track as
// A_PCM/INT/LIT. No FFmpeg anywhere.

#include "avi_reader.h"
#include "av1c.h"
#include "nvenc_av1.h"

#include <CFHDDecoder.h>
#include <CFHDTypes.h>

#include <mkvmuxer/mkvmuxer.h>
#include <mkvmuxer/mkvwriter.h>

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

int mkv_info(const char* path); // mkv_info.cpp

namespace {

using Clock = std::chrono::steady_clock;
double ms_since(Clock::time_point t) { return std::chrono::duration<double, std::milli>(Clock::now() - t).count(); }

int fail(const std::string& why)
{
    std::fprintf(stderr, "av1mkv: %s\n", why.c_str());
    return 1;
}

struct CfhdDecoder {
    CFHD_DecoderRef ref = nullptr;
    int width = 0, height = 0;
    CFHD_PixelFormat format = CFHD_PIXEL_FORMAT_BGRA;
    int32_t pitch = 0;
    std::vector<uint8_t> frame;

    std::string open(const AviMovie& m)
    {
        CFHD_Error err = CFHD_OpenDecoder(&ref, nullptr);
        if (err != CFHD_ERROR_OKAY) return "CFHD_OpenDecoder failed, code " + std::to_string(int(err));
        const AviChunk& first = m.video[0];
        int aw = 0, ah = 0;
        CFHD_PixelFormat af = format;
        err = CFHD_PrepareToDecode(ref, int(m.width), int(m.height), format, CFHD_DECODED_RESOLUTION_FULL, CFHD_DECODING_FLAGS_NONE,
                                   const_cast<uint8_t*>(m.data(first)), first.size, &aw, &ah, &af);
        if (err != CFHD_ERROR_OKAY) return "CFHD_PrepareToDecode failed, code " + std::to_string(int(err));
        width = aw;
        height = ah;
        format = af;
        CFHD_GetImagePitch(uint32_t(width), format, &pitch);
        frame.resize(size_t(pitch) * size_t(height));
        return std::string();
    }
    std::string decode(const AviMovie& m, size_t i)
    {
        const AviChunk& c = m.video[i];
        const CFHD_Error err = CFHD_DecodeSample(ref, const_cast<uint8_t*>(m.data(c)), c.size, frame.data(), pitch);
        if (err != CFHD_ERROR_OKAY) return "CFHD_DecodeSample failed on frame " + std::to_string(i) + ", code " + std::to_string(int(err));
        return std::string();
    }
    ~CfhdDecoder() { if (ref) CFHD_CloseDecoder(ref); }
};

// The decoder's BGRA is the Windows DIB convention, bottom row first (the MovieWriter
// flipped Godot's top-down rows into it, see movie_writer_cineform.cpp). NVENC wants
// top-down, so the rows are handed over through a reversed pitch: pointer to the last row
// and a walk upward. nvenc_av1 copies row by row, so a negative stride is expressed as a
// separate top-down buffer here; the copy is one memcpy per row either way.
void flip_rows(const std::vector<uint8_t>& src, int32_t pitch, int height, std::vector<uint8_t>& dst)
{
    dst.resize(src.size());
    for (int y = 0; y < height; y++)
        std::memcpy(dst.data() + size_t(pitch) * y, src.data() + size_t(pitch) * (height - 1 - y), size_t(pitch));
}

int dump_frame(const char* in, size_t index, const char* out)
{
    AviMovie m;
    std::string e = avi_read(in, m);
    if (!e.empty()) return fail(e);
    if (index >= m.video.size()) return fail("frame index out of range");
    CfhdDecoder dec;
    if (!(e = dec.open(m)).empty()) return fail(e);
    if (!(e = dec.decode(m, index)).empty()) return fail(e);
    std::vector<uint8_t> top;
    flip_rows(dec.frame, dec.pitch, dec.height, top);
    std::FILE* f = std::fopen(out, "wb");
    if (!f) return fail("cannot write ppm");
    std::fprintf(f, "P6\n%d %d\n255\n", dec.width, dec.height);
    for (int y = 0; y < dec.height; y++) {
        const uint8_t* row = top.data() + size_t(dec.pitch) * y;
        for (int x = 0; x < dec.width; x++) {
            const uint8_t rgb[3] = {row[x * 4 + 2], row[x * 4 + 1], row[x * 4 + 0]};
            std::fwrite(rgb, 1, 3, f);
        }
    }
    std::fclose(f);
    std::printf("wrote %s (%dx%d, frame %zu of %zu)\n", out, dec.width, dec.height, index, m.video.size());
    return 0;
}

int encode(int argc, char** argv)
{
    if (argc < 4) return fail("usage: av1mkv encode <in.cfhd> <out.mkv> [--cq N] [--gop N] [--gpu NAME] [--frames N]");
    const char* in = argv[2];
    const char* out = argv[3];
    NvencAv1Settings s;
    size_t max_frames = 0;
    for (int i = 4; i + 1 < argc; i += 2) {
        if (!std::strcmp(argv[i], "--cq")) s.cq = uint32_t(std::atoi(argv[i + 1]));
        else if (!std::strcmp(argv[i], "--gop")) s.gop = uint32_t(std::atoi(argv[i + 1]));
        else if (!std::strcmp(argv[i], "--gpu")) s.adapter_name_contains = argv[i + 1];
        else if (!std::strcmp(argv[i], "--frames")) max_frames = size_t(std::atoll(argv[i + 1]));
        else return fail(std::string("unknown option ") + argv[i]);
    }

    const auto t_all = Clock::now();
    AviMovie m;
    std::string e = avi_read(in, m);
    if (!e.empty()) return fail(e);
    std::printf("input: %s, %zu bytes, %ux%u, %u/%u fps, handler %s, %zu video chunks, %zu audio chunks",
                in, m.bytes.size(), m.width, m.height, m.fps_num, m.fps_den, m.handler, m.video.size(), m.audio.size());
    if (m.has_audio) std::printf(", audio %u Hz x%u %u-bit PCM", m.mix_rate, m.channels, m.bits);
    std::printf("\n");
    if (std::strcmp(m.handler, "CFHD")) return fail("the video handler is not CFHD");

    CfhdDecoder dec;
    if (!(e = dec.open(m)).empty()) return fail(e);
    if (dec.format != CFHD_PIXEL_FORMAT_BGRA) return fail("the decoder did not give BGRA");
    if (uint32_t(dec.width) != m.width || uint32_t(dec.height) != m.height) return fail("decoded size differs from the header");

    s.width = m.width;
    s.height = m.height;
    s.fps_num = m.fps_num;
    s.fps_den = m.fps_den;
    NvencAv1 enc;
    if (!(e = enc.open(s)).empty()) return fail(e);
    std::printf("encoder: NVENC AV1 on %s, preset P6, tuning high quality, VBR constant quality cq=%u, two-pass full "
                "resolution, spatial AQ, no B-frames, no lookahead, 8-bit 4:2:0, key frame every %u frames\n",
                enc.adapter_description().c_str(), s.cq, s.gop);

    mkvmuxer::MkvWriter writer;
    if (!writer.Open(out)) return fail(std::string("cannot open ") + out + " for writing");
    mkvmuxer::Segment seg;
    if (!seg.Init(&writer)) return fail("mkvmuxer Segment::Init failed");
    seg.set_mode(mkvmuxer::Segment::kFile);
    seg.OutputCues(true);
    seg.GetSegmentInfo()->set_writing_app("av1mkv (interactor-dress-on)");
    seg.GetSegmentInfo()->set_muxing_app("libwebm mkvmuxer (V-Sekai-fire/libwebm)");
    const uint64_t vtrack = seg.AddVideoTrack(int(m.width), int(m.height), 0);
    if (!vtrack) return fail("AddVideoTrack failed");
    auto* video = static_cast<mkvmuxer::VideoTrack*>(seg.GetTrackByNumber(vtrack));
    video->set_codec_id(mkvmuxer::Tracks::kAv1CodecId);
    video->set_frame_rate(double(m.fps_num) / double(m.fps_den));
    // Default duration of one frame in ns, so a player knows the frame period.
    video->set_default_duration(uint64_t(1000000000ull * m.fps_den / m.fps_num));
    seg.CuesTrack(vtrack);
    uint64_t atrack = 0;
    if (m.has_audio && !m.audio.empty() && m.bits == 16) {
        atrack = seg.AddAudioTrack(int(m.mix_rate), int(m.channels), 0);
        if (!atrack) return fail("AddAudioTrack failed");
        auto* audio = static_cast<mkvmuxer::AudioTrack*>(seg.GetTrackByNumber(atrack));
        audio->set_codec_id("A_PCM/INT/LIT");
        audio->set_bit_depth(m.bits);
    }

    const uint64_t frame_ns = 1000000000ull * m.fps_den / m.fps_num;
    const size_t frames = max_frames ? std::min(max_frames, m.video.size()) : m.video.size();
    size_t audio_next = 0;
    uint64_t audio_samples = 0;
    const uint32_t audio_block = m.channels * (m.bits / 8);
    bool have_private = false;
    size_t packets = 0, keyframes = 0;
    uint64_t out_bytes = 0;
    std::string mux_error;

    // Audio chunks up to the video frame's time go first: mkvmuxer wants the segment's
    // timestamps monotonic across tracks.
    auto flush_audio_to = [&](uint64_t ts_ns) {
        while (atrack && audio_next < m.audio.size()) {
            const uint64_t ts = audio_samples * 1000000000ull / m.mix_rate;
            if (ts > ts_ns) break;
            const AviChunk& c = m.audio[audio_next];
            if (!seg.AddFrame(m.data(c), c.size, atrack, ts, true)) { mux_error = "AddFrame (audio) failed"; return; }
            audio_samples += audio_block ? c.size / audio_block : 0;
            audio_next++;
        }
    };

    auto sink = [&](const NvencPacket& pkt) {
        if (!mux_error.empty()) return;
        std::vector<av1::Obu> obus;
        if (!av1::walk(pkt.data.data(), pkt.data.size(), obus)) { mux_error = "malformed OBU stream from NVENC"; return; }
        // The Matroska AV1 mapping: Blocks hold a Temporal Unit without its temporal
        // delimiter; the sequence header stays (it is on every key frame here).
        std::vector<uint8_t> block;
        block.reserve(pkt.data.size());
        for (const auto& o : obus) {
            if (o.type == av1::OBU_TEMPORAL_DELIMITER) continue;
            if (o.type == av1::OBU_SEQUENCE_HEADER && !have_private) {
                av1::SeqInfo info;
                if (!av1::parse_sequence_header(o.payload, o.payload_length, info)) { mux_error = "cannot parse the sequence header"; return; }
                const std::vector<uint8_t> av1c = av1::build_av1c(info, o);
                if (!video->SetCodecPrivate(av1c.data(), av1c.size())) { mux_error = "SetCodecPrivate failed"; return; }
                have_private = true;
                std::printf("av1C: profile %u level %u tier %u, %s-bit, %s, subsampling %ux%u, %ux%u max, %zu bytes\n",
                            info.profile, info.level, info.tier, info.high_bitdepth ? (info.twelve_bit ? "12" : "10") : "8",
                            info.mono ? "mono" : "colour", info.ss_x, info.ss_y, info.max_width, info.max_height, av1c.size());
            }
            block.insert(block.end(), o.start, o.start + o.length);
        }
        if (!have_private) { mux_error = "the first packet has no sequence header"; return; }
        const uint64_t ts = pkt.pts * frame_ns;
        flush_audio_to(ts);
        if (!mux_error.empty()) return;
        if (!seg.AddFrame(block.data(), block.size(), vtrack, ts, pkt.keyframe)) { mux_error = "AddFrame (video) failed"; return; }
        packets++;
        out_bytes += block.size();
        if (pkt.keyframe) keyframes++;
    };

    double decode_ms = 0, flip_ms = 0;
    std::vector<uint8_t> top;
    const auto t_loop = Clock::now();
    for (size_t i = 0; i < frames; i++) {
        auto t = Clock::now();
        if (!(e = dec.decode(m, i)).empty()) return fail(e);
        decode_ms += ms_since(t);
        t = Clock::now();
        flip_rows(dec.frame, dec.pitch, dec.height, top);
        flip_ms += ms_since(t);
        if (!(e = enc.encode(top.data(), uint32_t(dec.pitch), i, sink)).empty()) return fail(e);
        if (!mux_error.empty()) return fail(mux_error);
        if ((i + 1) % 100 == 0) { std::printf("  %zu/%zu frames, %.1f s\r", i + 1, frames, ms_since(t_loop) / 1000.0); std::fflush(stdout); }
    }
    if (!(e = enc.finish(sink)).empty()) return fail(e);
    if (!mux_error.empty()) return fail(mux_error);
    // Audio that lands after the last video frame's time (there is one chunk per frame, so
    // normally none is left).
    flush_audio_to(~0ull);
    if (!mux_error.empty()) return fail(mux_error);
    const double loop_ms = ms_since(t_loop);
    const double dur_s = double(frames) * double(m.fps_den) / double(m.fps_num);
    if (!seg.Finalize()) return fail("mkvmuxer Finalize failed");
    writer.Close();
    const double all_ms = ms_since(t_all);

    std::FILE* f = std::fopen(out, "rb");
    long long out_size = 0;
    if (f) { _fseeki64(f, 0, SEEK_END); out_size = _ftelli64(f); std::fclose(f); }
    std::printf("\nwrote %s: %lld bytes (%.1f%% of the input), %zu frames (%zu key), %zu audio chunks, %.2f s of video\n", out,
                out_size, 100.0 * double(out_size) / double(m.bytes.size()), packets, keyframes, audio_next, dur_s);
    std::printf("time: %.2f s total wall, %.2f s decode+encode loop = %.1f fps (%.1fx realtime); per frame: CFHD decode %.2f ms, "
                "row flip %.2f ms, NVENC copy-in %.2f ms, submit %.2f ms, lock/readback %.2f ms\n",
                all_ms / 1000.0, loop_ms / 1000.0, double(frames) * 1000.0 / loop_ms, dur_s * 1000.0 / loop_ms, decode_ms / double(frames),
                flip_ms / double(frames), enc.copy_ms() / double(frames), enc.submit_ms() / double(frames), enc.lock_ms() / double(frames));
    std::printf("video bitrate: %.2f Mbit/s\n", double(out_bytes) * 8.0 / dur_s / 1e6);
    return 0;
}

} // namespace

int main(int argc, char** argv)
{
    if (argc >= 2 && !std::strcmp(argv[1], "encode")) return encode(argc, argv);
    if (argc >= 3 && !std::strcmp(argv[1], "info")) return mkv_info(argv[2]);
    if (argc >= 5 && !std::strcmp(argv[1], "dump-frame")) return dump_frame(argv[2], size_t(std::atoll(argv[3])), argv[4]);
    std::fprintf(stderr,
                 "usage:\n  av1mkv encode <in.cfhd> <out.mkv> [--cq N] [--gop N] [--gpu NAME] [--frames N]\n"
                 "  av1mkv info <file.mkv>\n  av1mkv dump-frame <in.cfhd> <index> <out.ppm>\n");
    return 2;
}
