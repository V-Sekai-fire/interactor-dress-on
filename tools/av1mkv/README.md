# av1mkv: CineForm recording to AV1 in Matroska

Turns the `.cfhd` files entities-godot-cineform's MovieWriter writes (CFHD 12-bit RGB
4:4:4 in an AVI/RIFF container, with a 16-bit PCM track) into AV1 video in Matroska.
No FFmpeg anywhere, no HEVC, no H.264: the decoder is the org's CineForm SDK, the
encoder is NVENC AV1 through the NVIDIA driver, the muxer is the org's libwebm.

- `src/avi_reader.h`: walks the RIFF the writer emits (`LIST hdrl` with `avih`, a `vids`
  CFHD `strl`, an `auds` PCM `strl`; `LIST movi` of `00dc`/`01wb` chunks; `idx1` unused).
- `src/main.cpp`: `CFHD_OpenDecoder` / `CFHD_PrepareToDecode` / `CFHD_DecodeSample` to
  8-bit BGRA (the 12-bit source is rounded to 8 bits: the encoder is 8-bit 4:2:0), rows
  flipped from the DIB bottom-up order the SDK's BGRA uses to top-down, then the encoder
  and the muxer. The PCM track goes through unchanged as `A_PCM/INT/LIT`.
- `src/nvenc_av1.cpp`: `nvEncodeAPI64.dll` loaded at run time with
  `ffnvcodec/dynlink_loader.h` (struct layouts from V-Sekai-fire/nv-codec-headers, the
  copy under `tools/oxrsys/third_party`), a D3D11 device on the adapter whose name
  matches `--gpu` (default `RTX 4090`; never index 0, which is the 3090 on some boots and
  has no AV1 encoder), system-memory input buffers in `NV_ENC_BUFFER_FORMAT_ARGB`, a
  4-deep queue of input/bitstream buffer pairs. Copied from
  `tools/oxrsys/runtime/src/NvencVideoEncoder.cpp`.
- `src/av1c.h`: OBU walker and a real parse of the sequence header (profile, level, tier,
  color_config) for the `av1C` CodecPrivate; temporal delimiters are dropped from the
  blocks, sequence headers stay on every key frame.
- `src/mkv_info.cpp`: `av1mkv info`, the verification: reads the file back with
  libwebm's mkvparser.

## Build

```
tools/av1mkv/build.sh        # llvm-mingw clang + CMake + Ninja -> build/av1mkv.exe
```

Dependencies, all from github.com/V-Sekai-fire as squashed subtrees under `thirdparty/`
(each with a `CITATION.cff`): `cineform-sdk` (9c2973f, Apache-2.0/MIT; built here from
source as `CFHDCodecStatic`, OpenMP off, the two `-Wno-incompatible-*` flags
entities-godot-cineform builds it with, `--allow-multiple-definition` for its duplicated
`GetProcessorCount`) and `libwebm` (6184f44, BSD-3; mkvmuxer and mkvparser compiled in).
One static exe, nothing else next to it.

## Usage

```
av1mkv encode <in.cfhd> <out.mkv> [--cq 22] [--gop 60] [--gpu "RTX 4090"] [--frames N]
av1mkv info <file.mkv>
av1mkv dump-frame <in.cfhd> <index> <out.ppm>
```

## Encoder settings (recorded)

`NV_ENC_CODEC_AV1_GUID`, preset P6, `NV_ENC_TUNING_INFO_HIGH_QUALITY`, rate control
`NV_ENC_PARAMS_RC_VBR` with `averageBitRate = maxBitRate = 0` and `targetQuality = 22`
(NVENC's constant-quality form), two-pass full resolution, spatial AQ, no lookahead, no
B-frames (`frameIntervalP = 1`, so packets are in display order and PTS = DTS), 8-bit
4:2:0 (`chromaFormatIDC = 1`; the 4:4:4 of the source is not kept, NVENC AV1 has no 4:4:4),
`gopLength = idrPeriod = 60`, `repeatSeqHdr = 1`, low-overhead OBU format. Colour is
signalled BT.709 limited; the RGB to YUV conversion is NVENC's own for ARGB input.

## The run (2026-09-23, dress-on-fit.cfhd)

Input `C:/Users/ernest.lee/Desktop/dress-on-fit.cfhd`: 199,456,014 bytes, 1152x648,
30 fps, 1276 CFHD frames, 1276 PCM chunks (48 kHz stereo 16-bit, 6400 bytes each).

Output `C:/Users/ernest.lee/Desktop/dress-on-fit.mkv`: 10,196,086 bytes (5.1% of the
input), of which the AV1 video is 2,010,625 bytes (0.38 Mbit/s at CQ 22: the picture is
mostly static) and the PCM audio 8,166,400 bytes. 1276 video blocks, 22 key frames, 22
clusters, 22 cue points, 42.500 s.

Time: 6.27 s wall in total, 5.73 s for the decode+encode loop = 222.8 fps, 7.4x
realtime. Per frame: CFHD decode 2.66 ms (single-threaded, OpenMP off), row flip
0.43 ms, copy into the NVENC input buffer 0.46 ms, `nvEncEncodePicture` 0.79 ms,
bitstream lock 0.08 ms. The decoder is the larger share; NVENC on the 4090 is not the
bottleneck at this size.

`av1C`: profile 0, level 23 (5.3, as NVENC autoselected), tier 1, 8-bit, 4:2:0, 20 bytes
(`81 17 8c 00` + the 16-byte sequence header OBU).

## Verification

`av1mkv info` (libwebm mkvparser): doctype `matroska` v4, timecode scale 1 ms, duration
42.500 s; track 1 `V_AV1` 1152x648, frame rate 30, default duration 33,333,333 ns,
CodecPrivate 20 bytes; track 2 `A_PCM/INT/LIT` 48000 Hz 2 ch 16 bit; 1276 blocks on each
track, 22 key on video, first 0.000 s, last 42.499 s.

Not checked (the session's budget ran out before it): playback in Windows' Movies & TV
or a browser, and the decoded frame orientation by eye (`dump-frame` writes a PPM for
that; the flip follows the writer's own comment that its BGRA rows are bottom-up).
