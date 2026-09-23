"""Convert BiRefNet_HR-matting (model.safetensors, F16) to a GGUF for vendor/birefnet.

    .venv-convert/Scripts/python.exe tools/models/convert_birefnet_to_gguf.py \
        [--src models/BiRefNet_HR-matting/model.safetensors] \
        [--dst models/BiRefNet_HR-matting/birefnet_hr_matting_f16.gguf]

Source: HF ZhengPeng7/BiRefNet_HR-matting @5d6b6f8 (MIT), 754 tensors: 687 F16
(220,202,578 elements) + 67 I64 buffers. Census and graph:
gates/7-pixal3d/aux-models/birefnet.md.

Tensor names are the torch keys. Weight VALUES are the file's f16 values,
unchanged (only permuted); BatchNorm running stats become f32 scale/shift.

  conv k x k  torch [O, Ci, kh, kw] -> permute(0,2,3,1).reshape(O, kh*kw*Ci)
              = ggml ne [kh*kw*Ci, O]; contraction index = (ky*kw + kx)*Ci + ci,
              matching the gather-im2col tables in birefnet.cpp (row p*K + k of
              a token-major [Ci, H*W+1] activation). 1x1 is reshape(O, Ci).
              The deformable regular_conv uses the same order (sampler output
              row = k*Ci + ci).
  ipt_blkN.conv1 (dec_ipt_split): its input channel is torch's rearrange
              order c*g*g + ihg*g + iwg; the graph builds patches with one
              GET_ROWS whose channel order is (ihg*g + iwg)*3 + c, so the Ci
              axis is permuted to that order here (g = 32, 16, 8, 4, 1 for
              blocks 5..1 at the 1024 input).
  BatchNorm2d (eval, eps 1e-5) -> <prefix>.bn_scale = w / sqrt(var + eps),
              <prefix>.bn_shift = b - mean * bn_scale, both f32 (computed in
              float64 from the f16 stats); torch's CPU batch_norm evaluates
              the same x*alpha + beta form.
  relative_position_bias_table [529, nH]: kept (ggml ne [nH, 529]); the
              [nH, 144, 144] bias is expanded at load from the index formula.
  Dropped: relative_position_index (I64, regenerated), num_batches_tracked,
              and the training-only heads conv_ms_spvn_* / gdt_convs_pred_*.
  Linear, LayerNorm, biases: unchanged.

--ftype f32 writes every weight as f32 (identical values, 2x size).
"""
import argparse
import hashlib
import json
import os
import re

import numpy as np
import gguf
from safetensors.numpy import load_file

MODELS = os.environ.get("IDO_MODELS", "C:/interactor-dress-on/models")
SRC_SHA256 = "a5a4de698739ea5e0e8bbab28e1b293dde95092b87a442d566cbc585c53cef55"
SRC_REPO = "ZhengPeng7/BiRefNet_HR-matting"
SRC_REV = "5d6b6f8adcb5b417c871b1d84ceaae9871355b7f"

# dec_ipt_split patch grid per ipt block at the 1024 input (x4 is 32x32 etc.)
IPT_GRID = {5: 32, 4: 16, 3: 8, 2: 4, 1: 1}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=os.path.join(MODELS, "BiRefNet_HR-matting", "model.safetensors"))
    ap.add_argument("--dst", default=None)
    ap.add_argument("--ftype", choices=["f16", "f32"], default="f16")
    ap.add_argument("--no-verify", action="store_true", help="skip the sha256 check")
    a = ap.parse_args()
    if a.dst is None:
        a.dst = os.path.join(os.path.dirname(a.src), f"birefnet_hr_matting_{a.ftype}.gguf")

    if not a.no_verify:
        h = hashlib.sha256()
        with open(a.src, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 24), b""):
                h.update(chunk)
        if h.hexdigest() != SRC_SHA256:
            raise SystemExit(f"{a.src}: sha256 {h.hexdigest()} != {SRC_SHA256}")

    sd = load_file(a.src)
    assert len(sd) == 754, len(sd)
    wtype = np.float16 if a.ftype == "f16" else np.float32

    out = {}  # name -> np array (f16/f32)

    # BatchNorm prefixes: every key ending in running_mean
    bn_prefixes = sorted(k[: -len(".running_mean")] for k in sd if k.endswith(".running_mean"))
    for p in bn_prefixes:
        w = sd[p + ".weight"].astype(np.float64)
        b = sd[p + ".bias"].astype(np.float64)
        m = sd[p + ".running_mean"].astype(np.float64)
        v = sd[p + ".running_var"].astype(np.float64)
        s = w / np.sqrt(v + 1e-5)
        out[p + ".bn_scale"] = s.astype(np.float32)
        out[p + ".bn_shift"] = (b - m * s).astype(np.float32)
    bn_keys = {p + sfx for p in bn_prefixes for sfx in
               (".weight", ".bias", ".running_mean", ".running_var", ".num_batches_tracked")}

    n_drop = 0
    for k, t in sd.items():
        if k in bn_keys:
            continue
        if (k.endswith("relative_position_index") or k.endswith("num_batches_tracked")
                or ".conv_ms_spvn_" in k or ".gdt_convs_pred_" in k):
            n_drop += 1
            continue
        assert t.dtype == np.float16, (k, t.dtype)
        if t.ndim == 4:
            O, Ci, kh, kw = t.shape
            m = re.match(r"decoder\.ipt_blk(\d)\.conv1\.weight$", k)
            if m:
                g = IPT_GRID[int(m.group(1))]
                assert Ci == 3 * g * g, (k, t.shape)
                # [O, c, ihg, iwg, kh, kw] -> [O, kh, kw, ihg, iwg, c]
                t = t.reshape(O, 3, g, g, kh, kw).transpose(0, 4, 5, 2, 3, 1)
            else:
                t = t.transpose(0, 2, 3, 1)
            t = np.ascontiguousarray(t).reshape(O, kh * kw * Ci)
        out[k] = np.ascontiguousarray(t.astype(wtype) if t.dtype == np.float16 else t)

    w = gguf.GGUFWriter(a.dst, "birefnet")
    w.add_name("BiRefNet_HR-matting (ZhengPeng7), swin_v1_l, ASPPDeformable decoder")
    w.add_string("birefnet.source_repo", SRC_REPO)
    w.add_string("birefnet.source_revision", SRC_REV)
    w.add_string("birefnet.source_sha256", SRC_SHA256)
    w.add_string("birefnet.layout", "conv [kh*kw*Ci, O] tap-major; ipt conv1 Ci patch-major; bn scale/shift")
    w.add_uint32("birefnet.embed_dim", 192)
    w.add_array("birefnet.depths", [2, 2, 18, 2])
    w.add_array("birefnet.num_heads", [6, 12, 24, 48])
    w.add_uint32("birefnet.window_size", 12)
    w.add_uint32("birefnet.patch_size", 4)
    w.add_float32("birefnet.ln_eps", 1e-5)
    w.add_uint32("birefnet.input_size", 1024)
    w.add_array("birefnet.ipt_grid", [IPT_GRID[i] for i in (5, 4, 3, 2, 1)])
    w.add_file_type(gguf.LlamaFileType.MOSTLY_F16 if a.ftype == "f16" else gguf.LlamaFileType.ALL_F32)
    n_el = 0
    for k in sorted(out):
        w.add_tensor(k, out[k])
        n_el += out[k].size
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    print(json.dumps({"dst": a.dst, "tensors": len(out), "elements": int(n_el),
                      "bn_folded_to_scale_shift": len(bn_prefixes), "dropped": n_drop,
                      "bytes": os.path.getsize(a.dst)}))


if __name__ == "__main__":
    main()
