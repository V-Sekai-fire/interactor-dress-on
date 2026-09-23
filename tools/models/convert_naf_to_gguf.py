"""Convert the NAF release checkpoint (naf_release.pth) to a GGUF for vendor/trellis2/naf.cpp.

    .venv-convert/Scripts/python.exe tools/models/convert_naf_to_gguf.py \
        [--src models/NAF/naf_release.pth] [--dst models/NAF/naf_release_f32.gguf]

NAF (valeoai/NAF, mirrored as V-Sekai-fire/NAF @37f2dfc; Apache-2.0) is 662,528
fp32 parameters plus the rope `periods` buffer. Everything stays fp32 (2.65 MB;
f16 saves nothing that matters). Tensor names are the torch keys, unchanged.
Layout changes, all so the ggml graph needs no weight permute at run time
(gates/7-pixal3d/aux-models/naf.md section 3):

  conv 3x3  torch [O, Ci, 3, 3] -> w.permute(0,2,3,1).reshape(O, 9*Ci)
            = ggml ne [9*Ci, O], ne0 ordered (ci fastest, then tap t = 3*ky+kx),
            matching the host im2col table idx[p*9 + t].
  conv 1x1  torch [O, Ci, 1, 1] -> reshape(O, Ci) = ggml ne [Ci, O].
  GN, bias, periods: 1-D, unchanged.

Metadata (naf.*) records the config the graph assumes; the loader checks it.
"""
import argparse
import hashlib
import os

import numpy as np
import torch
import gguf

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
MODELS = os.environ.get("IDO_MODELS", "C:/interactor-dress-on/models")
SHA256 = "c096c1ab2217a5c3ac136365f721685e2201379cb69d509cfb0261183847c98f"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=os.path.join(MODELS, "NAF", "naf_release.pth"))
    ap.add_argument("--dst", default=os.path.join(MODELS, "NAF", "naf_release_f32.gguf"))
    a = ap.parse_args()

    raw = open(a.src, "rb").read()
    digest = hashlib.sha256(raw).hexdigest()
    if digest != SHA256:
        raise SystemExit(f"{a.src}: sha256 {digest} != release {SHA256}")
    sd = torch.load(a.src, map_location="cpu", weights_only=True)
    assert len(sd) == 37, len(sd)

    w = gguf.GGUFWriter(a.dst, "naf")
    w.add_name("NAF (valeoai/NAF release, V-Sekai-fire/NAF mirror)")
    w.add_string("naf.source_sha256", SHA256)
    w.add_string("naf.source_commit", "37f2dfc180f2de53d98bd601109c0da0dd6b0f43")
    w.add_uint32("naf.dim", 256)            # query/key channels (128 enc || 128 sem)
    w.add_uint32("naf.enc_dim", 128)
    w.add_uint32("naf.num_heads", 4)
    w.add_uint32("naf.kernel_size", 9)
    w.add_uint32("naf.gn_groups", 8)
    w.add_float32("naf.gn_eps", 1e-5)
    w.add_float32("naf.rope_base", 100.0)
    w.add_uint32("naf.enc_blocks", 2)

    n = 0
    for k, v in sd.items():
        t = v.detach().to(torch.float32).contiguous()
        if t.ndim == 4:
            O, Ci, kh, kw = t.shape
            t = t.permute(0, 2, 3, 1).reshape(O, kh * kw * Ci).contiguous()
        arr = t.numpy().astype(np.float32)
        w.add_tensor(k, arr)
        n += arr.size
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    print(f"{a.dst}: {len(sd)} tensors, {n} floats, {os.path.getsize(a.dst)} B")


if __name__ == "__main__":
    main()
