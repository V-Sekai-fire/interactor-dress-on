"""Fetch everything the Pixal3D service needs, pinned: source + patches + weights.

  pixi run fetch            # sources and weights
  pixi run fetch --src      # sources only (clone at the pinned commit, apply patches)
  pixi run fetch --weights  # weights only (download at the pinned revision, verify sha256)

Sources are cloned into ./src (gitignored), never copied into this repo. Every
Windows change is a patch under ./patches, applied here and listed in CITATION.cff.
Weights go to <main checkout>/models (gitignored) and are checked against
tools/models/manifest.tsv; a file with no manifest row gets one appended.
"""

import argparse
import hashlib
import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
SRC = HERE / "src"
PATCHES = HERE / "patches"

# name -> (org repo, pinned commit, patches applied in order)
SOURCES = {
    "service": ("https://github.com/V-Sekai-fire/interactor-pixal3d-image-to-textured-mesh",
                "61df29e1878c8f0dcda71dd735a002493a068ba2", ["service-win.patch"]),
    "pixal3d": ("https://github.com/V-Sekai-fire/Pixal3D",
                "cdbb2bbffbf4e6f298b5f2af3d1d76a8d823d2af", ["pixal3d-local-aux.patch"]),
    "moge": ("https://github.com/V-Sekai-fire/MoGe",
             "74fbce054ebed49800de42d0ad0e83495065719a", ["moge-flexgemm2.patch"]),
    # MoGe-3's refiner needs FlexGEMM 2.0 (flex_gemm.nn); Pixal3D needs the 0.0.1 API
    # (flex_gemm.ops.spconv / .grid_sample), which 2.0 removed. 2.0 runs pure Triton
    # without its CUDA extension and imports itself only relatively, so it is exposed
    # as a second package, flex_gemm2 (see link_flexgemm2). Org fork of JeffreyXiang's.
    "flexgemm2": ("https://github.com/V-Sekai-fire/FlexGEMM",
                  "b2fadb29d41846c7981ade6801ffc689fae119cf", ["flexgemm2-own-cache.patch"]),
    "naf": ("https://github.com/V-Sekai-fire/NAF",
            "37f2dfc180f2de53d98bd601109c0da0dd6b0f43", ["naf-na2d-block.patch"]),
}

# relpath under models/ -> (HF repo, revision). Pixal3D-src extends the rows the
# gate already holds for TencentARC/Pixal3D @b0cb2e1 (the single-view files only).
_PX = ("TencentARC/Pixal3D", "b0cb2e1b794cab9aa0ac38a95d794a4d9337437f")
WEIGHTS = {f"Pixal3D-src/{f}": _PX for f in [
    "pipeline.json",
    "ckpts/ss_dec_conv3d_16l8_fp16.json", "ckpts/ss_dec_conv3d_16l8_fp16.safetensors",
    "ckpts/ss_flow_img_dit_1_3B_64_bf16.json", "ckpts/ss_flow_img_dit_1_3B_64_bf16.safetensors",
    "ckpts/shape_dec_next_dc_f16c32_fp16.json", "ckpts/shape_dec_next_dc_f16c32_fp16.safetensors",
    "ckpts/tex_dec_next_dc_f16c32_fp16.json", "ckpts/tex_dec_next_dc_f16c32_fp16.safetensors",
    "ckpts/slat_flow_img2shape_dit_1_3B_512_bf16.json",
    "ckpts/slat_flow_img2shape_dit_1_3B_512_bf16.safetensors",
    "ckpts/slat_flow_img2shape_dit_1_3B_1024_bf16.json",
    "ckpts/slat_flow_img2shape_dit_1_3B_1024_bf16.safetensors",
    "ckpts/slat_flow_imgshape2tex_dit_1_3B_1024_bf16.json",
    "ckpts/slat_flow_imgshape2tex_dit_1_3B_1024_bf16.safetensors",
]}
_DINO = ("camenduru/dinov3-vitl16-pretrain-lvd1689m", "3c276edd87d6f6e569ff0c4400e086807d0f3881")
WEIGHTS.update({f"dinov3-vitl16/{f}": _DINO for f in
                ["config.json", "model.safetensors", "preprocessor_config.json"]})
WEIGHTS["moge-3-vitl/model.pt"] = ("Ruicheng/moge-3-vitl", "184008f877d7ad1ad4c2cd2182a9bd1f63d0e5be")
_BRN = ("ZhengPeng7/BiRefNet_HR-matting", "5d6b6f8adcb5b417c871b1d84ceaae9871355b7f")
WEIGHTS.update({f"BiRefNet_HR-matting/{f}": _BRN for f in
                ["config.json", "birefnet.py", "BiRefNet_config.py", "model.safetensors"]})
# NAF's release checkpoint is a GitHub release asset, not an HF file.
NAF_URL = "https://github.com/valeoai/NAF/releases/download/model/naf_release.pth"


sys.path.insert(0, str(HERE.parent))
from svc_common import main_checkout, models_dir as _models_dir  # noqa: E402


def models_dir() -> Path:
    return _models_dir("PIXAL3D_MODELS")


def git(*args, cwd=None):
    subprocess.check_call(["git", "-c", "core.autocrlf=false", *args], cwd=cwd)


def fetch_sources():
    SRC.mkdir(exist_ok=True)
    for name, (url, rev, patches) in SOURCES.items():
        d = SRC / name
        if not d.exists():
            git("clone", "-q", url, str(d))
            git("config", "core.autocrlf", "false", cwd=d)
        git("fetch", "-q", "origin", cwd=d)
        git("checkout", "-q", "-f", rev, cwd=d)
        git("clean", "-qfdx", cwd=d)
        for p in patches:
            git("apply", "--whitespace=nowarn", str(PATCHES / p), cwd=d)
        print(f"src/{name} @ {rev[:10]} + {patches or 'no patches'}")
    link_flexgemm2()


def link_flexgemm2():
    """src/flexgemm2-pkg/flex_gemm2 -> a copy of src/flexgemm2/flex_gemm (patched)."""
    import shutil
    pkg = SRC / "flexgemm2-pkg"
    shutil.rmtree(pkg, ignore_errors=True)
    shutil.copytree(SRC / "flexgemm2" / "flex_gemm", pkg / "flex_gemm2")
    print("src/flexgemm2-pkg/flex_gemm2 <- src/flexgemm2/flex_gemm")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 24), b""):
            h.update(block)
    return h.hexdigest()


def fetch_weights():
    from huggingface_hub import hf_hub_download

    root = models_dir()
    manifest = main_checkout() / "tools" / "models" / "manifest.tsv"
    rows = {}
    for line in manifest.read_text().splitlines()[1:]:
        cols = line.split("\t")
        if len(cols) >= 5:
            rows[cols[0]] = cols
    new_rows = []
    items = [(rel, repo, rev) for rel, (repo, rev) in WEIGHTS.items()]
    items.append(("NAF/naf_release.pth", "valeoai/NAF (release model; V-Sekai-fire/NAF mirror pending)",
                  "model@f63aadffba49c7131fb2b421e165b3e9acd5b52a"))
    for rel, repo, rev in items:
        dest = root / rel
        if not dest.exists():
            print(f"download {repo}@{rev[:10]} {rel}", flush=True)
            if rel.startswith("NAF/"):
                import urllib.request
                dest.parent.mkdir(parents=True, exist_ok=True)
                urllib.request.urlretrieve(NAF_URL, dest)
            else:
                sub = rel.split("/", 1)[1]
                hf_hub_download(repo, sub, revision=rev, local_dir=str(root / rel.split("/", 1)[0]))
        key = f"models/{rel}"
        digest = sha256(dest)
        if key in rows:
            if rows[key][2] != digest:
                sys.exit(f"FAIL sha256 {key}: {digest} != manifest {rows[key][2]}")
            print(f"  ok  {key}")
        else:
            new_rows.append("\t".join([key, str(dest.stat().st_size), digest, repo, rev]))
            print(f"  new {key} {digest[:12]}")
    if new_rows:
        # appended to the manifest of the checkout this script runs in
        here_manifest = Path(subprocess.check_output(
            ["git", "-C", str(HERE), "rev-parse", "--show-toplevel"], text=True).strip()) / "tools/models/manifest.tsv"
        with open(here_manifest, "a", newline="\n") as f:
            f.write("\n".join(new_rows) + "\n")
        print(f"appended {len(new_rows)} rows to {here_manifest}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", action="store_true")
    ap.add_argument("--weights", action="store_true")
    a = ap.parse_args()
    both = not (a.src or a.weights)
    if a.src or both:
        fetch_sources()
    if a.weights or both:
        fetch_weights()
