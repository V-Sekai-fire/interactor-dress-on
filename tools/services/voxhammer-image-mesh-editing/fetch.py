"""Fetch the VoxHammer image-mesh-editing service at pinned commits, patch it, place weights.

    pixi run fetch            # idempotent; re-running verifies and exits

src/service    V-Sekai-fire/interactor-voxhammer-image-mesh-editing @ SERVICE_REV + patches/*.patch
src/VoxHammer  V-Sekai-fire/VoxHammer @ VOXHAMMER_REV (win-64 edit path, USD driver, shims)
<models>/TRELLIS-image-large  microsoft/TRELLIS-image-large @ TRELLIS_REV, pipeline.json
               adapted to name the two encoders VoxHammer inverts with (its README, Step 1).

<models> is VOXHAMMER_MODELS, else MODELS_DIR, else models/ beside the MAIN checkout's .git
(worktrees share one copy), else models/ in this tree when there is no git. Every file is checked against tools/models/manifest.tsv.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
SRC = HERE / "src"
REPOS = {
    "service": ("https://github.com/V-Sekai-fire/interactor-voxhammer-image-mesh-editing.git",
                "51d677b61a44c3bb5b68e77d1ab16630e08a724d"),
    "VoxHammer": ("https://github.com/V-Sekai-fire/VoxHammer.git",
                  "63e5ff8daa342122986a2b69e40ea567572bc5e6"),
}
PATCHES = {"service": sorted((HERE / "patches").glob("service-*.patch"))}
TRELLIS_REPO = "microsoft/TRELLIS-image-large"
TRELLIS_REV = "25e0d31ffbebe4b5a97464dd851910efc3002d96"
ENCODERS = {"sparse_structure_encoder": "ckpts/ss_enc_conv3d_16l8_fp16",
            "slat_encoder": "ckpts/slat_enc_swin8_B_64l8_fp16"}


def git(*a, cwd=None):
    subprocess.run(["git", *a], cwd=cwd, check=True)


sys.path.insert(0, str(HERE.parent))
from svc_common import main_checkout as repo_root, models_dir  # noqa: E402


def fetch_repo(name: str) -> None:
    url, rev = REPOS[name]
    dst = SRC / name
    if not (dst / ".git").exists():
        dst.mkdir(parents=True, exist_ok=True)
        git("init", "-q", cwd=dst)
        git("remote", "add", "origin", url, cwd=dst)
        # trellis/.../flexicubes/examples/data/inputmodels/* passes MAX_PATH under %TEMP% worktrees
        git("config", "core.longpaths", "true", cwd=dst)
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=dst, capture_output=True, text=True).stdout.strip()
    if head != rev:
        git("fetch", "-q", "--depth", "1", "origin", rev, cwd=dst)
        git("checkout", "-q", "--force", "FETCH_HEAD", cwd=dst)
    git("reset", "-q", "--hard", rev, cwd=dst)
    git("clean", "-qfdx", "-e", "assets/preset/preset_grid64.usda", cwd=dst)
    for p in PATCHES.get(name, []):
        git("apply", "--whitespace=nowarn", str(p), cwd=dst)
        print(f"[fetch] {name}: applied {p.name}")
    print(f"[fetch] {name} @ {rev[:12]}")


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def fetch_weights(models: Path) -> None:
    from huggingface_hub import hf_hub_download, list_repo_files

    dst = models / "TRELLIS-image-large"
    (dst / "ckpts").mkdir(parents=True, exist_ok=True)
    files = [f for f in list_repo_files(TRELLIS_REPO, revision=TRELLIS_REV) if f.startswith("ckpts/")]
    for f in files:
        out = dst / f
        if out.exists():
            continue
        src = Path(hf_hub_download(TRELLIS_REPO, f, revision=TRELLIS_REV)).resolve()
        try:
            os.link(src, out)  # the HF cache and models/ share a volume: no second copy
        except OSError:
            shutil.copyfile(src, out)
        print(f"[fetch] {f}")
    # pipeline.json fresh from the hub (a cached copy may already carry someone's edit),
    # then the one adaptation: name the encoders the inversion needs.
    tmp = dst / ".pipeline"
    cfg = json.loads(Path(hf_hub_download(TRELLIS_REPO, "pipeline.json", revision=TRELLIS_REV,
                                          local_dir=tmp)).read_text())
    cfg["args"]["models"].update(ENCODERS)
    (dst / "pipeline.json").write_text(json.dumps(cfg, indent=4) + "\n")
    shutil.rmtree(tmp, ignore_errors=True)


def verify(root: Path, models: Path) -> int:
    rows = [l.split("\t") for l in (repo_root_of_manifest() / "tools/models/manifest.tsv").read_text().splitlines()[1:]]
    rows = [r for r in rows if r[0].startswith("models/TRELLIS-image-large/")]
    if not rows:
        print("[fetch] no manifest rows for TRELLIS-image-large"); return 1
    bad = 0
    for relpath, size, digest, *_ in rows:
        p = models / relpath[len("models/"):]
        if not p.exists() or p.stat().st_size != int(size) or sha256(p) != digest:
            print(f"[fetch] MISMATCH {relpath}"); bad += 1
    print(f"[fetch] verified {len(rows) - bad}/{len(rows)} weight files in {models}")
    return 1 if bad else 0


def repo_root_of_manifest() -> Path:
    return HERE.parents[2]  # tools/services/<name> -> this checkout


if __name__ == "__main__":
    for n in REPOS:
        fetch_repo(n)
    models = models_dir("VOXHAMMER_MODELS")
    fetch_weights(models)
    if "--manifest" in sys.argv:  # print rows to paste into tools/models/manifest.tsv
        d = models / "TRELLIS-image-large"
        for p in sorted([d / "pipeline.json", *(d / "ckpts").iterdir()]):
            rev = TRELLIS_REV + ("+encoders" if p.name == "pipeline.json" else "")
            print(f"models/{p.relative_to(models).as_posix()}\t{p.stat().st_size}\t{sha256(p)}\t{TRELLIS_REPO}\t{rev}")
        sys.exit(0)
    sys.exit(verify(repo_root(), models))
