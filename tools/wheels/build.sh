#!/usr/bin/env bash
# Build the five CUDA extension wheels the Pixal3D service needs (cumesh, flex_gemm,
# o_voxel, nvdiffrast, nvdiffrec_render) as cp311 linux_x86_64 wheels for torch 2.8.0 +
# cu128, each from its V-Sekai-fire fork at a pinned commit (AGENTS.md rule 1).
#
#   tools/wheels/build.sh [name ...]      # default: all five; OUT=dir (default tools/wheels/out)
#
# On the host it builds tools/wheels/Dockerfile and runs itself inside that image
# (no GPU needed); in the container (WHEELS_IN_CONTAINER=1) it fetches, builds and
# records. Output under $OUT: wheels/*.whl + *.whl.sha256 + SHA256SUMS, logs/build-<name>.log,
# logs/sources.log (commit, submodules), logs/toolchain.log, logs/cubins.log (cuobjdump
# --list-elf/--list-ptx of every extension module), logs/needed.log (readelf NEEDED).
# gates/7-pixal3d/own-wheels/README.md records a run.
set -euo pipefail

# name | org fork | pinned commit | package dir inside the checkout
# Why these commits: gates/7-pixal3d/own-wheels/README.md (matched file-for-file against
# the wheels Pixal3D pins and the Windows wheels the desk runs).
SOURCES='
cumesh           https://github.com/V-Sekai-fire/CuMesh      33ff5600b57c4872d0167763b1340d97a6a441c4 .
flex_gemm        https://github.com/V-Sekai-fire/FlexGEMM    74c3f686260f648cbd054c0b143b783838884bbe .
o_voxel          https://github.com/V-Sekai-fire/TRELLIS-2   75fbf0183001ed9876c8dbb35de6b68552ee08bd o-voxel
nvdiffrast       https://github.com/V-Sekai-fire/nvdiffrast  253ac4fcea7de5f396371124af597e6cc957bfae .
nvdiffrec_render https://github.com/V-Sekai-fire/nvdiffrec   b296927cc7fd01c2ac1087c8065c4d7248f72da4 .
'
# Submodules are never fetched from the URL .gitmodules names: each URL (".git" and a
# trailing "/" stripped) maps to its org fork, checked out at the commit the parent's
# gitlink pins. An unmapped URL fails the build.
SUBMODULE_FORKS='
https://github.com/JeffreyXiang/cubvh  https://github.com/V-Sekai-fire/cubvh
git@github.com:JeffreyXiang/cubvh     https://github.com/V-Sekai-fire/cubvh
https://gitlab.com/libeigen/eigen     https://github.com/V-Sekai-fire/interactor-libeigen-eigen
'
# SASS for sm_80 (A100), sm_86 (3090, A40, A6000; sm_87/88 too), sm_89 (4090, L4, L40S),
# sm_90 (H100, H200), sm_100 (B200), sm_120 (RTX 5090, RTX PRO 6000), and compute_120 PTX.
ARCHS="${TORCH_CUDA_ARCH_LIST:-8.0;8.6;8.9;9.0;10.0;12.0+PTX}"
IMAGE=pixal3d-wheels-build:cu128

if [ -z "${WHEELS_IN_CONTAINER:-}" ]; then
    here=$(cd "$(dirname "$0")" && pwd)
    out=${OUT:-$here/out}
    mkdir -p "$out"
    out=$(cd "$out" && pwd)
    if pwd -W >/dev/null 2>&1; then  # Git Bash on Windows: hand Docker C:/... paths
        here=$(cd "$here" && pwd -W); out=$(cd "$out" && pwd -W)
        export MSYS_NO_PATHCONV=1
    fi
    docker build -t "$IMAGE" "$here"
    exec docker run --rm -v "$here:/scripts:ro" -v "$out:/out" \
        -e TORCH_CUDA_ARCH_LIST="$ARCHS" -e MAX_JOBS="${MAX_JOBS:-16}" -e NVCC_THREADS="${NVCC_THREADS:-4}" \
        "$IMAGE" bash /scripts/build.sh "$@"
fi

OUT=/out
mkdir -p "$OUT/wheels" "$OUT/logs" /work
want=" $* "

fetch() {  # url commit dir: a depth-1 fetch of exactly that commit
    rm -rf "$3"; git init -q "$3"
    git -C "$3" remote add origin "$1"
    git -C "$3" fetch -q --depth 1 origin "$2"
    git -C "$3" -c advice.detachedHead=false checkout -q FETCH_HEAD
    [ "$(git -C "$3" rev-parse HEAD)" = "$2" ] || { echo "FAIL $1: HEAD is not $2"; exit 1; }
}

org_fork() {
    local u=${1%/}; u=${u%.git}
    echo "$SUBMODULE_FORKS" | awk -v u="$u" '$1 == u { print $2; found = 1 } END { exit !found }'
}

fill_submodules() {  # dir [indent]
    local d=$1 ind=${2:-  } key path name url sha fork
    [ -f "$d/.gitmodules" ] || return 0
    git -C "$d" config -f .gitmodules --get-regexp '\.path$' | while read -r key path; do
        name=${key#submodule.}; name=${name%.path}
        url=$(git -C "$d" config -f .gitmodules --get "submodule.$name.url")
        sha=$(git -C "$d" ls-tree HEAD -- "$path" | awk '$2 == "commit" { print $3 }')
        [ -n "$sha" ] || { echo "FAIL $d: $path is not a gitlink"; exit 1; }
        fork=$(org_fork "$url") || { echo "FAIL $d: no org fork for submodule $url"; exit 1; }
        fetch "$fork" "$sha" "$d/$path"
        echo "${ind}submodule $path: $url -> $fork @ $sha"
        fill_submodules "$d/$path" "$ind  "
    done
}

{
    echo "date $(date -u +%FT%TZ)"
    grep PRETTY_NAME /etc/os-release
    ldd --version | sed -n 1p  # not head: pipefail + SIGPIPE
    gcc --version | sed -n 1p  # not head: pipefail + SIGPIPE
    nvcc --version | tail -2
    python -c 'import sys, torch, setuptools; print("python", sys.version.split()[0]); print("torch", torch.__version__, "cuda", torch.version.cuda, "cxx11abi", torch.compiled_with_cxx11_abi()); print("setuptools", setuptools.__version__)'
    echo "TORCH_CUDA_ARCH_LIST=$ARCHS MAX_JOBS=$MAX_JOBS NVCC_APPEND_FLAGS=--threads $NVCC_THREADS"
} | tee "$OUT/logs/toolchain.log"

export TORCH_CUDA_ARCH_LIST="$ARCHS" NVCC_APPEND_FLAGS="--threads $NVCC_THREADS"
# nvdiffrec_render links -lcuda (the driver API); at build time that is the toolkit's stub
export LIBRARY_PATH=/usr/local/cuda/lib64/stubs

echo "$SOURCES" | while read -r name url sha sub; do
    [ -n "$name" ] || continue
    [ "$want" = "  " ] || [[ "$want" == *" $name "* ]] || continue
    src=/work/$name
    t0=$(date +%s)
    {
        echo "== $name: $url @ $sha (package dir $sub)"
        fetch "$url" "$sha" "$src"
        git -C "$src" log -1 --format='   commit %H %cI %s'
        fill_submodules "$src"
    } | tee -a "$OUT/logs/sources.log"
    # wheel zip timestamps from the commit, not the clock
    export SOURCE_DATE_EPOCH=$(git -C "$src" log -1 --format=%ct)
    if ! python -m pip wheel --no-build-isolation --no-deps --no-cache-dir -v -w "$OUT/wheels" "$src/$sub" \
            > "$OUT/logs/build-$name.log" 2>&1; then
        tail -40 "$OUT/logs/build-$name.log"; echo "FAIL build $name"; exit 1
    fi
    echo "   built $name in $(( $(date +%s) - t0 )) s" | tee -a "$OUT/logs/sources.log"
done

cd "$OUT/wheels"
rm -f SHA256SUMS
for w in *.whl; do sha256sum "$w" | tee "$w.sha256" >> SHA256SUMS; done
ls -l *.whl | tee "$OUT/logs/wheels.log"
cat SHA256SUMS | tee -a "$OUT/logs/wheels.log"

: > "$OUT/logs/cubins.log"; : > "$OUT/logs/needed.log"
for w in *.whl; do
    x=$(mktemp -d); python -m zipfile -e "$w" "$x"
    for so in $(cd "$x" && find . -name '*.so' | sort); do
        {
            echo "## $w ${so#./}"
            cuobjdump --list-elf "$x/$so" 2>&1 || true
            cuobjdump --list-ptx "$x/$so" 2>&1 || true
            printf '   => SASS: %s | PTX: %s\n' \
                "$(cuobjdump --list-elf "$x/$so" 2>/dev/null | grep -o 'sm_[0-9]*[a-z]*' | sort -u -V | tr '\n' ' ')" \
                "$(cuobjdump --list-ptx "$x/$so" 2>/dev/null | grep -o 'sm_[0-9]*[a-z]*' | sort -u -V | tr '\n' ' ')"
        } >> "$OUT/logs/cubins.log"
        { echo "## $w ${so#./}"; readelf -d "$x/$so" | awk '/NEEDED|RUNPATH|RPATH/ { print "   " $0 }'; } >> "$OUT/logs/needed.log"
    done
    rm -rf "$x"
done
grep -E '^## |=>' "$OUT/logs/cubins.log"
