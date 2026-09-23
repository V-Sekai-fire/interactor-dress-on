#!/usr/bin/env bash
# AGENTS.md rule 2: kernels are generated from Lean, never copied in. A
# prebuilt SPIR-V blob or a slangc -target cpp output (*.cpu.cpp) under
# vendor/ means a vendoring step pulled one in; fail loudly.
#
#   tools/vendor/check_no_prebuilt.sh [repo-root]
set -euo pipefail

ROOT="${1:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
bad="$(find "$ROOT/vendor" \( -name '*.spv' -o -name '*.spv.gen.h' -o -name '*.cpu.cpp' \) -print)"
if [ -n "$bad" ]; then
	echo "error: prebuilt kernels under vendor/ (AGENTS.md rule 2):" >&2
	echo "$bad" >&2
	exit 1
fi
echo "check_no_prebuilt: no *.spv / *.cpu.cpp under vendor/"
