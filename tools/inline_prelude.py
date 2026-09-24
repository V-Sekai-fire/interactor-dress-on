#!/usr/bin/env python3
"""Put the inline Slang prelude back into cpp emits.

slangc 2026.13.1 on Linux writes `#include "<its install>/slang-cpp-prelude.h"`
as the first line of a `-target cpp` emit; the reference (Windows/scoop)
slangc of the same version inlines the prelude, and that is the form the
committed kernels/*/cpp/*_emit.cpp carry. Beyond that first line the two
emits are byte-identical, so replacing the include with the inline block
taken from any committed emit makes a Linux emit byte-identical to the
reference one, and a build here churns no emit. kernels/*/gen.sh runs it
after slangc; it is idempotent.

    python3 tools/inline_prelude.py <checkout>
"""
import glob
import os
import sys

root = sys.argv[1] if len(sys.argv) > 1 else os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
emits = sorted(glob.glob(os.path.join(root, 'kernels', '*', 'cpp', '*_emit.cpp')))
prelude = None
for path in emits:
    s = open(path).read()
    if s.startswith('#ifndef SLANG_CPP_PRELUDE_H'):
        lines = s.split('\n')
        # The inline prelude ends where the emit body starts.
        end = next(i for i, l in enumerate(lines)
                   if l == '#ifdef SLANG_PRELUDE_NAMESPACE' and lines[i + 1].startswith('using namespace')) - 1
        prelude = '\n'.join(lines[:end])
        break
if prelude is None:
    sys.exit('no committed emit with the inline prelude found under kernels/*/cpp')
changed = 0
for path in emits:
    s = open(path).read()
    first, rest = s.split('\n', 1)
    if first.startswith('#include "') and first.endswith('slang-cpp-prelude.h"'):
        open(path, 'w', newline='\n').write(prelude + '\n' + rest)
        changed += 1
    elif not s.startswith('#ifndef SLANG_CPP_PRELUDE_H'):
        sys.exit('unexpected first line in %s: %r' % (path, first))
print('inline_prelude: %d of %d emits rewritten' % (changed, len(emits)))
