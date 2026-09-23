#!/usr/bin/env python3
"""Print the transitive #include closure of a set of C/C++ sources.

    include_closure.py -I DIR [-I DIR ...] SOURCE [SOURCE ...]

Every `#include "x"` / `#include <x>` is followed, whether or not a
preprocessor conditional would take it (a conservative superset: a header
behind a platform #ifdef still lands). Quoted includes resolve against the
including file's directory first, then the -I directories; angle includes
against the -I directories only. An include that resolves nowhere is taken
to be a system header and skipped; unresolved quoted includes are reported on
stderr so a missing project header is not mistaken for one.

Output: one absolute path per line (forward slashes), sources and headers,
sorted.
"""
import argparse
import os
import re
import sys

INCLUDE = re.compile(r'^\s*#\s*include\s*([<"])([^>"]+)[>"]')


def norm(p):
    return os.path.normpath(os.path.abspath(p)).replace("\\", "/")


def resolve(inc, kind, cur_dir, dirs):
    cands = ([cur_dir] if kind == '"' else []) + dirs
    for d in cands:
        p = os.path.join(d, inc)
        if os.path.isfile(p):
            return norm(p)
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-I", dest="dirs", action="append", default=[])
    ap.add_argument("sources", nargs="+")
    a = ap.parse_args()
    dirs = [norm(d) for d in a.dirs]
    seen = set()
    stack = [norm(s) for s in a.sources]
    while stack:
        f = stack.pop()
        if f in seen:
            continue
        seen.add(f)
        with open(f, encoding="utf-8", errors="replace") as fh:
            for line in fh:
                m = INCLUDE.match(line)
                if not m:
                    continue
                kind, inc = m.group(1), m.group(2).strip()
                r = resolve(inc, kind, os.path.dirname(f), dirs)
                if r is None:
                    if kind == '"':
                        print(f"unresolved: {f}: \"{inc}\"", file=sys.stderr)
                    continue
                if r not in seen:
                    stack.append(r)
    for f in sorted(seen):
        print(f)


if __name__ == "__main__":
    main()
