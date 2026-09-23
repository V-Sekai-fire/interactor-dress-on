#!/usr/bin/env python3
"""Extract the OpenVDB FitForm from a commit for FIT_SDF=openvdb.

fit_native's interim SDF is the one the upstream oracle samples: FitForm as
vendored (OpenVDB meshToSignedDistanceField + SplineSampler), taken from git
so the working tree's FitForm (the SdfGrid port) can change underneath. Its
two debug OBJ exports write to out_dir + "/...", which with the driver's
empty out_dir would be the drive root; both are guarded here, and the SDF
re-meshing that only feeds the second is skipped with it (its outputs are
never read back).

    extract_fitform.py <repo> <rev> <out dir>
"""
import pathlib
import subprocess
import sys

PATH = "vendor/cloth-fit/src/polyfem/solver/forms/garment_forms/"


def show(repo, rev, name):
    return subprocess.run(["git", "-C", repo, "show", "%s:%s%s" % (rev, PATH, name)],
                          check=True, capture_output=True).stdout.decode("utf-8")


def sub(text, old, new):
    n = text.count(old)
    if n != 1:
        raise SystemExit("extract_fitform: expected one %r, found %d" % (old[:50], n))
    return text.replace(old, new)


def main(repo, rev, out):
    out = pathlib.Path(out)
    out.mkdir(parents=True, exist_ok=True)
    hpp = show(repo, rev, "FitForm.hpp").replace("\r\n", "\n")
    cpp = show(repo, rev, "FitForm.cpp").replace("\r\n", "\n")
    if "openvdb" not in hpp:
        raise SystemExit("extract_fitform: %s FitForm.hpp is not the OpenVDB one" % rev)
    cpp = sub(cpp,
              '        io::OBJWriter::write_with_groups(out_dir + "/fit-faces-debug.obj", fit_faces_data);\n',
              '        if (!out_dir.empty()) // fit_native: no debug writes without an out_dir\n'
              '            io::OBJWriter::write_with_groups(out_dir + "/fit-faces-debug.obj", fit_faces_data);\n')
    cpp = sub(cpp,
              '        // export SDF as a triangle mesh\n        {\n',
              '        // export SDF as a triangle mesh\n'
              '        if (!out_dir.empty()) // fit_native: no debug writes without an out_dir\n'
              '        {\n')
    header = "// Extracted by tests/native/fit/extract_fitform.py from %s:%s. Do not edit.\n" % (rev, PATH)
    for name, text in (("FitForm.hpp", hpp), ("FitForm.cpp", cpp)):
        p = out / name
        text = header + text
        if not p.exists() or p.read_text(encoding="utf-8") != text:
            p.write_text(text, encoding="utf-8", newline="\n")
    print("extract_fitform: %s FitForm -> %s" % (rev, out.as_posix()))


if __name__ == "__main__":
    if len(sys.argv) != 4:
        raise SystemExit(__doc__)
    main(*sys.argv[1:])
