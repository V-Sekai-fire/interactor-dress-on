"""Gate 6d, "Does the rotation generalise": the authored skirt's variants and
the fit_native setups for them.

    python gates/6d-fit-avbd/generalise/make_variants.py

Reads generalise/authored.authored.obj (the loop's authored garment, body
space, from gate_fit_avbd.gd) and writes, into generalise/:
  v-base.obj        the authored skirt as is
  v-rot+30.obj      rotated +30 deg about the vertical axis through the
  v-rot-30.obj      skirt's own x-z centroid (a 3x3 matrix, rule 11)
  v-rad0.85.obj     the x-z radius about that axis scaled 0.85 / 1.15
  v-rad1.15.obj     (the skirt authored wider or narrower)
  v-len0.85.obj     the height below the waist scaled 0.85 / 1.15
  v-len1.15.obj     (the skirt authored shorter or longer)
and setup-<name>.json for fit_native: fixtures/foxgirl/fit_config.json with
the loop's fit budget edits (pipeline.gd _fit_budget: incremental_steps 1,
force_psd_projection true, the PolyFEM loop's own setting) and the loop's
Godot-frame inputs (avatar.obj, skeleton.obj as both skeletons, no nofit).
"""
import json
import os
import re
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.dont_write_bytecode = True
sys.path.insert(0, os.path.join(ROOT, "gates", "6-fit"))
from fit_gap import load_obj  # noqa: E402


def write_obj(path, v, f, comment):
    with open(path, "w", newline="\n") as o:
        o.write("# %s\n" % comment)
        for p in v:
            o.write("v %.9f %.9f %.9f\n" % tuple(p))
        for t in f:
            o.write("f %d %d %d\n" % (t[0] + 1, t[1] + 1, t[2] + 1))


def rot_y(c, s):
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])


def main():
    v, f = load_obj(os.path.join(HERE, "authored.authored.obj"))
    axis = v[:, [0, 2]].mean(0)
    waist_y = v[:, 1].max()
    variants = {"base": v.copy()}
    for name, deg in [("rot+30", 30.0), ("rot-30", -30.0)]:
        R = rot_y(np.cos(np.radians(deg)), np.sin(np.radians(deg)))
        q = v.copy()
        q[:, [0, 2]] -= axis
        q = q @ R.T
        q[:, [0, 2]] += axis
        variants[name] = q
    for k in (0.85, 1.15):
        q = v.copy()
        q[:, [0, 2]] = (q[:, [0, 2]] - axis) * k + axis
        variants["rad%.2f" % k] = q
        q = v.copy()
        q[:, 1] = waist_y - (waist_y - q[:, 1]) * k
        variants["len%.2f" % k] = q
    cfg = open(os.path.join(ROOT, "project", "fixtures", "foxgirl", "fit_config.json")).read()
    cfg, n1 = re.subn(r'("incremental_steps"\s*:\s*)\d+', r"\g<1>1", cfg)
    cfg, n2 = re.subn(r'("nonlinear"\s*:\s*[{]\s*"Newton"\s*:\s*[{])', r'\g<1>"force_psd_projection": true, ', cfg)
    assert n1 == 1 and n2 == 1
    setup = json.loads(cfg)
    fx = os.path.join(ROOT, "project", "fixtures", "foxgirl").replace("\\", "/")
    for name, q in variants.items():
        obj = os.path.join(HERE, "v-%s.obj" % name)
        write_obj(obj, q, f, "Gate 6d generalise variant %s of the authored skirt (body space)" % name)
        s = dict(setup)
        s["avatar_mesh_path"] = fx + "/avatar.obj"
        s["target_skeleton_path"] = fx + "/skeleton.obj"
        s["source_skeleton_path"] = fx + "/skeleton.obj"
        s["avatar_skin_weights_path"] = ""
        s["garment_mesh_path"] = obj.replace("\\", "/")
        s["no_fit_spec_path"] = ""
        with open(os.path.join(HERE, "setup-%s.json" % name), "w", newline="\n") as o:
            json.dump(s, o, indent=2)
        print("%-8s y %.3f..%.3f radius mean %.4f" % (name, q[:, 1].min(), q[:, 1].max(),
              np.hypot(q[:, 0] - axis[0], q[:, 2] - axis[1]).mean()))


if __name__ == "__main__":
    main()
