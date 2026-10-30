"""Make the exported scene light enough for the web without changing its physics.

The chassis mesh (mushr_base_nano, ~200k triangles, 10 MB) is visual only: its
geom has contype=0 and conaffinity=0. It still matters to physics in one way --
MuJoCo derives the car body's mass properties from its geoms. So this tool:

1. compiles the scene as exported and records the car body's compiled inertial
   (mass, centre of mass, principal axes, principal moments);
2. writes those values into the body as an explicit <inertial>, so geoms no
   longer contribute;
3. replaces the chassis mesh with a vertex-clustered simplification;
4. recompiles and checks that every body's mass properties are unchanged.

parity.html then checks the result against the rollout recorded in Python.

Usage (from lab/, with the research venv's MuJoCo 3.12):
    python tools/lighten_scene.py assets/scene --cell 0.004
"""

import argparse
import gzip
import json
import re
import struct
from pathlib import Path

import mujoco
import numpy as np

CAR_BODY = "car/buddy"
CHASSIS_MESH = "car/buddy_mushr_base_nano"


def read_stl(path):
    data = path.read_bytes()
    (n,) = struct.unpack_from("<I", data, 80)
    rec = np.frombuffer(data, dtype=np.dtype([("n", "<f4", 3), ("v", "<f4", (3, 3)), ("a", "<u2")]),
                        count=n, offset=84)
    return rec["v"].astype(np.float64)  # (n, 3, 3)


def write_stl(path, tris):
    # Facet normals are left zero: MuJoCo computes its own, and zeros compress well.
    rec = np.zeros(len(tris), dtype=np.dtype([("n", "<f4", 3), ("v", "<f4", (3, 3)), ("a", "<u2")]))
    rec["v"] = tris.astype(np.float32)
    path.write_bytes(b"lighten_scene.py: vertex-clustered".ljust(80, b" ") + struct.pack("<I", len(tris)) + rec.tobytes())


def cluster(tris, cell):
    """Vertex clustering: snap vertices to a grid, average each cell, drop collapsed faces."""
    verts = tris.reshape(-1, 3)
    keys = np.floor((verts - verts.min(0)) / cell).astype(np.int64)
    _, cell_id = np.unique(keys, axis=0, return_inverse=True)
    cell_id = cell_id.ravel()
    centres = np.zeros((cell_id.max() + 1, 3))
    np.add.at(centres, cell_id, verts)
    centres /= np.bincount(cell_id)[:, None]
    faces = cell_id.reshape(-1, 3)
    keep = (faces[:, 0] != faces[:, 1]) & (faces[:, 1] != faces[:, 2]) & (faces[:, 0] != faces[:, 2])
    faces = faces[keep]
    # The same triangle can survive from several source faces; keep one per vertex set and winding.
    canon = np.sort(faces, axis=1)
    _, first = np.unique(canon, axis=0, return_index=True)
    faces = faces[np.sort(first)]
    return centres[faces]


def mass_properties(model):
    return {k: getattr(model, k).copy() for k in ("body_mass", "body_ipos", "body_iquat", "body_inertia", "body_subtreemass")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("scene_dir", type=Path)
    ap.add_argument("--cell", type=float, default=0.004, help="clustering cell size in metres")
    args = ap.parse_args()

    scene_dir = args.scene_dir
    manifest = json.loads((scene_dir / "manifest.json").read_text())
    xml_path = scene_dir / manifest["scene"]
    xml = xml_path.read_text()
    if "<inertial" in xml.split(f'name="{CAR_BODY}"')[1].split("<body")[0]:
        raise SystemExit("car body already has an explicit inertial; run on a fresh export")

    before = mujoco.MjModel.from_xml_path(str(xml_path))
    props = mass_properties(before)
    b = mujoco.mj_name2id(before, mujoco.mjtObj.mjOBJ_BODY, CAR_BODY)
    fmt = lambda a: " ".join(repr(float(x)) for x in a)
    inertial = (f'<inertial pos="{fmt(before.body_ipos[b])}" quat="{fmt(before.body_iquat[b])}" '
                f'mass="{float(before.body_mass[b])!r}" diaginertia="{fmt(before.body_inertia[b])}"/>')

    mesh_tag = re.search(rf'<mesh name="{re.escape(CHASSIS_MESH)}"[^>]*file="([^"]+)"', xml)
    src = scene_dir / mesh_tag.group(1)
    tris = read_stl(src)
    light = cluster(tris, args.cell)
    out_name = src.stem.split("-")[0] + f"-web{int(args.cell * 1000)}mm.stl"
    write_stl(scene_dir / out_name, light)

    xml = re.sub(rf'(<body name="{re.escape(CAR_BODY)}"[^>]*>)', lambda m: m.group(1) + "\n        " + inertial, xml, count=1)
    xml = xml.replace(f'file="{src.name}"', f'file="{out_name}"')
    xml_path.write_text(xml)
    manifest["files"] = [out_name if f == src.name else f for f in manifest["files"]]
    manifest["chassis_mesh"] = {"source": src.name, "triangles": len(tris), "web_triangles": len(light),
                                "cell_m": args.cell}
    (scene_dir / "manifest.json").write_text(json.dumps(manifest, indent=1))

    after = mujoco.MjModel.from_xml_path(str(xml_path))
    worst = max(float(np.abs(props[k] - getattr(after, k)).max()) for k in props)
    print(f"chassis: {len(tris)} -> {len(light)} triangles, {src.stat().st_size} -> "
          f"{(scene_dir / out_name).stat().st_size} bytes")
    print(f"largest change in any body's mass properties: {worst:.3e}")
    if worst > 1e-12:
        raise SystemExit("mass properties changed; not safe")
    src.unlink()

    # GitHub Pages compresses XML but serves STL as-is, so ship meshes gzipped;
    # the browser inflates them with DecompressionStream (src/sim.js).
    for i, name in enumerate(manifest["files"]):
        if name.endswith(".stl"):
            raw = scene_dir / name
            (scene_dir / (name + ".gz")).write_bytes(gzip.compress(raw.read_bytes(), 9, mtime=0))
            raw.unlink()
            manifest["files"][i] = name + ".gz"
    (scene_dir / "manifest.json").write_text(json.dumps(manifest, indent=1))
    print("files:", manifest["files"])


if __name__ == "__main__":
    main()
