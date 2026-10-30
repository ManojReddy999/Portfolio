"""Render what the navigation policy sees: the car's 64x128 depth image.

Uses the exported training scene (the same model Live Lab runs) and the car's
RealSense camera at the training resolution, picks the heading from the start
pose with the most obstacles in view, and writes an upscaled, colour-mapped
image for the portfolio. Near is bright, far is dark; pixels beyond 4 m fade out.

Usage (from lab/, research venv): python tools/render_depth_view.py ../assets/images/research
"""

import sys
from pathlib import Path

import mujoco
import numpy as np
from PIL import Image

H, W, SCALE, FAR = 64, 128, 6, 4.0
SCENE = Path(__file__).resolve().parent.parent / "assets" / "scene"


def load_model():
    # The web export ships its meshes gzipped; inflate them into an asset dict.
    import gzip
    import json
    manifest = json.loads((SCENE / "manifest.json").read_text())
    assets = {}
    for name in manifest["files"]:
        if name.endswith(".gz"):
            assets[name[:-3]] = gzip.decompress((SCENE / name).read_bytes())
    return mujoco.MjModel.from_xml_string((SCENE / manifest["scene"]).read_text(), assets)


def main(out_dir):
    model = load_model()
    data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, H, W)
    renderer.enable_depth_rendering()
    cam = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "car/buddy_realsense_d435i")
    free = next(j for j in range(model.njnt) if model.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE)
    adr = model.jnt_qposadr[free]

    best = None
    for yaw in np.linspace(0, 2 * np.pi, 24, endpoint=False):
        mujoco.mj_resetData(model, data)
        data.qpos[adr + 3:adr + 7] = [np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)]
        mujoco.mj_forward(model, data)
        renderer.update_scene(data, camera=cam)
        depth = renderer.render().copy()
        top = depth[: H // 2]  # above the horizon: obstacles only, no floor
        score = np.mean((top > 0.8) & (top < 3.5)) + 0.5 * np.std(np.clip(top, 0, FAR))
        if best is None or score > best[0]:
            best = (score, yaw, depth)

    _, yaw, depth = best
    near = np.clip(1 - depth / FAR, 0, 1) ** 1.4
    # Signal-blue ramp: dark navy far, near-white near.
    lo, mid, hi = np.array([2, 6, 11]), np.array([32, 110, 180]), np.array([225, 242, 255])
    t = near[..., None]
    rgb = np.where(t < 0.5, lo + (mid - lo) * (t / 0.5), mid + (hi - mid) * ((t - 0.5) / 0.5)).astype(np.uint8)
    img = Image.fromarray(rgb).resize((W * SCALE, H * SCALE), Image.NEAREST)
    out_dir.mkdir(parents=True, exist_ok=True)
    img.save(out_dir / "policy-depth.webp", quality=88, method=6)
    print(f"yaw {np.degrees(yaw):.0f} deg, score {best[0]:.2f} -> {out_dir / 'policy-depth.webp'}")


if __name__ == "__main__":
    main(Path(sys.argv[1]))
