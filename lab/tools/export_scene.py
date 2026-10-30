"""Export the exact MuJoCo scene the navigation policy trains on, for the browser.

The car model mixes angle conventions (radian camera Eulers, degree wheel
Eulers) that dm_control reconciles while composing the arena. Rather than
re-deriving that by hand, this builds the training task with the training
code and lets dm_control write out the compiled scene and its meshes.

It also records a short reference rollout so the browser build can be checked
against the Python physics it is meant to reproduce.

Usage (from any directory):
    <python with dm_control> tools/export_scene.py <path to LocoTransformer_v2>
"""

import json
import os
import sys

import mujoco
import numpy as np

V2 = os.path.abspath(sys.argv[1])
HERE = os.path.dirname(os.path.abspath(__file__))
SCENE_DIR = os.path.join(HERE, "..", "assets", "scene")
PARITY_DIR = os.path.join(HERE, "..", "assets", "parity")

# The training code resolves the car XML relative to its own working directory.
os.chdir(V2)
sys.path.insert(0, V2)

from dm_control import mjcf  # noqa: E402
from task import Survive  # noqa: E402

NUM_OBSTACLES = 150  # matches results/depth-reverse-s2-2026-10-28/manifest.json
ARENA_SIZE = 8

task = Survive(num_obstacles=NUM_OBSTACLES, arena_size=ARENA_SIZE, random_seed=0)
os.makedirs(SCENE_DIR, exist_ok=True)
mjcf.export_with_assets(task.root_entity.mjcf_model, SCENE_DIR, out_file_name="scene.xml")

with open(os.path.join(SCENE_DIR, "manifest.json"), "w") as f:
    json.dump({
        "scene": "scene.xml",
        "files": sorted(n for n in os.listdir(SCENE_DIR) if n != "manifest.json"),
        "num_obstacles": NUM_OBSTACLES,
        "arena_size": ARENA_SIZE,
    }, f, indent=1)

model = mujoco.MjModel.from_xml_path(os.path.join(SCENE_DIR, "scene.xml"))
data = mujoco.MjData(model)

actuators = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i) for i in range(model.nu)]
car = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "car/buddy")

# Reference rollout: fixed controls, no randomness, sampled every control step.
CONTROL = [0.2, 1.5]  # steering (rad), throttle
SUBSTEPS = 10  # 0.05 s control step over 0.005 s physics steps
mujoco.mj_resetData(model, data)
mujoco.mj_forward(model, data)
samples = []
for step in range(60):
    data.ctrl[:] = CONTROL
    for _ in range(SUBSTEPS):
        mujoco.mj_step(model, data)
    samples.append({
        "t": round(float(data.time), 6),
        "xpos": [float(v) for v in data.xpos[car]],
        "xquat": [float(v) for v in data.xquat[car]],
    })

os.makedirs(PARITY_DIR, exist_ok=True)
with open(os.path.join(PARITY_DIR, "reference.json"), "w") as f:
    json.dump({
        "mujoco_version": mujoco.__version__,
        "control": CONTROL,
        "substeps": SUBSTEPS,
        "timestep": float(model.opt.timestep),
        "car_body": "car/buddy",
        "samples": samples,
    }, f, indent=1)

print(f"mujoco {mujoco.__version__}")
print(f"timestep {model.opt.timestep}  ngeom {model.ngeom}  nbody {model.nbody}  nmesh {model.nmesh}")
print(f"actuators {actuators}")
print(f"car body id {car}")
print(f"final car pos after {len(samples)} control steps: {np.round(data.xpos[car], 4).tolist()}")
print("files:", sorted(os.listdir(SCENE_DIR)))
