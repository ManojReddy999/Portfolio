// three.js view of the MuJoCo scene: a dark world lit only by what the LiDAR sees.
//
// Every world surface samples a top-down visibility mask drawn from the latest
// LiDAR polygon, so "lit" means "sensed right now" and obstacles cast the same
// occlusion shadows a range sensor has. The occupancy map is drawn on the floor
// as a glow that persists in the dark: what the robot remembers.

import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { EffectComposer } from 'three/addons/postprocessing/EffectComposer.js';
import { RenderPass } from 'three/addons/postprocessing/RenderPass.js';
import { UnrealBloomPass } from 'three/addons/postprocessing/UnrealBloomPass.js';
import { OutputPass } from 'three/addons/postprocessing/OutputPass.js';

const MASK_RES = 1024;
const MAP_COLORS = {
  occupied: [88, 185, 255, 255],
  uncertain: [40, 95, 140, 70],
  free: [22, 58, 92, 85],
};
const TRAIL_POINTS = 4000;

// Colours for scans written while the car was not localised: the map as
// odometry alone would have drawn it. Only surfaces, so the drift reads clearly.
export const DRIFT_COLORS = {
  occupied: [255, 107, 87, 255],
  uncertain: [0, 0, 0, 0],
  free: [0, 0, 0, 0],
};

// An occupancy grid painted onto a floor-level texture.
class MapLayer {
  constructor(grid, colors, z) {
    this.grid = grid;
    this.colors = colors;
    this.canvas = document.createElement('canvas');
    this.canvas.width = this.canvas.height = grid.size;
    this.ctx = this.canvas.getContext('2d');
    this.image = this.ctx.createImageData(grid.size, grid.size);
    this.tex = new THREE.CanvasTexture(this.canvas);
    this.tex.colorSpace = THREE.SRGBColorSpace;
    this.tex.magFilter = THREE.LinearFilter;
    const span = grid.size * grid.res;
    this.plane = new THREE.Mesh(
      new THREE.PlaneGeometry(span, span),
      new THREE.MeshBasicMaterial({
        map: this.tex, transparent: true, depthWrite: false,
        blending: THREE.AdditiveBlending, toneMapped: false,
      }));
    this.plane.position.set(-grid.half + span / 2, -grid.half + span / 2, z);
    this.plane.renderOrder = 1;
    this.version = -1;
  }

  update() {
    if (this.grid.version === this.version) return;
    this.grid.paint(this.image, this.colors);
    this.ctx.putImageData(this.image, 0, 0);
    this.tex.needsUpdate = true;
    this.version = this.grid.version;
  }
}

// A path drawn on the floor, keeping the most recent TRAIL_POINTS points.
class Trail {
  constructor(color, opacity, z) {
    this.z = z;
    this.pos = new Float32Array(3 * TRAIL_POINTS);
    const geo = new THREE.BufferGeometry();
    geo.setAttribute('position', new THREE.BufferAttribute(this.pos, 3));
    geo.setDrawRange(0, 0);
    this.line = new THREE.Line(geo, new THREE.LineBasicMaterial({
      color, transparent: true, opacity, blending: THREE.AdditiveBlending, depthWrite: false, toneMapped: false,
    }));
    this.line.frustumCulled = false;
    this.line.renderOrder = 2;
    this.count = 0;
  }

  add(x, y) {
    if (this.count >= TRAIL_POINTS) {
      this.pos.copyWithin(0, 3); // drop the oldest point
      this.count = TRAIL_POINTS - 1;
    }
    const i = 3 * this.count++;
    this.pos[i] = x; this.pos[i + 1] = y; this.pos[i + 2] = this.z;
    this.line.geometry.setDrawRange(0, this.count);
    this.line.geometry.attributes.position.needsUpdate = true;
  }

  clear() {
    this.count = 0;
    this.line.geometry.setDrawRange(0, 0);
  }
}

export class Renderer {
  // interactive: orbit controls (off for an embedded hero, so the page still scrolls).
  // frame(w, h) -> [fx, fy]: shift the followed car by these fractions of the view.
  constructor(container, sim, {
    maskHalf, lidarRange, mapGrid, lowPower = false, interactive = true, maxPixelRatio = 2,
    frame = (w) => (w < 760 ? [0, -0.18] : [0, 0]), // the lab's phone panel covers the bottom
    chase = [3.2, 2.4], // camera starts this far behind the car and this high (m)
  }) {
    this.sim = sim;
    this.container = container;
    this.maskHalf = maskHalf;
    this.lowPower = lowPower;
    this.frame = frame;
    this.chase = chase;

    this.renderer = new THREE.WebGLRenderer({ antialias: !lowPower, powerPreference: 'high-performance' });
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, lowPower ? 1 : maxPixelRatio));
    this.renderer.toneMapping = THREE.ACESFilmicToneMapping;
    this.renderer.toneMappingExposure = 1.05;
    container.appendChild(this.renderer.domElement);

    this.scene = new THREE.Scene();
    this.scene.background = new THREE.Color(0x020407);

    this.camera = new THREE.PerspectiveCamera(42, 1, 0.05, 80);
    if (interactive) {
      this.controls = new OrbitControls(this.camera, this.renderer.domElement);
      this.controls.enableDamping = true;
      this.controls.maxPolarAngle = Math.PI * 0.47;
      this.controls.minDistance = 1.2;
      this.controls.maxDistance = 14;
      this.controls.enablePan = false;
    }

    // MuJoCo is z-up; three.js is y-up. Everything simulated lives in this group.
    this.world = new THREE.Group();
    this.world.rotation.x = -Math.PI / 2;
    this.scene.add(this.world);

    this.#buildMask(lidarRange);
    this.#buildGeoms();
    this.#buildMapOverlay(mapGrid);
    this.#buildTrail();
    this.#buildEstimate();
    this.#buildLights(lidarRange);

    if (!lowPower) {
      this.composer = new EffectComposer(this.renderer);
      this.composer.addPass(new RenderPass(this.scene, this.camera));
      this.bloom = new UnrealBloomPass(new THREE.Vector2(1, 1), 0.4, 0.55, 0.86);
      this.composer.addPass(this.bloom);
      this.composer.addPass(new OutputPass());
    }

    this.resize();
    new ResizeObserver(() => this.resize()).observe(container);
  }

  // ---- visibility mask -----------------------------------------------------

  #buildMask(lidarRange) {
    this.lidarRange = lidarRange;
    this.maskCanvas = document.createElement('canvas');
    this.maskCanvas.width = this.maskCanvas.height = MASK_RES;
    this.maskCtx = this.maskCanvas.getContext('2d');
    this.maskCtx.fillStyle = '#000';
    this.maskCtx.fillRect(0, 0, MASK_RES, MASK_RES);
    this.maskTex = new THREE.CanvasTexture(this.maskCanvas);
    this.maskTex.colorSpace = THREE.NoColorSpace;
    this.shared = {
      uMask: { value: this.maskTex },
      uMaskHalf: { value: this.maskHalf },
      uLidar: { value: new THREE.Vector2() },
      uDark: { value: 0.0 },
    };
  }

  #masked(material) {
    const shared = this.shared;
    material.onBeforeCompile = (shader) => {
      Object.assign(shader.uniforms, shared);
      shader.vertexShader = shader.vertexShader
        .replace('#include <common>', '#include <common>\nvarying vec3 vMaskPos;\nvarying float vMaskUp;')
        .replace('#include <worldpos_vertex>', `#include <worldpos_vertex>
          vMaskPos = (modelMatrix * vec4(transformed, 1.0)).xyz;
          vMaskUp = normalize(mat3(modelMatrix) * objectNormal).y;`);
      shader.fragmentShader = shader.fragmentShader
        .replace('#include <common>', `#include <common>
          uniform sampler2D uMask; uniform float uMaskHalf; uniform vec2 uLidar; uniform float uDark;
          varying vec3 vMaskPos; varying float vMaskUp;`)
        .replace('#include <opaque_fragment>', `#include <opaque_fragment>
          {
            // three.js (x, y, z) -> MuJoCo (x, -z) in the ground plane.
            vec2 p = vec2(vMaskPos.x, -vMaskPos.z);
            vec2 toL = uLidar - p;
            float d = length(toL);
            // Sample slightly toward the sensor: a face a ray struck sits on the lit
            // side of the polygon edge. Tops of obstacles are above the scan plane,
            // so judge them by whether their near edge was seen.
            float pull = (vMaskUp > 0.7 && vMaskPos.y > 0.02) ? 0.45 : 0.05;
            vec2 sp = p + (d > 1e-4 ? toL / d : vec2(0.0)) * min(pull, d);
            float m = texture2D(uMask, (sp + uMaskHalf) / (2.0 * uMaskHalf)).r;
            gl_FragColor.rgb *= mix(uDark, 1.0, m);
          }`);
    };
    material.customProgramCacheKey = () => 'lidar-masked';
    return material;
  }

  updateMask(lidar) {
    this.lastLidar = lidar;
    if (!this.build) this.#drawMask(lidar, 1);
  }

  // The LiDAR polygon with every ray shortened to `reach` of its length.
  #drawMask(lidar, reach) {
    const ctx = this.maskCtx, N = MASK_RES, h = this.maskHalf;
    const sx = (x) => ((x + h) / (2 * h)) * N;
    const sy = (y) => ((h - y) / (2 * h)) * N;
    ctx.fillStyle = '#000';
    ctx.fillRect(0, 0, N, N);

    const ox = sx(lidar.origin[0]), oy = sy(lidar.origin[1]);
    const r = Math.max(1, (reach * lidar.range / (2 * h)) * N);
    const g = ctx.createRadialGradient(ox, oy, 0, ox, oy, r);
    g.addColorStop(0, '#ffffff');
    g.addColorStop(0.3, '#d2d2d2');
    g.addColorStop(0.62, '#5e5e5e');
    g.addColorStop(0.86, '#181818');
    g.addColorStop(1, '#000000');

    ctx.beginPath();
    const [lx, ly] = lidar.origin;
    for (let k = 0; k < lidar.n; k++) {
      const [ex, ey] = lidar.endpoint(k);
      const px = sx(lx + reach * (ex - lx)), py = sy(ly + reach * (ey - ly));
      if (k === 0) ctx.moveTo(px, py); else ctx.lineTo(px, py);
    }
    ctx.closePath();
    ctx.fillStyle = g;
    ctx.fill();
    // Dilate slightly so the faces the rays struck are inside the lit region.
    ctx.strokeStyle = g;
    ctx.lineWidth = 4;
    ctx.lineJoin = 'round';
    ctx.stroke();

    this.maskTex.needsUpdate = true;
    this.shared.uLidar.value.set(lidar.origin[0], lidar.origin[1]);
    this.lidarLight.position.set(lidar.origin[0], lidar.origin[1], lidar.origin[2] + 1.1);
  }

  // ---- geometry from the MuJoCo model --------------------------------------

  #buildGeoms() {
    const { sim } = this;
    const m = sim.model, mj = sim.mujoco, G = mj.mjtGeom;
    const T = {
      plane: G.mjGEOM_PLANE.value, sphere: G.mjGEOM_SPHERE.value, capsule: G.mjGEOM_CAPSULE.value,
      ellipsoid: G.mjGEOM_ELLIPSOID.value, cylinder: G.mjGEOM_CYLINDER.value, box: G.mjGEOM_BOX.value,
      mesh: G.mjGEOM_MESH.value,
    };

    const mat = {
      floor: this.#masked(new THREE.MeshStandardMaterial({ color: 0x18222c, roughness: 0.92, metalness: 0.0 })),
      wall: this.#masked(new THREE.MeshStandardMaterial({ color: 0x33424f, roughness: 0.85 })),
      obstacle: [
        this.#masked(new THREE.MeshStandardMaterial({ color: 0x8e9aa6, roughness: 0.62 })),
        this.#masked(new THREE.MeshStandardMaterial({ color: 0x7d8b98, roughness: 0.66 })),
        this.#masked(new THREE.MeshStandardMaterial({ color: 0xa39c91, roughness: 0.58 })),
      ],
      chassis: new THREE.MeshStandardMaterial({ color: 0x2a323b, roughness: 0.38, metalness: 0.55 }),
      wheel: new THREE.MeshStandardMaterial({ color: 0x0d1013, roughness: 0.9 }),
      sensor: new THREE.MeshStandardMaterial({ color: 0x15191d, roughness: 0.4, metalness: 0.3 }),
      lidar: new THREE.MeshStandardMaterial({ color: 0x0b1620, emissive: 0x58b9ff, emissiveIntensity: 2.2, roughness: 0.3 }),
      pusher: new THREE.MeshStandardMaterial({ color: 0x58b9ff, roughness: 0.4, metalness: 0.2 }),
    };

    this.dynamic = [];
    const rgba = m.geom_rgba, size = m.geom_size, type = m.geom_type, body = m.geom_bodyid;
    const meshName = (id) => mj.mj_id2name(m, mj.mjtObj.mjOBJ_MESH.value, id) || '';
    const geomName = (id) => mj.mj_id2name(m, mj.mjtObj.mjOBJ_GEOM.value, id) || '';

    for (let g = 0; g < m.ngeom; g++) {
      if (rgba[4 * g + 3] < 0.01) continue; // collision-only geoms are transparent in the model
      const t = type[g], s = [size[3 * g], size[3 * g + 1], size[3 * g + 2]];
      const onCar = sim.carBodies.has(body[g]);
      let geo, material;

      if (t === T.plane) {
        const w = s[0] > 0 ? 2 * s[0] : 40, h = s[1] > 0 ? 2 * s[1] : 40;
        geo = new THREE.PlaneGeometry(w + 1.5, h + 1.5);
        material = mat.floor;
      } else if (t === T.box) {
        geo = new THREE.BoxGeometry(2 * s[0], 2 * s[1], 2 * s[2]);
      } else if (t === T.sphere) {
        geo = new THREE.SphereGeometry(s[0], 40, 24);
      } else if (t === T.cylinder) {
        geo = new THREE.CylinderGeometry(s[0], s[0], 2 * s[1], 48).rotateX(Math.PI / 2);
      } else if (t === T.capsule) {
        geo = new THREE.CapsuleGeometry(s[0], 2 * s[1], 8, 24).rotateX(Math.PI / 2);
      } else if (t === T.ellipsoid) {
        geo = new THREE.SphereGeometry(1, 32, 16).scale(s[0], s[1], s[2]);
      } else if (t === T.mesh) {
        geo = this.#meshGeometry(m.geom_dataid[g]);
      } else {
        continue;
      }

      if (!material) {
        if (onCar) {
          const mn = t === T.mesh ? meshName(m.geom_dataid[g]) : '';
          const gn = geomName(g);
          material = /wheel/.test(mn) ? mat.wheel
            : /ydlidar/.test(mn) ? mat.lidar
            : /base/.test(mn) ? mat.chassis
            : /realsense/.test(gn) ? mat.sensor
            : /pusher/.test(gn) ? mat.pusher
            : mat.chassis;
        } else {
          const isWall = Math.max(...s) > 1;
          material = isWall ? mat.wall : mat.obstacle[g % mat.obstacle.length];
        }
      }

      const mesh = new THREE.Mesh(geo, material);
      mesh.matrixAutoUpdate = false;
      this.world.add(mesh);
      this.#placeGeom(mesh, g);
      if (onCar) this.dynamic.push({ mesh, g });
      else {
        mesh.userData.staticGeom = g;
        mesh.userData.obstacle = t !== T.plane && material !== mat.wall;
      }
    }
  }

  #meshGeometry(dataid) {
    const m = this.sim.model;
    const va = m.mesh_vertadr[dataid], vn = m.mesh_vertnum[dataid];
    const fa = m.mesh_faceadr[dataid], fn = m.mesh_facenum[dataid];
    const geo = new THREE.BufferGeometry();
    geo.setAttribute('position', new THREE.BufferAttribute(Float32Array.from(m.mesh_vert.subarray(3 * va, 3 * (va + vn))), 3));
    geo.setIndex(new THREE.BufferAttribute(Uint32Array.from(m.mesh_face.subarray(3 * fa, 3 * (fa + fn))), 1));
    const flat = geo.toNonIndexed();
    flat.computeVertexNormals();
    geo.dispose();
    return flat;
  }

  #placeGeom(mesh, g) {
    const p = this.sim.data.geom_xpos, R = this.sim.data.geom_xmat;
    const i = 3 * g, j = 9 * g;
    mesh.matrix.set(
      R[j], R[j + 1], R[j + 2], p[i],
      R[j + 3], R[j + 4], R[j + 5], p[i + 1],
      R[j + 6], R[j + 7], R[j + 8], p[i + 2],
      0, 0, 0, 1);
    mesh.matrixWorldNeedsUpdate = true;
  }

  refreshStatic() {
    this.world.traverse((o) => { if (o.userData.staticGeom !== undefined) this.#placeGeom(o, o.userData.staticGeom); });
  }

  // ---- map overlay and trail ------------------------------------------------

  #buildMapOverlay(grid) {
    this.layers = [];
    this.mapLayer = this.addMapLayer(grid, MAP_COLORS, 0.004);
  }

  // The SLAM map's canvas, for the minimap.
  get mapCanvas() { return this.mapLayer.canvas; }

  // Draw an occupancy grid on the floor as a glowing additive overlay.
  addMapLayer(grid, colors, z) {
    const layer = new MapLayer(grid, colors, z);
    this.world.add(layer.plane);
    this.layers.push(layer);
    return layer;
  }

  updateMap() {
    for (const layer of this.layers) layer.update();
  }

  #buildTrail() {
    this.trueTrail = new Trail(0xf2a93b, 0.55, 0.012); // ground truth
    this.estTrail = new Trail(0x8fd3ff, 0.75, 0.018); // where SLAM thinks it went
    this.world.add(this.trueTrail.line, this.estTrail.line);
  }

  addTrailPoints(truth, estimate) {
    this.trueTrail.add(truth[0], truth[1]);
    if (estimate) this.estTrail.add(estimate[0], estimate[1]);
  }

  clearTrail() {
    this.trueTrail.clear();
    this.estTrail.clear();
  }

  // ---- SLAM estimate: footprint, heading and 3-sigma uncertainty -------------

  #buildEstimate() {
    const line = (opacity) => new THREE.LineBasicMaterial({
      color: 0x8fd3ff, transparent: true, opacity, blending: THREE.AdditiveBlending,
      depthWrite: false, toneMapped: false,
    });
    const v = (x, y) => new THREE.Vector3(x, y, 0);
    this.estimate = new THREE.Group();
    this.estimate.add(new THREE.LineLoop(
      new THREE.BufferGeometry().setFromPoints([v(-0.21, -0.165), v(0.225, -0.165), v(0.225, 0.165), v(-0.21, 0.165)]),
      line(0.95)));
    this.estimate.add(new THREE.Line(
      new THREE.BufferGeometry().setFromPoints([v(0.08, -0.09), v(0.38, 0), v(0.08, 0.09)]), line(0.95)));
    this.estimate.renderOrder = 3;
    this.world.add(this.estimate);

    const ring = Array.from({ length: 64 }, (_, i) => v(Math.cos((i / 64) * 2 * Math.PI), Math.sin((i / 64) * 2 * Math.PI)));
    this.ellipse = new THREE.LineLoop(new THREE.BufferGeometry().setFromPoints(ring), line(0.55));
    this.ellipse.renderOrder = 3;
    this.world.add(this.ellipse);
  }

  // Pose estimate and its covariance (row-major 3x3 over x, y, theta).
  setEstimate(pose, P) {
    const [x, y, th] = pose;
    this.estimate.position.set(x, y, 0.022);
    this.estimate.rotation.set(0, 0, th);
    // 3-sigma ellipse of the x-y block.
    const a = P[0], b = P[1], c = P[4];
    const mid = 0.5 * (a + c), rad = Math.sqrt(0.25 * (a - c) ** 2 + b * b);
    const r1 = 3 * Math.sqrt(Math.max(mid + rad, 0)), r2 = 3 * Math.sqrt(Math.max(mid - rad, 0));
    this.ellipse.position.set(x, y, 0.02);
    this.ellipse.rotation.set(0, 0, 0.5 * Math.atan2(2 * b, a - c));
    this.ellipse.scale.set(Math.max(r1, 0.02), Math.max(r2, 0.02), 1);
  }

  // ---- lights, camera, frame -----------------------------------------------

  #buildLights(range) {
    // The world is lit from the sensor; the car carries its own small fill so it reads in the dark.
    this.lidarLight = new THREE.PointLight(0xfff1e0, 7, range * 1.25, 1.5);
    this.world.add(this.lidarLight);
    this.scene.add(new THREE.HemisphereLight(0x8fb8ff, 0x050608, 0.14));
  }

  followCar() {
    const p = this.sim.carPosition();
    const target = new THREE.Vector3(p[0], p[2] + 0.15, -p[1]);
    if (!this.lastTarget) {
      const yaw = this.sim.carYaw();
      const [back, up] = this.chase;
      this.camera.position.set(target.x - back * Math.cos(yaw), target.y + up, target.z + back * Math.sin(yaw));
      this.lastTarget = target.clone();
      if (this.controls) this.controls.target.copy(target);
    }
    // Follow the car's position but not its heading, so the view doesn't swing.
    const delta = target.clone().sub(this.lastTarget);
    this.camera.position.add(delta);
    this.lastTarget.copy(target);
    if (this.controls) {
      this.controls.target.add(delta);
      this.controls.update();
    } else {
      this.camera.lookAt(target);
    }
  }

  // ---- arena build-in ----------------------------------------------------------

  // Obstacles grow out of the floor in a wave outward from the car while the LiDAR
  // light sweeps out to full range. Resolves when done; the lab holds the car
  // still until then (see `building`).
  buildIn(duration = 1.8) {
    const p = this.sim.data.geom_xpos, [cx, cy] = this.sim.carPosition();
    const items = [];
    this.world.traverse((o) => {
      if (!o.userData.obstacle) return;
      const g = o.userData.staticGeom;
      items.push({ mesh: o, base: o.matrix.clone(), d: Math.hypot(p[3 * g] - cx, p[3 * g + 1] - cy) });
    });
    const far = Math.max(...items.map((i) => i.d), 1);
    for (const i of items) i.delay = 0.6 * (i.d / far);
    this.build = { start: performance.now(), duration: duration * 1000, items };
    this.#applyBuild(0);
    return new Promise((resolve) => setTimeout(resolve, duration * 1000));
  }

  get building() {
    if (this.build && performance.now() - this.build.start >= this.build.duration) this.finishBuild();
    return !!this.build;
  }

  finishBuild() {
    if (!this.build) return;
    this.#applyBuild(1);
    this.build = null;
    if (this.lastLidar) this.#drawMask(this.lastLidar, 1);
  }

  // t in [0, 1] over the whole build; each obstacle rises over 40% of it, after its delay.
  #applyBuild(t) {
    const ease = (x) => 1 - (1 - Math.min(Math.max(x, 0), 1)) ** 3;
    const S = new THREE.Matrix4();
    for (const { mesh, base, delay } of this.build.items) {
      const s = Math.max(ease((t - delay) / 0.4), 1e-3);
      mesh.matrix.copy(base).premultiply(S.makeScale(1, 1, s)); // world z is up: flatten onto the floor
      mesh.matrixWorldNeedsUpdate = true;
    }
    if (this.lastLidar) this.#drawMask(this.lastLidar, 0.05 + 0.95 * ease(t / 0.8));
  }

  render() {
    if (this.building) this.#applyBuild((performance.now() - this.build.start) / this.build.duration);
    for (const { mesh, g } of this.dynamic) this.#placeGeom(mesh, g);
    this.followCar();
    this.updateMap();
    if (this.composer) this.composer.render(); else this.renderer.render(this.scene, this.camera);
  }

  resize() {
    const w = this.container.clientWidth, h = this.container.clientHeight;
    if (!w || !h) return;
    this.camera.aspect = w / h;
    // Keep the car clear of whatever overlays the view (the lab's panel, the hero's text).
    const [fx, fy] = this.frame(w, h);
    if (fx || fy) this.camera.setViewOffset(w, h, -fx * w, -fy * h, w, h);
    else this.camera.clearViewOffset();
    this.camera.updateProjectionMatrix();
    this.renderer.setSize(w, h, false);
    if (this.composer) {
      this.composer.setSize(w, h);
      this.bloom.setSize(w, h);
    }
  }
}
