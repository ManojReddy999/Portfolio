// Live Lab: MuJoCo in the browser, a car exploring in the dark, mapping and
// localising itself as it goes.
//
// startLab() runs it inside any element. The full lab page passes its whole
// HUD; the portfolio home page embeds it as a non-interactive hero.

import { loadSim } from './sim.js';
import { Lidar } from './lidar.js';
import { LocalPlanner } from './controller.js';
import { OccupancyGrid } from './mapping.js';
import { Slam } from './slam.js';
import { Renderer, DRIFT_COLORS } from './render.js';

const CONTROL_DT = 0.05; // 20 Hz, as in training
const ARENA = 8; // training arena half-size (m)
const LIDAR = { rays: 360, range: 4.0 };
const MAX_CONTROL_STEPS_PER_FRAME = 6;
const HERO_ARENA_SECONDS = 180;
// The home page widens the arena (same obstacle density) and keeps the car near the
// middle, so the walls never come into view.
const HERO_ARENA = 16;
const HERO_LEASH = 6; // m from the centre before the driver starts heading back
const HERO_MINIMAP_HALF = 8; // the hero minimap shows a window around the car

const wrap = (a) => Math.atan2(Math.sin(a), Math.cos(a));

// Small seeded RNG so a layout (and its sensor errors) can be reproduced from its number.
function mulberry32(seed) {
  return () => {
    seed |= 0; seed = (seed + 0x6d2b79f5) | 0;
    let t = Math.imul(seed ^ (seed >>> 15), 1 | seed);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function isLowPower() {
  return matchMedia('(max-width: 760px)').matches || (navigator.hardwareConcurrency || 8) <= 4;
}

/**
 * Run the lab in `stage`.
 *   ui        element whose descendants hold the optional HUD (#stat-*, #minimap, #btn-*)
 *   hero      embedded mode: no camera controls, no URL updates, pauses when off screen
 *   frame     (w, h) => [fx, fy]: shift the car by these fractions of the view
 *   onStatus  loading messages, one line
 *   onStep    (step, fraction) as each real loading stage starts: 'wasm', 'compile', 'scene', 'build'
 *   seed      layout number; random if omitted
 * Resolves once the scene is drawn; `lab.built` resolves when the arena has finished
 * building itself and the car starts driving.
 */
export async function startLab({ stage, ui = document, hero = false, frame, onStatus = () => {}, onStep = () => {}, seed } = {}) {
  const $ = (id) => ui.querySelector(`#${id}`);
  const setText = (id, text) => { const el = $(id); if (el) el.textContent = text; };
  const STATUS = {
    wasm: (f) => `Downloading MuJoCo physics… ${Math.round(100 * f)}%`,
    compile: () => 'Compiling the physics engine…',
    scene: () => 'Loading the training scene…',
    build: () => 'Building the arena…',
  };
  const step = (name, f = 0) => { onStep(name, f); onStatus(STATUS[name](f)); };

  const arena = hero ? HERO_ARENA : ARENA;
  const sim = await loadSim(step, { arena });
  step('build');

  const lidar = new Lidar(sim, LIDAR);
  const lidarOffset = sim.lidarOffset();
  const grid = new OccupancyGrid({ half: arena + 0.6, res: 0.05 });
  const view = new Renderer(stage, sim, {
    maskHalf: arena + 0.6, lidarRange: LIDAR.range, mapGrid: grid, lowPower: isLowPower(),
    interactive: !hero, maxPixelRatio: hero ? 1.5 : 2, frame,
    chase: hero ? [2.2, 4.6] : undefined, // the hero looks down more steeply, so obstacles rarely hide the car
  });
  // Scans taken while not localised: drawn, but kept out of the map SLAM matches against.
  const drift = new OccupancyGrid({ half: arena + 0.6, res: 0.05 });
  const driftLayer = view.addMapLayer(drift, DRIFT_COLORS, 0.006);

  const state = {
    seed: 0, driver: null, slam: null, noise: 1, matching: true, loopClosure: true, loops: [],
    distance: 0, collisions: 0, wasColliding: false,
    lastXY: null, progressAt: 0, progressXY: null, paused: false, speed: 1, visible: true,
    frames: 0, fpsAt: performance.now(), fps: 0, posErr: 0, headErr: 0,
  };

  let built = Promise.resolve();

  // Start SLAM from a known pose: at the start, or after a respawn.
  function startLocalisation() {
    lidar.scan();
    state.slam.reinit(...sim.carPose(), lidar);
  }

  function newLayout(layout = Math.floor(Math.random() * 1e6)) {
    state.seed = layout;
    sim.reset();
    sim.scatterObstacles(mulberry32(layout), arena);
    sim.placeCar(0, 0, 0);
    view.refreshStatic();
    grid.clear();
    drift.clear();
    view.clearTrail();
    state.driver = new LocalPlanner();
    state.slam = new Slam(sim, grid, mulberry32(layout ^ 0x5bd1e995), state.noise, lidarOffset,
      { loopClosure: state.loopClosure });
    state.slam.matching = state.matching;
    state.loops = [];
    state.respawnRng = mulberry32(layout ^ 0x2545f491); // so a layout number reproduces the whole run
    state.distance = 0; state.collisions = 0; state.wasColliding = false;
    state.lastXY = null; state.progressAt = sim.time; state.progressXY = sim.carPosition();
    startLocalisation();
    view.updateMask(lidar);
    built = view.buildIn(); // obstacles rise around the car and the LiDAR light sweeps out
    setText('layout', `#${layout}`);
    if (!hero) history.replaceState(null, '', `?layout=${layout}`);
  }

  // Clear the map and restart localisation where the car is.
  function restartMapping() {
    grid.clear();
    drift.clear();
    view.clearTrail();
    state.slam.resetMap();
    state.loops = [];
    startLocalisation();
  }

  // Respawn somewhere clear (keeping the map) if the stand-in driver is truly stuck.
  // This teleports the car, so SLAM is re-initialised at the new pose.
  function respawn() {
    const rng = state.respawnRng;
    for (let attempt = 0; attempt < 200; attempt++) {
      const x = (rng() * 2 - 1) * (hero ? HERO_LEASH : ARENA - 0.8);
      const y = (rng() * 2 - 1) * (hero ? HERO_LEASH : ARENA - 0.8);
      sim.placeCar(x, y, rng() * 2 * Math.PI);
      lidar.scan();
      if (Math.min(...lidar.ranges) > 0.6 && !sim.inCollision()) break;
    }
    state.respawns = (state.respawns || 0) + 1;
    state.driver = new LocalPlanner();
    state.slam.reinit(...sim.carPose(), lidar); // lidar holds the scan from the chosen spot
    drift.clear();
    state.lastXY = null; state.progressAt = sim.time; state.progressXY = sim.carPosition();
  }

  function controlStep() {
    const collided = sim.inCollision();
    if (collided && !state.wasColliding) state.collisions++;
    state.wasColliding = collided;

    // The driver reacts to the true scan; SLAM never feeds back into driving.
    // In the hero, past the leash it is pulled back toward the middle of the arena.
    let home = null;
    if (hero) {
      const [x, y, yaw] = sim.carPose(), past = Math.hypot(x, y) - HERO_LEASH;
      if (past > 0) home = { angle: wrap(Math.atan2(-y, -x) - yaw), weight: 0.6 * Math.min(1, past / 3) };
    }
    const cmd = state.driver.act(lidar, sim.time, sim.carForwardSpeed(), collided, home);
    state.mode = cmd.mode;
    const slam = state.slam;
    sim.step(cmd.steer, cmd.throttle, (dt) => slam.odom.sample(dt));
    lidar.scan();
    if (slam.update(lidar)) drift.clear(); // relocalised: the drifted picture is no longer believed
    // SLAM writes the map itself, from keyframes; scans taken blind go to the drift layer.
    if (!slam.localised) drift.integrate(lidar, slam.sensorPose());
    for (const loop of slam.takeLoopEvents()) {
      view.flashLoop(loop.from, loop.to);
      if (loop.path) view.setEstimatePath(loop.path); // the corrected route replaces the drifted one
      state.loops.push({ ...loop, at: sim.time });
    }

    const truth = sim.carPose(), est = slam.pose();
    state.posErr = Math.hypot(est[0] - truth[0], est[1] - truth[1]);
    state.headErr = Math.abs(wrap(est[2] - truth[2]));
    if (state.lastXY) state.distance += Math.hypot(truth[0] - state.lastXY[0], truth[1] - state.lastXY[1]);
    state.lastXY = [truth[0], truth[1]];
    view.addTrailPoints(truth, est);

    // No real progress for 6 s, or rolled over: start again somewhere clear.
    if (Math.hypot(truth[0] - state.progressXY[0], truth[1] - state.progressXY[1]) > 0.4) {
      state.progressAt = sim.time; state.progressXY = [truth[0], truth[1]];
    } else if (sim.time - state.progressAt > 6 || sim.carUpright() < 0.6) {
      respawn();
    }
  }

  // ---- HUD and minimap -------------------------------------------------------

  const mini = $('minimap'), mctx = mini && mini.getContext('2d');
  // The part of the map the minimap shows: centre and half-size (m).
  let win = { cx: 0, cy: 0, h: grid.half };
  function updateWindow() {
    if (!hero) return;
    const [x, y] = sim.carPosition(), h = HERO_MINIMAP_HALF, lim = grid.half - h;
    win = { cx: Math.max(-lim, Math.min(lim, x)), cy: Math.max(-lim, Math.min(lim, y)), h };
  }
  const mx = (x) => ((x - win.cx + win.h) / (2 * win.h)) * mini.width;
  const my = (y) => ((win.cy + win.h - y) / (2 * win.h)) * mini.width;
  function drawCar(x, y, yaw, fill, stroke) {
    mctx.save();
    mctx.translate(mx(x), my(y));
    mctx.rotate(-yaw);
    mctx.beginPath();
    mctx.moveTo(7, 0); mctx.lineTo(-5, 4.5); mctx.lineTo(-5, -4.5);
    mctx.closePath();
    if (fill) { mctx.fillStyle = fill; mctx.fill(); }
    if (stroke) { mctx.strokeStyle = stroke; mctx.lineWidth = 1.5; mctx.stroke(); }
    mctx.restore();
  }

  function drawMinimap() {
    if (!mini) return;
    updateWindow();
    const S = mini.width, sx = mx, sy = my;
    mctx.fillStyle = '#04080c';
    mctx.fillRect(0, 0, S, S);
    mctx.imageSmoothingEnabled = false;
    // source rectangle of the map canvases (one pixel per cell, +y up)
    const ppm = grid.size / (2 * grid.half);
    const src = [(win.cx - win.h + grid.half) * ppm, (grid.half - win.cy - win.h) * ppm, 2 * win.h * ppm, 2 * win.h * ppm];
    mctx.drawImage(view.mapCanvas, ...src, 0, 0, S, S);
    mctx.drawImage(driftLayer.canvas, ...src, 0, 0, S, S);

    // Where it really went (amber) and where it believes it went (blue).
    const path = (trail, color) => {
      if (trail.count < 2) return;
      mctx.beginPath();
      const P = trail.pos;
      for (let i = 0; i < trail.count; i++) {
        const x = sx(P[3 * i]), y = sy(P[3 * i + 1]);
        // a jump of over 1 m is a respawn (or a relocalisation), not a path
        const jump = i === 0 || Math.hypot(P[3 * i] - P[3 * i - 3], P[3 * i + 1] - P[3 * i - 2]) > 1;
        if (jump) mctx.moveTo(x, y); else mctx.lineTo(x, y);
      }
      mctx.strokeStyle = color;
      mctx.lineWidth = 1.5;
      mctx.stroke();
    };
    path(view.trueTrail, 'rgba(242, 169, 59, 0.75)');
    path(view.estTrail, 'rgba(143, 211, 255, 0.75)');

    // Where the current scan really landed.
    mctx.fillStyle = 'rgba(255, 244, 230, 0.9)';
    for (let k = 0; k < lidar.n; k += 2) {
      if (!lidar.hit[k]) continue;
      const [ex, ey] = lidar.endpoint(k);
      mctx.fillRect(sx(ex) - 0.6, sy(ey) - 0.6, 1.2, 1.2);
    }
    for (const loop of state.loops) {
      const age = sim.time - loop.at;
      if (age > 4) continue;
      mctx.strokeStyle = `rgba(255, 244, 230, ${0.9 * (1 - age / 4)})`;
      mctx.lineWidth = 2;
      mctx.beginPath();
      mctx.moveTo(sx(loop.from[0]), sy(loop.from[1]));
      mctx.lineTo(sx(loop.to[0]), sy(loop.to[1]));
      mctx.stroke();
    }
    const [tx, ty, tyaw] = sim.carPose(), [ex, ey, eyaw] = state.slam.pose();
    drawCar(tx, ty, tyaw, '#f2a93b', null);
    drawCar(ex, ey, eyaw, null, '#8fd3ff');
  }

  function updateHud() {
    setText('stat-time', `${sim.time.toFixed(0)} s`);
    setText('stat-distance', `${state.distance.toFixed(1)} m`);
    // The hero's arena is too big to state a useful percentage; it shows the area mapped.
    setText('stat-coverage', hero ? `${Math.round(grid.coverage(arena) * (2 * arena) ** 2)} m²`
      : `${(100 * grid.coverage(ARENA)).toFixed(1)}%`);
    setText('stat-collisions', String(state.collisions));
    setText('stat-mode', state.mode || '—');
    const cm = 100 * state.posErr;
    setText('stat-error', `${cm.toFixed(cm < 10 ? 1 : 0)} cm · ${(state.headErr * 180 / Math.PI).toFixed(1)}°`);
    const toDeg = 180 / Math.PI, s = state.slam;
    setText('stat-loops', String(s.closer.closures));
    setText('stat-bias', `${(s.ekf.x[3] * toDeg).toFixed(2)} (${(s.odom.gyroBias * toDeg).toFixed(2)})`);
    setText('stat-loc', s.slipping ? `${s.state} · wheel slip` : s.state);
    const loc = $('stat-loc');
    if (loc) loc.dataset.state = s.slipping ? 'slip' : s.state;
    setText('stat-fps', String(state.fps));
  }

  function drawFrame() {
    view.setEstimate(state.slam.pose(), state.slam.ekf.poseCov());
    view.render();
    drawMinimap();
  }

  // ---- loop ----------------------------------------------------------------

  // Off screen (an embedded hero scrolled away), do no work at all.
  new IntersectionObserver(([entry]) => { state.visible = entry.isIntersecting; }).observe(stage);

  let acc = 0, last = performance.now();
  function tick(now) {
    const dt = Math.min(0.1, (now - last) / 1000);
    last = now;
    if (state.visible) {
      if (!state.paused && !view.building) {
        acc += dt * state.speed;
        let n = 0;
        while (acc >= CONTROL_DT && n < MAX_CONTROL_STEPS_PER_FRAME) { controlStep(); acc -= CONTROL_DT; n++; }
        if (n === MAX_CONTROL_STEPS_PER_FRAME) acc = 0; // fell behind; don't spiral
        if (n > 0) view.updateMask(lidar);
        // The hero runs for as long as the page is open: start a fresh arena every
        // few minutes, so the map and pose graph stay small and the build plays again.
        if (hero && sim.time > HERO_ARENA_SECONDS) newLayout();
      }
      drawFrame();
      if (++state.frames % 10 === 0) {
        state.fps = Math.round(10000 / (now - state.fpsAt));
        state.fpsAt = now;
        updateHud();
      }
    }
    requestAnimationFrame(tick);
  }

  // ---- controls --------------------------------------------------------------

  const on = (id, fn) => { const el = $(id); if (el) el.onclick = fn; };
  const setPaused = (paused) => {
    state.paused = paused;
    setText('btn-pause', paused ? 'Play' : 'Pause');
  };
  on('btn-layout', () => newLayout());
  on('btn-clear', () => restartMapping());
  on('btn-pause', () => setPaused(!state.paused));
  for (const b of ui.querySelectorAll('[data-speed]')) {
    b.onclick = () => {
      state.speed = Number(b.dataset.speed);
      ui.querySelectorAll('[data-speed]').forEach((o) => o.classList.toggle('on', o === b));
    };
  }
  on('btn-loops', () => {
    state.loopClosure = !state.loopClosure;
    state.slam.loopClosure = state.loopClosure;
    setText('btn-loops', state.loopClosure ? 'Loop closure: on' : 'Loop closure: off');
    $('btn-loops').classList.toggle('on', state.loopClosure);
  });
  on('btn-matching', () => {
    state.matching = !state.matching;
    state.slam.matching = state.matching;
    setText('btn-matching', state.matching ? 'Scan matching: on' : 'Scan matching: off');
    $('btn-matching').classList.toggle('on', state.matching);
  });
  const noise = $('noise');
  if (noise) {
    noise.oninput = (e) => {
      state.noise = Number(e.target.value);
      state.slam.setLevel(state.noise);
      setText('noise-value', `${state.noise.toFixed(2).replace(/\.?0+$/, '')}×`);
    };
  }

  newLayout(seed);
  if (!hero && matchMedia('(prefers-reduced-motion: reduce)').matches) setPaused(true);
  drawFrame();
  updateHud();

  // Test hook: advance the simulation without relying on animation frames
  // (which browsers suspend for hidden views), then draw one frame.
  const lab = {
    sim, lidar, grid, view, state, newLayout, restartMapping, setPaused,
    get built() { return built; },
    // Benchmark hook: n control steps with no drawing at all.
    step(n = 1) { view.finishBuild(); for (let i = 0; i < n; i++) controlStep(); },
    advance(seconds) {
      view.finishBuild();
      for (let i = 0, n = Math.round(seconds / CONTROL_DT); i < n; i++) controlStep();
      view.updateMask(lidar);
      drawFrame();
      updateHud();
    },
  };
  window.__lab = lab;
  requestAnimationFrame(tick);
  return lab;
}
