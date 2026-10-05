# Demo Playground

Welcome to the live MAVIS browser demo space. This page is designed for quick, fun validation that your webcam and browser environment are ready before running the desktop demo. This browser playground uses its own browser-side model; it does not validate the installed Python package.

## Interactive Camera Playground

Use the camera permission button, then try the two live modes.

- Pinch Ink: thumb/index fingertip pinch-to-draw with fist-to-erase
- Gesture 3D Morph Lab: left-hand rotate/zoom and right-hand fingertip morph control

<div class="ht-demo-shell" data-ht-camera-demo>
  <div class="ht-demo-topbar">
    <button type="button" class="ht-demo-btn primary" data-ht-request-camera>Enable Camera</button>
    <button type="button" class="ht-demo-btn" data-ht-stop-camera>Stop Camera</button>
    <button type="button" class="ht-demo-btn" data-ht-clear-ink>Clear Ink</button>
    <label class="ht-demo-control" for="ht-line-width">
      Width
      <input id="ht-line-width" type="range" min="1" max="14" value="4" step="1" data-ht-line-width />
      <span data-ht-line-width-value>4px</span>
    </label>
    <label class="ht-demo-control" for="ht-ink-color">
      Color
      <input id="ht-ink-color" type="color" value="#62a8ff" data-ht-ink-color />
    </label>
    <label class="ht-demo-control" for="ht-eraser-size">
      Eraser
      <input id="ht-eraser-size" type="range" min="10" max="60" value="20" step="1" data-ht-eraser-size />
      <span data-ht-eraser-size-value>20px</span>
    </label>
    <span class="ht-demo-status" data-ht-demo-status>Idle: camera not started.</span>
  </div>

  <div class="ht-switcher" data-ht-demo-modes>
    <button type="button" data-ht-mode="air" class="is-active">Pinch Ink</button>
    <button type="button" data-ht-mode="morph">Gesture 3D Morph Lab</button>
  </div>

  <div class="ht-demo-stage">
    <video autoplay playsinline muted data-ht-demo-video></video>
    <canvas data-ht-demo-canvas></canvas>
    <div class="ht-gesture-hud" data-ht-gesture-hud>
      <div class="ht-gesture-badge" data-ht-eraser-badge>Eraser: OFF</div>
      <div class="ht-gesture-badge" data-ht-pinch-badge>Pinch Draw: OFF</div>
    </div>
  </div>

  <p class="ht-demo-note">
    Camera access requires HTTPS or localhost. On GitHub Pages, this is supported by default. Pinch Ink draws on pinch and switches to an eraser when your hand is in a fist; Morph Lab keeps a full camera view and overlays a smaller interactive 3D model in the top-left corner.
  </p>
</div>

## Top 3 Fun Examples

<div class="ht-video-grid">
  <div class="ht-video-card">
    <img src="../source/_static/optitrack_gif_3_2_25.gif" alt="OptiTrack live demo" />
    <p><strong>1) Stereo Triangulation</strong><br>3D reconstruction from synchronized camera views.</p>
  </div>
  <div class="ht-video-card">
    <img src="../source/_static/robot-grasping-gui-trim.gif" alt="Robot integration demo" />
    <p><strong>2) Robot Interaction Flow</strong><br>Hand state feeding a downstream control interface.</p>
  </div>
  <div class="ht-video-card">
    <img src="../source/_static/stereo_hand_track.gif" alt="Stereo tracking demo" />
    <p><strong>3) Live OptiTrack Tracking</strong><br>Multi-camera tracking and visual overlays for real-time feedback.</p>
  </div>
</div>

## Yoha-Inspired Interaction Lab

<div class="ht-idea-grid">
  <div class="ht-idea-card">
    <h3>Pose-First UX</h3>
    <p>Define intent-driven controls around a few stable poses first, then expand interaction vocabulary.</p>
    <div class="ht-chip-row">
      <span class="ht-chip">Pinch = Select</span>
      <span class="ht-chip">Fist = Hold</span>
      <span class="ht-chip">Open Palm = Reset</span>
    </div>
  </div>
  <div class="ht-idea-card">
    <h3>Practical Setup Rules</h3>
    <ul>
      <li>Use localhost/HTTPS for camera access.</li>
      <li>Calibrate once per camera arrangement.</li>
      <li>Prefer lower-latency settings when tuning live control.</li>
    </ul>
  </div>
  <div class="ht-idea-card">
    <h3>Privacy + Performance</h3>
    <p>Keep browser interactions local by default and expose confidence thresholds so operators can tune reliability.</p>
  </div>
</div>

<div class="ht-mission-board" data-ht-mission-board>
  <strong class="ht-mission-title">Gesture Mission Board</strong>
  <div class="ht-switcher" data-ht-mission-controls>
    <button type="button" data-ht-mission="pinch" class="is-active">Pinch Select</button>
    <button type="button" data-ht-mission="fist">Fist Hold</button>
    <button type="button" data-ht-mission="palm">Palm Reset</button>
  </div>
  <p class="ht-demo-note" data-ht-mission-status>Active mission: Pinch Select. Try mapping this to your next UI click target.</p>
</div>

## Run the Full Desktop Demo

```bash
mavis-track demo
mavis-track gui --backend webcam --advanced-hands
```

## Tips

- Keep lighting even and avoid strong backlight for cleaner results.
- If browser camera permission fails, verify OS privacy settings and retry.
- If camera ordering changed since calibration, rerun calibration once.
