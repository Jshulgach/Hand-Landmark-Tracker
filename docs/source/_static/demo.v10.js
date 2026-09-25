(function () {
  const MP_HAND_CONNECTIONS = [
    [0, 1], [1, 2], [2, 3], [3, 4],
    [0, 5], [5, 6], [6, 7], [7, 8],
    [5, 9], [9, 10], [10, 11], [11, 12],
    [9, 13], [13, 14], [14, 15], [15, 16],
    [13, 17], [17, 18], [18, 19], [19, 20],
    [0, 17],
  ];

  let mediaPipeLoader = null;
  let threeLoader = null;

  function loadScriptOnce(src) {
    return new Promise((resolve, reject) => {
      const existing = document.querySelector('script[data-src="' + src + '"]');
      if (existing) {
        if (existing.getAttribute('data-loaded') === 'true') {
          resolve();
          return;
        }
        existing.addEventListener('load', () => resolve(), { once: true });
        existing.addEventListener('error', () => reject(new Error('Failed to load ' + src)), { once: true });
        return;
      }

      const script = document.createElement('script');
      script.src = src;
      script.async = true;
      script.defer = true;
      script.setAttribute('data-src', src);
      script.addEventListener('load', () => {
        script.setAttribute('data-loaded', 'true');
        resolve();
      }, { once: true });
      script.addEventListener('error', () => reject(new Error('Failed to load ' + src)), { once: true });
      document.head.appendChild(script);
    });
  }

  function ensureMediaPipeLoaded() {
    if (mediaPipeLoader) {
      return mediaPipeLoader;
    }

    mediaPipeLoader = Promise.all([
      loadScriptOnce('https://cdn.jsdelivr.net/npm/@mediapipe/hands/hands.js'),
    ]).then(() => {
      if (!window.Hands) {
        throw new Error('MediaPipe Hands global not found');
      }
    });

    return mediaPipeLoader;
  }

  function ensureThreeLoaded() {
    if (threeLoader) {
      return threeLoader;
    }

    threeLoader = loadScriptOnce('https://cdn.jsdelivr.net/npm/three@0.161.0/build/three.min.js').then(() => {
      if (!window.THREE) {
        throw new Error('Three.js global not found');
      }
    });

    return threeLoader;
  }

  function initCameraDemo(root) {
    const requestButton = root.querySelector('[data-ht-request-camera]');
    const stopButton = root.querySelector('[data-ht-stop-camera]');
    const status = root.querySelector('[data-ht-demo-status]');
    const video = root.querySelector('[data-ht-demo-video]');
    const canvas = root.querySelector('[data-ht-demo-canvas]');
    const clearInkButton = root.querySelector('[data-ht-clear-ink]');
    const lineWidthInput = root.querySelector('[data-ht-line-width]');
    const lineWidthValue = root.querySelector('[data-ht-line-width-value]');
    const colorInput = root.querySelector('[data-ht-ink-color]');
    const eraserSizeInput = root.querySelector('[data-ht-eraser-size]');
    const eraserSizeValue = root.querySelector('[data-ht-eraser-size-value]');
    const gestureHud = root.querySelector('[data-ht-gesture-hud]');
    const eraserBadge = root.querySelector('[data-ht-eraser-badge]');
    const pinchBadge = root.querySelector('[data-ht-pinch-badge]');
    const ctx = canvas.getContext('2d');
    const modeButtons = root.querySelectorAll('[data-ht-mode]');

    let stream = null;
    let rafId = null;
    let mode = 'air';
    let prevFrame = null;
    let inkLayer = null;
    let inkCtx = null;
    let inkPoint = null;
    let mpHands = null;
    let mpBusy = false;
    let mpReady = false;
    let mpResults = null;
    let pinchStableFrames = 0;
    let inkLineWidth = 4;
    let inkColor = '#62a8ff';
    let eraserSize = 20;
    let morphScene = null;
    let morphCamera = null;
    let morphRenderer = null;
    let morphMesh = null;
    let morphBasePositions = null;
    let morphCubePositions = null;
    let morphReady = false;
    let morphInitPromise = null;
    let morphTarget = 0;
    let morphCurrent = 0;
    let morphZoomTarget = 3.1;
    let morphLeftAnchor = null;
    let morphClock = 0;
    let morphFallback = false;
    let morphFallbackRotX = 0;
    let morphFallbackRotY = 0;
    let statusText = '';

    function hexToRgba(hex, alpha) {
      const clean = hex.replace('#', '');
      if (clean.length !== 6) {
        return 'rgba(98,168,255,' + alpha + ')';
      }
      const r = parseInt(clean.slice(0, 2), 16);
      const g = parseInt(clean.slice(2, 4), 16);
      const b = parseInt(clean.slice(4, 6), 16);
      return 'rgba(' + r + ',' + g + ',' + b + ',' + alpha + ')';
    }

    function syncInkControls() {
      if (lineWidthInput) {
        inkLineWidth = Number(lineWidthInput.value) || 4;
      }
      if (lineWidthValue) {
        lineWidthValue.textContent = String(inkLineWidth) + 'px';
      }
      if (colorInput && colorInput.value) {
        inkColor = colorInput.value;
      }

      if (eraserSizeInput) {
        eraserSize = Number(eraserSizeInput.value) || 20;
      }
      if (eraserSizeValue) {
        eraserSizeValue.textContent = String(eraserSize) + 'px';
      }
    }

    function setGestureBadge(badge, label, active) {
      if (!badge) {
        return;
      }
      const on = !!active;
      badge.classList.toggle('is-active', on);
      badge.textContent = label + ': ' + (on ? 'ON' : 'OFF');
    }

    function setEraserBadge(active) {
      setGestureBadge(eraserBadge, 'Eraser', active);
    }

    function setPinchBadge(active) {
      setGestureBadge(pinchBadge, 'Pinch Draw', active);
    }

    function clamp(v, min, max) {
      return Math.min(max, Math.max(min, v));
    }

    function setStatus(text) {
      if (!status || statusText === text) {
        return;
      }
      statusText = text;
      status.textContent = text;
    }

    function setMode(newMode) {
      mode = newMode;
      modeButtons.forEach((button) => {
        button.classList.toggle('is-active', button.getAttribute('data-ht-mode') === newMode);
      });

      if (newMode === 'air') {
        setStatus('Pinch Ink active. Bring thumb and index fingertip together to draw.');
      } else if (newMode === 'morph') {
        setStatus('Gesture 3D Morph Lab active. Left hand rotates/zooms, right fingertips morph.');
      }

      if (gestureHud) {
        gestureHud.classList.toggle('is-hidden', newMode !== 'air');
      }

      if (newMode !== 'air') {
        setEraserBadge(false);
        setPinchBadge(false);
      }
    }

    function stopLoop() {
      if (rafId !== null) {
        cancelAnimationFrame(rafId);
        rafId = null;
      }
    }

    function stopCamera() {
      stopLoop();
      if (stream) {
        stream.getTracks().forEach((track) => track.stop());
        stream = null;
      }
      video.srcObject = null;
      prevFrame = null;
      inkPoint = null;
      pinchStableFrames = 0;
      mpResults = null;
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      setEraserBadge(false);
      setPinchBadge(false);
      setStatus('Camera stopped.');
    }

    function ensureCanvasSize() {
      const width = video.videoWidth || 960;
      const height = video.videoHeight || 540;
      if (canvas.width !== width || canvas.height !== height) {
        canvas.width = width;
        canvas.height = height;
      }

      if (!inkLayer || inkLayer.width !== width || inkLayer.height !== height) {
        const nextLayer = document.createElement('canvas');
        nextLayer.width = width;
        nextLayer.height = height;
        const nextCtx = nextLayer.getContext('2d');

        if (inkLayer) {
          nextCtx.drawImage(inkLayer, 0, 0, width, height);
        }

        inkLayer = nextLayer;
        inkCtx = nextCtx;
      }
    }

    function clearInkLayer() {
      if (!inkLayer || !inkCtx) {
        return;
      }
      inkCtx.clearRect(0, 0, inkLayer.width, inkLayer.height);
      inkPoint = null;
      setStatus('Pinch Ink cleared.');
    }

    function drawLandmarkDot(x, y, color) {
      ctx.beginPath();
      ctx.arc(x, y, 5, 0, Math.PI * 2);
      ctx.fillStyle = color;
      ctx.fill();
    }

    function drawLandmarkLine(x1, y1, x2, y2, color) {
      ctx.beginPath();
      ctx.moveTo(x1, y1);
      ctx.lineTo(x2, y2);
      ctx.lineWidth = 2.5;
      ctx.strokeStyle = color;
      ctx.stroke();
    }

    function drawHandSkeleton(landmarks, w, h) {
      for (let i = 0; i < MP_HAND_CONNECTIONS.length; i += 1) {
        const pair = MP_HAND_CONNECTIONS[i];
        const a = landmarks[pair[0]];
        const b = landmarks[pair[1]];
        const ax = (1 - a.x) * w;
        const ay = a.y * h;
        const bx = (1 - b.x) * w;
        const by = b.y * h;
        drawLandmarkLine(ax, ay, bx, by, 'rgba(170, 198, 255, 0.38)');
      }
    }

    async function setupMediaPipe() {
      try {
        await ensureMediaPipeLoaded();
        mpHands = new window.Hands({
          locateFile: (file) => 'https://cdn.jsdelivr.net/npm/@mediapipe/hands/' + file,
        });
        mpHands.setOptions({
          maxNumHands: 2,
          modelComplexity: 1,
          minDetectionConfidence: 0.6,
          minTrackingConfidence: 0.6,
        });
        mpHands.onResults((results) => {
          mpResults = results;
          mpBusy = false;
        });
        mpReady = true;
      } catch (error) {
        mpReady = false;
        setStatus('MediaPipe failed to load. Check internet connection and refresh.');
      }
    }

    function drawMirrorPulse() {
      ensureCanvasSize();
      const w = canvas.width;
      const h = canvas.height;
      const t = performance.now() * 0.002;
      const glow = 0.45 + 0.25 * Math.sin(t * 2.3);

      ctx.save();
      ctx.translate(w, 0);
      ctx.scale(-1, 1);
      ctx.drawImage(video, 0, 0, w, h);
      ctx.restore();

      ctx.lineWidth = 10;
      ctx.strokeStyle = 'rgba(14, 165, 166,' + glow.toFixed(3) + ')';
      ctx.strokeRect(6, 6, w - 12, h - 12);
    }

    function drawEdgeNeon() {
      ensureCanvasSize();
      const w = canvas.width;
      const h = canvas.height;

      ctx.drawImage(video, 0, 0, w, h);
      const frame = ctx.getImageData(0, 0, w, h);
      const data = frame.data;
      const out = new Uint8ClampedArray(data.length);

      for (let y = 1; y < h - 1; y++) {
        for (let x = 1; x < w - 1; x++) {
          const idx = (y * w + x) * 4;
          const right = idx + 4;
          const left = idx - 4;
          const up = idx - w * 4;
          const down = idx + w * 4;

          const gx = Math.abs(data[right] - data[left]) + Math.abs(data[right + 1] - data[left + 1]) + Math.abs(data[right + 2] - data[left + 2]);
          const gy = Math.abs(data[down] - data[up]) + Math.abs(data[down + 1] - data[up + 1]) + Math.abs(data[down + 2] - data[up + 2]);
          const edge = Math.min(255, (gx + gy) * 0.6);

          out[idx] = 20;
          out[idx + 1] = Math.min(255, edge * 1.1);
          out[idx + 2] = Math.min(255, edge * 1.5);
          out[idx + 3] = 255;
        }
      }

      frame.data.set(out);
      ctx.putImageData(frame, 0, 0);
    }

    function drawMotionHeat() {
      ensureCanvasSize();
      const w = canvas.width;
      const h = canvas.height;

      ctx.drawImage(video, 0, 0, w, h);
      const frame = ctx.getImageData(0, 0, w, h);
      const data = frame.data;

      if (!prevFrame || prevFrame.length !== data.length) {
        prevFrame = new Uint8ClampedArray(data);
      }

      for (let i = 0; i < data.length; i += 4) {
        const dr = Math.abs(data[i] - prevFrame[i]);
        const dg = Math.abs(data[i + 1] - prevFrame[i + 1]);
        const db = Math.abs(data[i + 2] - prevFrame[i + 2]);
        const motion = Math.min(255, dr + dg + db);

        if (motion > 36) {
          data[i] = Math.min(255, data[i] + motion * 0.8);
          data[i + 1] = Math.max(0, data[i + 1] - motion * 0.35);
          data[i + 2] = Math.max(0, data[i + 2] - motion * 0.6);
        }

        prevFrame[i] = data[i];
        prevFrame[i + 1] = data[i + 1];
        prevFrame[i + 2] = data[i + 2];
      }

      ctx.putImageData(frame, 0, 0);
    }

    function drawPinchInk() {
      ensureCanvasSize();
      const w = canvas.width;
      const h = canvas.height;

      ctx.save();
      ctx.translate(w, 0);
      ctx.scale(-1, 1);
      ctx.drawImage(video, 0, 0, w, h);
      ctx.restore();

      if (!mpBusy && mpReady && mpHands && video.readyState >= 2) {
        mpBusy = true;
        mpHands.send({ image: video }).catch(() => {
          mpBusy = false;
        });
      }

      const handList = mpResults && mpResults.multiHandLandmarks;
      if (handList && handList.length > 0) {
        const landmarks = handList[0];
        const thumbTip = landmarks[4];
        const indexTip = landmarks[8];
        const wrist = landmarks[0];

        drawHandSkeleton(landmarks, w, h);

        function isFistPose() {
          const indexMcp = landmarks[5];
          const pinkyMcp = landmarks[17];
          const handSpan = Math.hypot(indexMcp.x - pinkyMcp.x, indexMcp.y - pinkyMcp.y) + 1e-6;
          const tips = [8, 12, 16, 20];
          const pips = [6, 10, 14, 18];
          let curled = 0;

          for (let i = 0; i < tips.length; i += 1) {
            const tip = landmarks[tips[i]];
            const pip = landmarks[pips[i]];
            const tipToWrist = Math.hypot(tip.x - wrist.x, tip.y - wrist.y);
            const pipToWrist = Math.hypot(pip.x - wrist.x, pip.y - wrist.y);
            if (tipToWrist < pipToWrist + handSpan * 0.18) {
              curled += 1;
            }
          }

          const thumbIp = landmarks[3];
          const thumbToWrist = Math.hypot(thumbTip.x - wrist.x, thumbTip.y - wrist.y);
          const thumbIpToWrist = Math.hypot(thumbIp.x - wrist.x, thumbIp.y - wrist.y);
          const thumbCurled = thumbToWrist < thumbIpToWrist + handSpan * 0.2;

          return curled >= 3 && thumbCurled;
        }

        const thumbX = (1 - thumbTip.x) * w;
        const thumbY = thumbTip.y * h;
        const indexX = (1 - indexTip.x) * w;
        const indexY = indexTip.y * h;
        const isFist = isFistPose();

        const dx = thumbTip.x - indexTip.x;
        const dy = thumbTip.y - indexTip.y;
        const pinchDistance = Math.sqrt(dx * dx + dy * dy);
        const isPinching = pinchDistance < 0.06;

        drawLandmarkLine(thumbX, thumbY, indexX, indexY, isFist ? 'rgba(255, 196, 87, 0.92)' : (isPinching ? 'rgba(72, 235, 160, 0.9)' : 'rgba(246, 86, 101, 0.9)'));
        drawLandmarkDot(thumbX, thumbY, '#ffd166');
        drawLandmarkDot(indexX, indexY, '#66b3ff');

        if (isPinching && !isFist) {
          pinchStableFrames += 1;
        } else {
          pinchStableFrames = 0;
        }

        const drawX = (thumbX + indexX) * 0.5;
        const drawY = (thumbY + indexY) * 0.5;
        const jitter = inkPoint ? Math.hypot(drawX - inkPoint.x, drawY - inkPoint.y) : 0;

        if (isFist) {
          const eraseX = (1 - landmarks[9].x) * w;
          const eraseY = landmarks[9].y * h;
          const eraserRadius = Math.max(8, eraserSize);

          inkCtx.save();
          inkCtx.globalCompositeOperation = 'destination-out';
          inkCtx.beginPath();
          inkCtx.arc(eraseX, eraseY, eraserRadius, 0, Math.PI * 2);
          inkCtx.fill();
          inkCtx.restore();

          ctx.beginPath();
          ctx.arc(eraseX, eraseY, eraserRadius, 0, Math.PI * 2);
          ctx.lineWidth = 2;
          ctx.strokeStyle = 'rgba(255, 210, 140, 0.9)';
          ctx.stroke();

          inkPoint = null;
          setEraserBadge(true);
          setPinchBadge(false);
          setStatus('Fist detected. Eraser active.');
        } else if (pinchStableFrames >= 2) {
          setEraserBadge(false);
          setPinchBadge(true);
          inkCtx.lineWidth = inkLineWidth;
          inkCtx.lineCap = 'round';
          inkCtx.lineJoin = 'round';
          inkCtx.strokeStyle = hexToRgba(inkColor, 0.95);
          inkCtx.shadowColor = hexToRgba(inkColor, 0.45);
          inkCtx.shadowBlur = 9;

          if (inkPoint && jitter < 90) {
            inkCtx.beginPath();
            inkCtx.moveTo(inkPoint.x, inkPoint.y);
            inkCtx.lineTo(drawX, drawY);
            inkCtx.stroke();
          } else {
            inkCtx.beginPath();
            inkCtx.arc(drawX, drawY, 2.5, 0, Math.PI * 2);
            inkCtx.fillStyle = hexToRgba(inkColor, 0.95);
            inkCtx.fill();
          }

          inkPoint = { x: drawX, y: drawY };
          setStatus('Pinch detected. Drawing active.');
        } else {
          setEraserBadge(false);
          setPinchBadge(false);
          inkPoint = null;
          setStatus('Show thumb and index fingertip, then pinch to draw.');
        }
      } else {
        pinchStableFrames = 0;
        inkPoint = null;
        setEraserBadge(false);
        setPinchBadge(false);
        setStatus('Searching for hand... keep thumb/index visible.');
      }

      ctx.drawImage(inkLayer, 0, 0);
    }

    async function setupMorphScene() {
      if (morphReady) {
        return;
      }

      await ensureThreeLoaded();
      const THREE = window.THREE;

      const offscreenCanvas = document.createElement('canvas');
      morphRenderer = new THREE.WebGLRenderer({ canvas: offscreenCanvas, antialias: true, alpha: true });
      morphRenderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));

      morphScene = new THREE.Scene();
      morphCamera = new THREE.PerspectiveCamera(45, 1, 0.1, 100);
      morphCamera.position.set(0, 0, morphZoomTarget);

      const key = new THREE.DirectionalLight(0xb7d2ff, 1.1);
      key.position.set(2.2, 1.7, 2.3);
      morphScene.add(key);
      morphScene.add(new THREE.AmbientLight(0x2a4a7a, 0.78));

      const geometry = new THREE.IcosahedronGeometry(1, 4);
      const material = new THREE.MeshStandardMaterial({
        color: 0x62a8ff,
        roughness: 0.32,
        metalness: 0.14,
        emissive: 0x11203a,
        emissiveIntensity: 0.7,
      });

      morphMesh = new THREE.Mesh(geometry, material);
      morphScene.add(morphMesh);

      const source = geometry.attributes.position.array;
      morphBasePositions = new Float32Array(source.length);
      morphCubePositions = new Float32Array(source.length);

      for (let i = 0; i < source.length; i += 3) {
        const x = source[i];
        const y = source[i + 1];
        const z = source[i + 2];
        morphBasePositions[i] = x;
        morphBasePositions[i + 1] = y;
        morphBasePositions[i + 2] = z;

        const maxAbs = Math.max(Math.abs(x), Math.abs(y), Math.abs(z), 1e-6);
        morphCubePositions[i] = (x / maxAbs) * 0.95;
        morphCubePositions[i + 1] = (y / maxAbs) * 0.95;
        morphCubePositions[i + 2] = (z / maxAbs) * 0.95;
      }

      morphReady = true;
    }

    function updateMorphMeshShape() {
      if (!morphMesh || !morphBasePositions || !morphCubePositions) {
        return;
      }

      morphClock += 0.016;
      morphCurrent += (morphTarget - morphCurrent) * 0.16;

      const pos = morphMesh.geometry.attributes.position.array;
      for (let i = 0; i < pos.length; i += 3) {
        const bx = morphBasePositions[i];
        const by = morphBasePositions[i + 1];
        const bz = morphBasePositions[i + 2];
        const cx = morphCubePositions[i];
        const cy = morphCubePositions[i + 1];
        const cz = morphCubePositions[i + 2];
        const wave = Math.sin((bx + by + bz) * 7 + morphClock * 1.8) * morphCurrent * 0.08;

        pos[i] = bx * (1 - morphCurrent) + cx * morphCurrent + bx * wave;
        pos[i + 1] = by * (1 - morphCurrent) + cy * morphCurrent + by * wave;
        pos[i + 2] = bz * (1 - morphCurrent) + cz * morphCurrent + bz * wave;
      }

      morphMesh.geometry.attributes.position.needsUpdate = true;
      morphMesh.geometry.computeVertexNormals();
    }

    function resetMorphModel() {
      morphTarget = 0;
      morphCurrent = 0;
      morphZoomTarget = 3.1;
      morphLeftAnchor = null;
      morphFallbackRotX = 0;
      morphFallbackRotY = 0;
      if (morphMesh) {
        morphMesh.rotation.set(0, 0, 0);
      }
      setStatus('Morph model reset.');
    }

    function drawMorphFallback(x, y, width, height) {
      const cx = x + width * 0.5;
      const cy = y + height * 0.5;
      const base = Math.min(width, height) * 0.28;
      const zoomScale = 3.1 / Math.max(1.35, morphZoomTarget);
      const r = base * zoomScale;

      morphCurrent += (morphTarget - morphCurrent) * 0.16;
      morphClock += 0.016;

      ctx.save();
      ctx.translate(cx, cy);
      ctx.rotate(morphFallbackRotY * 0.8);

      const sides = 24;
      ctx.beginPath();
      for (let i = 0; i <= sides; i += 1) {
        const t = (i / sides) * Math.PI * 2;
        const cubeWarp = morphCurrent * 0.38;
        const roundness = 1 + cubeWarp * Math.cos(4 * t);
        const wave = 1 + Math.sin(t * 6 + morphClock * 2.4) * morphCurrent * 0.1;
        const x = Math.cos(t) * r * roundness;
        const y = Math.sin(t) * r * wave * (1 - Math.min(0.3, Math.abs(morphFallbackRotX) * 0.4));
        if (i === 0) {
          ctx.moveTo(x, y);
        } else {
          ctx.lineTo(x, y);
        }
      }
      ctx.closePath();

      const grad = ctx.createLinearGradient(-r, -r, r, r);
      grad.addColorStop(0, '#8bc1ff');
      grad.addColorStop(1, '#2f5f99');
      ctx.fillStyle = grad;
      ctx.strokeStyle = 'rgba(206, 226, 255, 0.72)';
      ctx.lineWidth = 2;
      ctx.fill();
      ctx.stroke();

      ctx.restore();
    }

    function drawMorphMode() {
      ensureCanvasSize();
      const w = canvas.width;
      const h = canvas.height;
      const overlayPadding = 16;
      const overlayW = clamp(Math.floor(w * 0.34), 180, 340);
      const overlayH = clamp(Math.floor(overlayW * 0.72), 130, Math.floor(h * 0.48));
      const overlayX = overlayPadding;
      const overlayY = overlayPadding;

      if (!morphReady && !morphInitPromise) {
        morphInitPromise = setupMorphScene()
          .catch(() => {
            morphFallback = true;
            setStatus('3D overlay could not load (network/browser). Fallback Morph renderer is active.');
          })
          .finally(() => {
            morphInitPromise = null;
          });
      }

      ctx.fillStyle = '#040914';
      ctx.fillRect(0, 0, w, h);

      ctx.save();
      ctx.translate(w, 0);
      ctx.scale(-1, 1);
      ctx.drawImage(video, 0, 0, video.videoWidth || w, video.videoHeight || h, 0, 0, w, h);
      ctx.restore();

      if (morphReady) {
        morphRenderer.setSize(overlayW, overlayH, false);
        morphCamera.aspect = overlayW / Math.max(1, overlayH);
        morphCamera.updateProjectionMatrix();
        morphCamera.position.z += (morphZoomTarget - morphCamera.position.z) * 0.14;
        morphMesh.rotation.y += 0.003;
        updateMorphMeshShape();
        morphRenderer.render(morphScene, morphCamera);
        ctx.drawImage(morphRenderer.domElement, 0, 0, overlayW, overlayH, overlayX, overlayY, overlayW, overlayH);
      } else {
        drawMorphFallback(overlayX, overlayY, overlayW, overlayH);
        if (!morphFallback) {
          ctx.fillStyle = 'rgba(196, 214, 255, 0.85)';
          ctx.font = '600 16px Segoe UI, sans-serif';
          ctx.fillText('Loading 3D Morph Lab...', 20, 32);
        }
      }

      ctx.strokeStyle = 'rgba(210, 228, 255, 0.55)';
      ctx.lineWidth = 1.5;
      ctx.strokeRect(overlayX, overlayY, overlayW, overlayH);

      if (!mpBusy && mpReady && mpHands && video.readyState >= 2) {
        mpBusy = true;
        mpHands.send({ image: video }).catch(() => {
          mpBusy = false;
        });
      }

      let leftSeen = false;
      let rightSeen = false;

      const multiLandmarks = mpResults && mpResults.multiHandLandmarks;
      const multiHandedness = mpResults && mpResults.multiHandedness;
      if (multiLandmarks && multiLandmarks.length > 0) {
        for (let i = 0; i < multiLandmarks.length; i += 1) {
          const landmarks = multiLandmarks[i];
          const handed = multiHandedness && multiHandedness[i] && multiHandedness[i].label ? multiHandedness[i].label : 'Right';

          if (handed === 'Left') {
            leftSeen = true;
            const thumb = landmarks[4];
            const index = landmarks[8];
            const middle = landmarks[12];
            const tx = (1 - thumb.x) * w;
            const ty = thumb.y * h;
            const ix = (1 - index.x) * w;
            const iy = index.y * h;

            const pinchDist = Math.hypot(thumb.x - index.x, thumb.y - index.y);
            const pinch = pinchDist < 0.075;

            drawLandmarkLine(tx, ty, ix, iy, pinch ? 'rgba(72, 235, 160, 0.95)' : 'rgba(255, 109, 126, 0.95)');
            drawLandmarkDot(tx, ty, '#ffd166');
            drawLandmarkDot(ix, iy, '#66b3ff');

            const centerX = (tx + ix) * 0.5;
            const centerY = (ty + iy) * 0.5;
            if (pinch && (morphMesh || morphFallback)) {
              if (morphLeftAnchor) {
                const dx = centerX - morphLeftAnchor.x;
                const dy = centerY - morphLeftAnchor.y;
                if (morphMesh) {
                  morphMesh.rotation.y += dx * 0.012;
                  morphMesh.rotation.x += dy * 0.01;
                }
                morphFallbackRotY += dx * 0.012;
                morphFallbackRotX += dy * 0.01;
              }
              morphLeftAnchor = { x: centerX, y: centerY };
            } else {
              morphLeftAnchor = null;
            }

            const tm = Math.hypot(thumb.x - middle.x, thumb.y - middle.y);
            const zoomSignal = (pinchDist + tm) * 0.5;
            morphZoomTarget = clamp(1.45 + zoomSignal * 15.5, 1.35, 5.4);
          } else {
            rightSeen = true;
            const tipIds = [4, 8, 12, 16, 20];
            let cx = 0;
            let cy = 0;

            for (let t = 0; t < tipIds.length; t += 1) {
              const lm = landmarks[tipIds[t]];
              const px = (1 - lm.x) * w;
              const py = lm.y * h;
              cx += lm.x;
              cy += lm.y;
              drawLandmarkDot(px, py, '#ff9d6b', 4.6);
            }

            cx /= tipIds.length;
            cy /= tipIds.length;
            let spread = 0;
            for (let t = 0; t < tipIds.length; t += 1) {
              const lm = landmarks[tipIds[t]];
              spread += Math.hypot(lm.x - cx, lm.y - cy);
            }
            spread /= tipIds.length;
            morphTarget = clamp((spread - 0.04) / 0.16, 0, 1);
          }
        }
      }

      if (!leftSeen) {
        morphLeftAnchor = null;
      }

      if (leftSeen && rightSeen) {
        setStatus((morphFallback ? 'Fallback Morph: ' : 'Morph mode: ') + 'full camera with top-left 3D overlay. Left hand rotates/zooms, right fingertips morph shape.');
      } else if (leftSeen) {
        setStatus('Left hand detected. Pinch-drag rotates the top-left model, thumb/index/middle spacing zooms.');
      } else if (rightSeen) {
        setStatus('Right hand detected. Spread fingertips to morph the top-left model shape.');
      } else {
        setStatus('Show left hand for rotate/zoom and right fingertips to morph the top-left model.');
      }
    }

    function render() {
      if (!stream) {
        return;
      }

      if (mode === 'air') {
        drawPinchInk();
      } else {
        drawMorphMode();
      }

      rafId = requestAnimationFrame(render);
    }

    async function requestCamera() {
      try {
        stopCamera();
        setStatus('Requesting camera permission...');
        stream = await navigator.mediaDevices.getUserMedia({
          video: { width: { ideal: 1280 }, height: { ideal: 720 } },
          audio: false,
        });
        video.srcObject = stream;
        await video.play();
        setStatus('Camera active. Initializing hand tracking...');
        await setupMediaPipe();
        if (mpReady && mode === 'air') {
          setStatus('Pinch Ink ready. Show thumb/index and pinch to draw.');
        } else if (mpReady && mode === 'morph') {
          setStatus('Morph Lab ready. Left hand rotates/zooms, right hand morphs.');
        }
        render();
      } catch (error) {
        setStatus('Camera unavailable or permission denied. Check browser and OS camera permissions.');
      }
    }

    modeButtons.forEach((button) => {
      button.addEventListener('click', () => {
        setMode(button.getAttribute('data-ht-mode'));
      });
    });

    requestButton.addEventListener('click', requestCamera);
    stopButton.addEventListener('click', stopCamera);
    if (clearInkButton) {
      clearInkButton.addEventListener('click', () => {
        if (mode === 'morph') {
          resetMorphModel();
        } else {
          clearInkLayer();
        }
      });
    }
    if (lineWidthInput) {
      lineWidthInput.addEventListener('input', syncInkControls);
    }
    if (colorInput) {
      colorInput.addEventListener('input', syncInkControls);
    }
    if (eraserSizeInput) {
      eraserSizeInput.addEventListener('input', syncInkControls);
    }
    syncInkControls();
    setEraserBadge(false);
    setPinchBadge(false);
    setMode(mode);

    window.addEventListener('beforeunload', stopCamera);
  }

  function initMissionBoard(root) {
    const buttons = root.querySelectorAll('[data-ht-mission]');
    const status = root.querySelector('[data-ht-mission-status]');

    const labels = {
      pinch: 'Pinch Select. Try mapping this to your next UI click target.',
      fist: 'Fist Hold. Use this as a safe state for drag, lock, or clutch actions.',
      palm: 'Palm Reset. Use this to cancel and return to a neutral interaction state.',
    };

    function setMission(name) {
      buttons.forEach((button) => {
        button.classList.toggle('is-active', button.getAttribute('data-ht-mission') === name);
      });

      if (status) {
        status.textContent = 'Active mission: ' + (labels[name] || labels.pinch);
      }
    }

    buttons.forEach((button) => {
      button.addEventListener('click', () => {
        setMission(button.getAttribute('data-ht-mission'));
      });
    });

    setMission('pinch');
  }

  function init() {
    const demos = document.querySelectorAll('[data-ht-camera-demo]');
    demos.forEach((root) => initCameraDemo(root));

    const missionBoards = document.querySelectorAll('[data-ht-mission-board]');
    missionBoards.forEach((root) => initMissionBoard(root));
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
  } else {
    init();
  }
})();
