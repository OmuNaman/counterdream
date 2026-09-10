const $ = (id) => document.getElementById(id);
const canvas = $("canvas"),
  ctx = canvas.getContext("2d");
let ws = null,
  playing = false,
  busy = false,
  frame = 0,
  dx = 0,
  dy = 0,
  fire = false,
  drag = false,
  lastRequest = 0,
  streaming = false,
  qualityChosen = false,
  paceChosen = false,
  streamReady = false,
  lastFrameAt = 0,
  timer = null;
const keys = new Set();
let recorder = null, recordingStream = null, recordingTimer = null;
function stopRecording() {
  if (recorder?.state === 'recording') recorder.stop();
}
const keyMap = {
  KeyW: "w",
  KeyA: "a",
  KeyS: "s",
  KeyD: "d",
  Space: "space",
  ControlLeft: "ctrl",
  ShiftLeft: "shift",
  Digit1: "1",
  Digit2: "2",
  Digit3: "3",
  KeyR: "r",
};
function status(text, live = false) {
  $("status").textContent = text;
  document.querySelector(".live").classList.toggle("running", live);
}
function message(text, error = false) {
  $("message").textContent = text;
  $("message").classList.toggle("error", error);
}
function clearControls() {
  keys.clear();
  dx = dy = 0;
  fire = drag = false;
}
function pause() {
  stopRecording();
  playing = false;
  clearControls();
  clearTimeout(timer);
  lastFrameAt = 0;
  if (streaming && ws?.readyState === WebSocket.OPEN)
    ws.send(JSON.stringify({ type: "pause" }));
  $("pause").textContent = "RESUME";
  status("PAUSED");
}
function request() {
  if (!playing || (streaming ? !streamReady : busy) || ws?.readyState !== WebSocket.OPEN) return;
  if (ws.bufferedAmount > 4096) {
    if (streaming) timer = setTimeout(request, 1000 / Number($("pace").value));
    return;
  }
  const speed = Number($("turn-speed").value);
  const mx =
    (keys.has("ArrowLeft") ? -speed : 0) + (keys.has("ArrowRight") ? speed : 0) + dx;
  const my =
    (keys.has("ArrowUp") ? -speed : 0) + (keys.has("ArrowDown") ? speed : 0) + dy;
  ws.send(
    JSON.stringify({
      type: "step",
      keys: [...keys].map((k) => keyMap[k]).filter(Boolean),
      dx: Math.max(-1000, Math.min(1000, mx)),
      dy: Math.max(-200, Math.min(200, my)),
      fire: fire || keys.has("KeyF"),
      steps: Number($("quality").value),
      fps: Number($("pace").value),
    }),
  );
  dx = dy = 0;
  busy = !streaming;
  lastRequest = performance.now();
  if (streaming) timer = setTimeout(request, 1000 / Number($("pace").value));
}
function resume() {
  clearTimeout(timer);
  playing = true;
  $("pause").textContent = "PAUSE";
  $("overlay").classList.add("hidden");
  status("IMAGINING", true);
  $("viewport").focus();
  request();
}
async function connect() {
  if (ws?.readyState === WebSocket.CONNECTING) return;
  if (ws?.readyState === WebSocket.OPEN) {
    resume();
    return;
  }
  $("start").disabled = true;
  $("start").textContent = "CONNECTING…";
  status("CONNECTING");
  await loadInfo();
  streamReady = false;
  ws = new WebSocket(
    `${location.protocol === "https:" ? "wss:" : "ws:"}//${location.host}/ws?spawn=${encodeURIComponent($("spawn").value)}`,
  );
  ws.binaryType = "blob";
  ws.onopen = () => {
    loadInfo();
    $("start").disabled = false;
    $("pause").disabled = false;
    $("pause").textContent = "PAUSE";
    $("reset").disabled = false;
    $("record").disabled = !window.MediaRecorder;
    playing = true;
    busy = true;
    frame = 0;
    $("frame").textContent = "0000";
    $("overlay").classList.add("hidden");
    status(streaming ? "STARTING CLOUD GPU…" : "IMAGINING", true);
    $("viewport").focus();
  };
  ws.onmessage = async (event) => {
    if (typeof event.data === "string") {
      const data = JSON.parse(event.data);
      if (data.error) {
        pause();
        busy = false;
        message(data.error, true);
        return;
      }
      if (data.frame !== undefined) {
        frame = data.frame;
        $("frame").textContent = String(frame).padStart(4, "0");
      }
      if (data.ms !== undefined)
        $("latency").textContent = `${Math.round(data.ms)} ms`;
      if (data.remaining !== undefined)
        $("remaining").textContent =
          `${data.remaining.toLocaleString()} frames remaining`;
      return;
    }
    const bitmap = await createImageBitmap(event.data);
    if (canvas.width !== bitmap.width || canvas.height !== bitmap.height) {
      canvas.width = bitmap.width;
      canvas.height = bitmap.height;
    }
    ctx.drawImage(bitmap, 0, 0);
    bitmap.close();
    busy = false;
    if (streaming) {
      const now = performance.now();
      if (lastFrameAt) $("latency").textContent = `${Math.round(now - lastFrameAt)} ms / frame`;
      lastFrameAt = now;
      if (!streamReady) {
        streamReady = true;
        if (playing) {
          status("IMAGINING", true);
          request();
        }
      }
    } else if (playing)
      timer = setTimeout(
        request,
        Math.max(0, 1000 / Number($("pace").value) - (performance.now() - lastRequest)),
      );
  };
  ws.onerror = () => {
    message("Could not connect to the local inference server.", true);
  };
  ws.onclose = () => {
    pause();
    busy = false;
    streamReady = false;
    lastFrameAt = 0;
    ws = null;
    $("start").disabled = false;
    $("start").textContent = "RECONNECT ↗";
    $("overlay-title").textContent = "Connection closed.";
    $("overlay-copy").textContent = "Reconnect to start from a saved view.";
    $("overlay").classList.remove("hidden");
    status("DISCONNECTED");
    $("pause").disabled = true;
    $("reset").disabled = true;
    $("record").disabled = true;
  };
}
$("start").onclick = connect;
$("quality").onchange = () => { qualityChosen = true; };
$("pace").onchange = () => { paceChosen = true; };
$("display-mode").onchange = () => {
  canvas.style.imageRendering = $("display-mode").value === 'pixelated' ? 'pixelated' : 'auto';
};
async function toggleFullscreen() {
  try {
    if (document.fullscreenElement) await document.exitFullscreen();
    else await $("viewport").requestFullscreen();
  } catch { message('Fullscreen is unavailable in this browser.'); }
}
$("fullscreen").onclick = toggleFullscreen;
$("exit-fullscreen").onclick = toggleFullscreen;
$("record").onclick = () => {
  if (recorder?.state === 'recording') { stopRecording(); return; }
  if (!playing || !frame) { message('Resume the world before recording.'); return; }
  try {
    const mimeType = ['video/webm;codecs=vp9', 'video/webm;codecs=vp8', 'video/webm'].find(type => MediaRecorder.isTypeSupported(type));
    if (!mimeType) throw new Error('No supported recorder');
    recordingStream = canvas.captureStream(Number($("pace").value));
    recorder = new MediaRecorder(recordingStream, {mimeType, videoBitsPerSecond: 4000000});
    const chunks = [], firstFrame = frame;
    recorder.ondataavailable = event => { if (event.data.size) chunks.push(event.data); };
    recorder.onstop = () => {
      clearTimeout(recordingTimer);
      recordingStream.getTracks().forEach(track => track.stop());
      const url = URL.createObjectURL(new Blob(chunks, {type: mimeType}));
      const link = document.createElement('a');
      link.href = url;
      link.download = `counterdream-session-frames-${firstFrame}-${frame}.webm`;
      link.click();
      setTimeout(() => URL.revokeObjectURL(url), 60000);
      $("record").textContent = 'RECORD 30s';
      message('Session clip saved. This records the displayed model output.');
    };
    recorder.start(1000);
    recordingTimer = setTimeout(stopRecording, 30000);
    $("record").textContent = 'STOP RECORDING';
    message('Recording your session for up to 30 seconds. Pausing or resetting ends the clip.');
    $("viewport").focus();
  } catch {
    recordingStream?.getTracks().forEach(track => track.stop());
    message('Recording is unavailable in this browser.', true);
  }
};
$("pause").onclick = () => (playing ? pause() : resume());
function reset() {
  if (!ws || ws.readyState !== WebSocket.OPEN) return;
  pause();
  busy = true;
  ws.send(JSON.stringify({ type: "reset", spawn: Number($("spawn").value) }));
  message("World reset. Press Resume when ready.");
}
$("reset").onclick = reset;
$("spawn").onchange = () => {
  if (ws) reset();
};
window.addEventListener("keydown", (e) => {
  if (e.target.matches("select")) return;
  if (e.code === "Escape") {
    pause();
    return;
  }
  if (e.code === "Enter" && !playing) {
    connect();
    e.preventDefault();
    return;
  }
  if (
    playing &&
    (keyMap[e.code] || e.code.startsWith("Arrow") || e.code === "KeyF")
  ) {
    e.preventDefault();
    keys.add(e.code);
  }
});
window.addEventListener("keyup", (e) => keys.delete(e.code));
window.addEventListener("blur", pause);
document.addEventListener("visibilitychange", () => {
  if (document.hidden) pause();
});
$("viewport").addEventListener("pointerdown", (e) => {
  if (e.target.closest("button")) return;
  if (!playing) return;
  $("viewport").focus();
  fire = e.button === 0;
  drag = true;
});
window.addEventListener("pointerup", () => {
  fire = drag = false;
});
$("viewport").addEventListener("pointermove", (e) => {
  if (drag && playing) {
    dx += e.movementX;
    dy += e.movementY;
  }
});
$("viewport").addEventListener("contextmenu", (e) => e.preventDefault());
function loadInfo() {
  const selected = Number($("spawn").value) || 0;
  return fetch("/api/info")
  .then((r) => r.json())
  .then((info) => {
    streaming = Boolean(info.streaming);
    if (info.recommended_fps && !playing && !paceChosen)
      $("pace").value = String(info.recommended_fps);
    if (info.recommended_steps && !playing && !qualityChosen)
      $("quality").value = String(info.recommended_steps);
    $("spawn").replaceChildren(
      ...info.spawns.map((name, i) =>
        Object.assign(document.createElement("option"), {
          value: i,
          textContent: name,
        }),
      ),
    );
    $("spawn").value = String(Math.min(selected, info.spawns.length - 1));
    if (info.resolution) {
      $("resolution-copy").textContent = `${info.resolution[0]} × ${info.resolution[1]} model output, enlarged for viewing.`;
      if (info.display_note) $("resolution-copy").textContent += ` ${info.display_note}`;
    }
    if (info.quality_note && !playing) message(info.quality_note);
    if (info.context_frames) {
      $("seed-copy").textContent = `Reset loads ${info.context_frames} recorded frames; subsequent frames are generated by the model.`;
    }
    if (info.session_note) {
      $("session-note").textContent = info.session_note;
      $("session-note").hidden = false;
    }
    $("checkpoint").textContent = info.checkpoint_step
      ? `${Number(info.checkpoint_step).toLocaleString()} training steps · ${info.device || "GPU"}`
      : "Checkpoint ready";
    $("remaining").textContent =
      `${Math.max(0, info.budget_frames - info.generated_frames).toLocaleString()} frames remaining`;
  })
  .catch(() => message("Server metadata could not be loaded.", true));
}
loadInfo();
