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
  timer = null;
const keys = new Set();
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
  playing = false;
  clearControls();
  clearTimeout(timer);
  $("pause").textContent = "RESUME";
  status("PAUSED");
}
function request() {
  if (!playing || busy || ws?.readyState !== WebSocket.OPEN) return;
  const mx =
    (keys.has("ArrowLeft") ? -60 : 0) + (keys.has("ArrowRight") ? 60 : 0) + dx;
  const my =
    (keys.has("ArrowUp") ? -50 : 0) + (keys.has("ArrowDown") ? 50 : 0) + dy;
  ws.send(
    JSON.stringify({
      type: "step",
      keys: [...keys].map((k) => keyMap[k]).filter(Boolean),
      dx: Math.max(-1000, Math.min(1000, mx)),
      dy: Math.max(-200, Math.min(200, my)),
      fire: fire || keys.has("KeyF"),
      steps: Number($("quality").value),
    }),
  );
  dx = dy = 0;
  busy = true;
  lastRequest = performance.now();
}
function resume() {
  playing = true;
  $("pause").textContent = "PAUSE";
  $("overlay").classList.add("hidden");
  status("IMAGINING", true);
  $("viewport").focus();
  request();
}
function connect() {
  if (ws?.readyState === WebSocket.CONNECTING) return;
  if (ws?.readyState === WebSocket.OPEN) {
    resume();
    return;
  }
  $("start").disabled = true;
  $("start").textContent = "CONNECTING…";
  status("CONNECTING");
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
    playing = true;
    busy = true;
    frame = 0;
    $("frame").textContent = "0000";
    $("overlay").classList.add("hidden");
    status("IMAGINING", true);
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
    ctx.drawImage(bitmap, 0, 0);
    bitmap.close();
    busy = false;
    if (playing)
      timer = setTimeout(
        request,
        Math.max(0, 125 - (performance.now() - lastRequest)),
      );
  };
  ws.onerror = () => {
    message("Could not connect to the local inference server.", true);
  };
  ws.onclose = () => {
    pause();
    busy = false;
    ws = null;
    $("start").disabled = false;
    $("start").textContent = "RECONNECT ↗";
    $("overlay-title").textContent = "Connection closed.";
    $("overlay-copy").textContent = "Reconnect to start from a saved view.";
    $("overlay").classList.remove("hidden");
    status("DISCONNECTED");
    $("pause").disabled = true;
    $("reset").disabled = true;
  };
}
$("start").onclick = connect;
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
    $("spawn").replaceChildren(
      ...info.spawns.map((name, i) =>
        Object.assign(document.createElement("option"), {
          value: i,
          textContent: name,
        }),
      ),
    );
    $("spawn").value = String(Math.min(selected, info.spawns.length - 1));
    $("checkpoint").textContent = info.checkpoint_step
      ? `${Number(info.checkpoint_step).toLocaleString()} training steps · ${info.device || "GPU"}`
      : "Checkpoint ready";
    $("remaining").textContent =
      `${Math.max(0, info.budget_frames - info.generated_frames).toLocaleString()} frames remaining`;
  })
  .catch(() => message("Server metadata could not be loaded.", true));
}
loadInfo();
