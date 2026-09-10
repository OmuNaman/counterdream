const $ = id => document.getElementById(id);
const canvas = $('canvas'), ctx = canvas.getContext('2d', {alpha:false});
const input = new CounterDreamClient.Inputs();
let ws = null, playing = false, busy = false, frame = 0, streaming = false;
let streamReady = false, qualityChosen = false, paceChosen = false;
let timer = null, lastRequest = 0, inputId = 0, metadata = {}, awaitingReset = false;
let lookDrag = false, mousePending = false, lastReceived = 0, lastPaint = 0, painted = [], responseTimes = [];
let recorder = null, recordingStream = null, recordingTimer = null;
function status(text, live = false) {
  $('status').textContent = text;
  document.querySelector('.live').classList.toggle('running', live);
}
function message(text, error = false) {
  $('message').textContent = text; $('message').classList.toggle('error', error);
}
function describe(control) {
  if (!control) return 'WAITING';
  const labels = (control.keys || []).map(key => ({w:'FORWARD',s:'BACK',a:'LEFT',d:'RIGHT',space:'JUMP',r:'RELOAD'})[key] || key.toUpperCase());
  if (control.fire) labels.push('FIRE');
  if (control.scope) labels.push('SCOPE');
  if (control.dx || control.dy || control.look_x || control.look_y) labels.push('LOOK');
  return labels.join(' + ') || 'IDLE';
}
function showInput() {
  $('input-local').textContent = playing ? describe(input.snapshot(Number($('turn-speed').value))) : 'PAUSED';
  for (const button of document.querySelectorAll('[data-control]')) {
    button.disabled = !playing || (streaming && !streamReady);
    const active = playing && (input.keys.has(button.dataset.control) || [...input.pointers.values()].includes(button.dataset.control));
    button.classList.toggle('pressed', active); button.setAttribute('aria-pressed', String(active));
  }
}
function stopRecording() { if (recorder?.state === 'recording') recorder.stop(); }
function clearControls() { input.clear(); lookDrag = false; showInput(); }
function pause() {
  stopRecording(); playing = false; clearControls(); clearTimeout(timer); painted = [];
  if (document.pointerLockElement) document.exitPointerLock();
  if (ws?.readyState === WebSocket.OPEN && streaming) ws.send(JSON.stringify({type:'pause'}));
  $('pause').textContent = 'RESUME';
  if (ws?.readyState === WebSocket.OPEN) {
    $('overlay-title').textContent = 'Paused.';
    $('overlay-copy').textContent = 'Resume, then use W A S D or the buttons below to move.';
    $('start').textContent = 'RESUME & PLAY ↗'; $('start').disabled = false;
    $('overlay').classList.remove('hidden'); status('PAUSED');
  }
}
function request() {
  clearTimeout(timer);
  if (!playing || ws?.readyState !== WebSocket.OPEN) return;
  if ((streaming ? streamReady : !busy) && ws.bufferedAmount < 4096) {
    const control = input.snapshot(Number($('turn-speed').value));
    lastRequest = performance.now();
    ws.send(JSON.stringify({type:'step', ...control, steps:Number($('quality').value),
      fps:Number($('pace').value), input_id:++inputId, client_time_ms:lastRequest}));
    input.sent(); busy = !streaming;
  }
  // The server holds controls between heartbeats, independently of rendering.
  if (streaming) timer = setTimeout(request, 50);
}
function changed() { showInput(); request(); }
function resume() {
  playing = true; painted = []; $('pause').textContent = 'PAUSE';
  $('overlay').classList.add('hidden');
  status(streamReady || !streaming ? 'LIVE' : 'STARTING CLOUD GPU…', true);
  $('viewport').focus({preventScroll:true}); showInput(); request();
}
const frames = new CounterDreamClient.LatestFrame({
  decode: blob => createImageBitmap(blob), schedule: callback => requestAnimationFrame(callback),
  draw(bitmap, info) {
    if (canvas.width !== bitmap.width || canvas.height !== bitmap.height) { canvas.width = bitmap.width; canvas.height = bitmap.height; }
    ctx.drawImage(bitmap, 0, 0); lastPaint = performance.now(); busy = false;
    frame = info.frame ?? frame; $('frame').textContent = String(frame).padStart(4, '0');
    if (frame && playing) {
      painted.push(lastPaint);
      while (painted.length > 1 && painted[0] < lastPaint - 2000) painted.shift();
    }
    if (info.applied) {
      const action = describe(info.applied);
      $('input-cloud').textContent = action;
      if (action !== 'IDLE') $('last-action').textContent = `Last action: ${action}`;
    }
    if (info.client_time_ms > 0) {
      const delay = lastPaint - info.client_time_ms;
      if (delay >= 0 && delay < 10000) { responseTimes.push(delay); if (responseTimes.length > 50) responseTimes.shift(); }
    }
    if (streaming && !streamReady) { streamReady = true; loadInfo(); if (playing) resume(); }
    else if (!streaming && playing) timer = setTimeout(request, Math.max(0,1000/Number($('pace').value)-(performance.now()-lastRequest)));
  },
  error() { message('A video frame could not be decoded. Waiting for the next frame.', true); }
});
setInterval(() => {
  const now = performance.now();
  const fps = painted.length > 1 && now - lastPaint < 1000 ? (painted.length-1)*1000/(painted.at(-1)-painted[0]) : 0;
  $('latency').textContent = `${fps.toFixed(1)} FPS`;
  const sorted = [...responseTimes].sort((a,b)=>a-b);
  $('response').textContent = sorted.length ? `${Math.round(sorted[Math.floor(sorted.length/2)])} ms response` : 'Waiting for cloud';
  if (playing && streamReady) status(now-lastReceived > 2500 ? 'WAITING FOR FRAMES' : 'LIVE', now-lastReceived <= 2500);
}, 500);
async function connect() {
  if (ws?.readyState === WebSocket.CONNECTING) return;
  if (ws?.readyState === WebSocket.OPEN) { resume(); return; }
  $('start').disabled = true; $('start').textContent = 'CONNECTING…'; status('CONNECTING');
  await loadInfo(); frames.clear(); metadata = {}; awaitingReset = false;
  streamReady = false; responseTimes = []; painted = [];
  const socket = new WebSocket(`${location.protocol === 'https:' ? 'wss:' : 'ws:'}//${location.host}/ws?spawn=${encodeURIComponent($('spawn').value)}`);
  ws = socket; socket.binaryType = 'blob';
  socket.onopen = () => {
    if (ws !== socket) return;
    $('pause').disabled = $('reset').disabled = false; $('record').disabled = !window.MediaRecorder;
    playing = true; busy = true; frame = 0; $('frame').textContent = '0000'; showInput();
    $('overlay-title').textContent = 'Loading your world…';
    $('overlay-copy').textContent = 'The cloud GPU is getting ready. Controls start when the image appears.';
    $('start').textContent = 'LOADING…'; status(streaming ? 'STARTING CLOUD GPU…' : 'LOADING', true);
    $('viewport').focus({preventScroll:true});
  };
  socket.onmessage = event => {
    if (ws !== socket) return;
    lastReceived = performance.now();
    if (typeof event.data === 'string') {
      const data = JSON.parse(event.data);
      if (data.error) { pause(); busy = false; message(data.error, true); return; }
      if (data.reset) { frames.clear(); awaitingReset = false; painted = []; $('last-action').textContent = 'Last action: —'; }
      metadata = data;
      if (data.remaining !== undefined) $('remaining').textContent = `${data.remaining.toLocaleString()} frames remaining`;
    } else if (!awaitingReset) {
      frames.push(event.data, metadata);
      if (!streaming && playing) { $('overlay').classList.add('hidden'); status('LIVE',true); }
    }
  };
  socket.onerror = () => { if (ws === socket) message('Connection interrupted. Reconnect to restart the stream.', true); };
  socket.onclose = () => {
    if (ws !== socket) return;
    pause(); frames.clear(); busy = false; streamReady = false; ws = null;
    $('start').disabled = false; $('start').textContent = 'RECONNECT ↗';
    $('overlay-title').textContent = 'Connection closed.'; $('overlay-copy').textContent = 'Reconnect to start from a saved view.';
    $('overlay').classList.remove('hidden');
    $('pause').disabled = $('reset').disabled = $('record').disabled = true; status('DISCONNECTED');
    if (!$('message').classList.contains('error')) message('The connection ended. Press Reconnect to start from a saved view.');
  };
}
function reset() {
  if (ws?.readyState !== WebSocket.OPEN) return;
  pause(); frames.clear(); awaitingReset = true; busy = true;
  ws.send(JSON.stringify({type:'reset',spawn:Number($('spawn').value)}));
  message('World reset. Press Resume when ready.');
}
$('start').onclick = connect; $('pause').onclick = () => playing ? pause() : resume(); $('reset').onclick = reset;
$('spawn').onchange = () => { if (ws) reset(); };
$('quality').onchange = () => { qualityChosen = true; $('viewport').focus({preventScroll:true}); changed(); };
$('pace').onchange = () => { paceChosen = true; $('viewport').focus({preventScroll:true}); changed(); };
$('turn-speed').onchange = () => { $('viewport').focus({preventScroll:true}); changed(); };
$('display-mode').onchange = () => { canvas.style.imageRendering = $('display-mode').value === 'pixelated' ? 'pixelated' : 'auto'; $('viewport').focus({preventScroll:true}); };
window.addEventListener('keydown', event => {
  if (event.target.matches('select,input,textarea') || event.metaKey || event.altKey) return;
  if (event.code === 'Escape') { pause(); return; }
  if (event.code === 'Enter' && !playing) { connect(); event.preventDefault(); return; }
  if (playing && input.accepts(event.code)) { event.preventDefault(); if (input.down(event.code)) changed(); }
});
window.addEventListener('keyup', event => { if (input.up(event.code)) { event.preventDefault(); changed(); } });
window.addEventListener('blur', pause);
document.addEventListener('visibilitychange', () => { if (document.hidden) pause(); });
for (const button of document.querySelectorAll('[data-control]')) {
  let lastPointer = -Infinity;
  button.addEventListener('pointerdown', event => {
    if (!playing) return;
    lastPointer = performance.now();
    event.preventDefault(); button.setPointerCapture(event.pointerId);
    input.pointers.set(event.pointerId, button.dataset.control); changed();
  });
  const release = event => { if (input.pointers.delete(event.pointerId)) changed(); };
  button.addEventListener('pointerup', release); button.addEventListener('pointercancel', release); button.addEventListener('lostpointercapture', release);
  // Keyboard/assistive activation can emit click without pointerdown/up.
  button.addEventListener('click', () => {
    if (!playing || performance.now()-lastPointer < 350) return;
    const id = Symbol('button-tap');
    input.pointers.set(id, button.dataset.control); changed();
    setTimeout(() => { input.pointers.delete(id); changed(); }, 100);
  });
}
$('viewport').addEventListener('pointerdown', event => {
  if (!playing || event.target.closest('button')) return;
  event.preventDefault(); $('viewport').focus({preventScroll:true});
  if (event.button === 0) input.fire = true;
  if (event.button === 2 && !document.pointerLockElement) lookDrag = true;
  changed();
});
window.addEventListener('pointerup', event => {
  if (event.button === 0) input.fire = false;
  if (event.button === 2) lookDrag = false;
  if (playing) changed();
});
document.addEventListener('pointermove', event => {
  if (playing && (lookDrag || document.pointerLockElement === $('viewport'))) {
    const sensitivity = Number($('turn-speed').value)/20;
    input.dx += event.movementX*sensitivity; input.dy += event.movementY*sensitivity;
    if (!mousePending) { mousePending = true; requestAnimationFrame(() => { mousePending = false; changed(); }); }
  }
});
$('viewport').addEventListener('contextmenu', event => event.preventDefault());
$('capture-mouse').onclick = async () => {
  if (!playing) { message('Resume the world before capturing the mouse.'); return; }
  try { await $('viewport').requestPointerLock(); }
  catch { message('Mouse capture is unavailable here. Hold the right mouse button to look, or use the arrow keys.'); }
};
document.addEventListener('pointerlockchange', () => { $('capture-mouse').textContent = document.pointerLockElement ? 'MOUSE CAPTURED · ESC' : 'CAPTURE MOUSE'; });
document.addEventListener('pointerlockerror', () => message('Use right-drag or arrow keys to look; mouse capture is unavailable in this browser.'));
async function toggleFullscreen() {
  try { if (document.fullscreenElement) await document.exitFullscreen(); else await $('viewport').requestFullscreen(); }
  catch { message('Fullscreen is unavailable in this browser.'); }
}
$('fullscreen').onclick = toggleFullscreen; $('exit-fullscreen').onclick = toggleFullscreen;
$('record').onclick = () => {
  if (recorder?.state === 'recording') { stopRecording(); return; }
  if (!playing || !frame) { message('Resume the world before recording.'); return; }
  try {
    const mimeType = ['video/webm;codecs=vp9','video/webm;codecs=vp8','video/webm'].find(type => MediaRecorder.isTypeSupported(type));
    if (!mimeType) throw new Error('No recorder');
    recordingStream = canvas.captureStream(Number($('pace').value));
    recorder = new MediaRecorder(recordingStream,{mimeType,videoBitsPerSecond:4000000});
    const chunks = [], firstFrame = frame;
    recorder.ondataavailable = event => { if (event.data.size) chunks.push(event.data); };
    recorder.onstop = () => {
      clearTimeout(recordingTimer); recordingStream.getTracks().forEach(track=>track.stop());
      const url = URL.createObjectURL(new Blob(chunks,{type:mimeType}));
      const link = document.createElement('a'); link.href = url;
      link.download = `counterdream-session-frames-${firstFrame}-${frame}.webm`; link.click();
      setTimeout(()=>URL.revokeObjectURL(url),60000);
      $('record').textContent = 'RECORD 30s'; message('Session clip saved. This records the displayed model output.');
    };
    recorder.start(1000); recordingTimer = setTimeout(stopRecording,30000); $('record').textContent = 'STOP RECORDING';
    message('Recording for up to 30 seconds. Pausing or resetting ends the clip.'); $('viewport').focus({preventScroll:true});
  } catch { recordingStream?.getTracks().forEach(track=>track.stop()); message('Recording is unavailable in this browser.',true); }
};
async function loadInfo() {
  try {
    const selected = Number($('spawn').value)||0, info = await (await fetch('/api/info')).json();
    streaming = Boolean(info.streaming);
    if (info.recommended_fps && !playing && !paceChosen) $('pace').value = String(info.recommended_fps);
    if (info.recommended_steps && !playing && !qualityChosen) $('quality').value = String(info.recommended_steps);
    $('spawn').replaceChildren(...info.spawns.map((name,i)=>Object.assign(document.createElement('option'),{value:i,textContent:name})));
    $('spawn').value = String(Math.min(selected,info.spawns.length-1));
    if (info.resolution) $('resolution-copy').textContent = `${info.resolution[0]} × ${info.resolution[1]} native prediction. ${info.display_note||'Enlarged for display.'}`;
    if (info.quality_note && !playing) message(info.quality_note);
    if (info.context_frames) $('seed-copy').textContent = `Reset loads ${info.context_frames} recorded frames; subsequent frames are generated by the model.`;
    if (info.session_note) { $('session-note').textContent = info.session_note; $('session-note').hidden = false; }
    $('checkpoint').textContent = info.checkpoint_step ? `${Number(info.checkpoint_step).toLocaleString()} training steps · ${info.device||'GPU'}` : 'Checkpoint ready';
    if (!streamReady) $('remaining').textContent = `${Math.max(0,info.budget_frames-info.generated_frames).toLocaleString()} frames remaining`;
  } catch { message('The local server is unavailable. Reconnect after it restarts.',true); }
}
showInput(); loadInfo();
