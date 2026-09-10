/* Bounded frame decoding and physical inputs, shared with the headless tests. */
(function (root) {
  class LatestFrame {
    constructor({ decode, schedule, draw, error }) {
      Object.assign(this, { decode, schedule, draw, error });
      this.epoch = 0;
      this.pending = this.ready = null;
      this.decoding = this.scheduled = false;
      this.dropped = 0;
    }
    clear() {
      this.epoch++;
      this.pending = null;
      this.ready?.bitmap.close();
      this.ready = null;
    }
    push(blob, metadata) {
      if (this.pending) this.dropped++;
      this.pending = { blob, metadata, epoch: this.epoch };
      this.pump();
    }
    async pump() {
      if (this.decoding || !this.pending) return;
      this.decoding = true;
      const item = this.pending;
      this.pending = null;
      try {
        const bitmap = await this.decode(item.blob);
        if (item.epoch !== this.epoch) bitmap.close();
        else {
          if (this.ready) { this.ready.bitmap.close(); this.dropped++; }
          this.ready = { bitmap, metadata: item.metadata };
          if (!this.scheduled) {
            this.scheduled = true;
            this.schedule(() => {
              this.scheduled = false;
              const ready = this.ready;
              this.ready = null;
              if (ready) {
                try { this.draw(ready.bitmap, ready.metadata); }
                finally { ready.bitmap.close(); }
              }
            });
          }
        }
      } catch (error) {
        if (item.epoch === this.epoch) this.error(error);
      } finally {
        this.decoding = false;
        this.pump();
      }
    }
  }
  const keyMap = { KeyW:'w', KeyA:'a', KeyS:'s', KeyD:'d', Space:'space', ControlLeft:'ctrl',
    ControlRight:'ctrl', ShiftLeft:'shift', ShiftRight:'shift', Digit1:'1', Digit2:'2', Digit3:'3', KeyR:'r' };
  class Inputs {
    constructor() { this.clear(); }
    clear() { this.keys = new Set(); this.pointers = new Map(); this.dx = this.dy = 0; this.fire = this.scope = false; }
    accepts(code) { return Boolean(keyMap[code]) || ['ArrowLeft','ArrowRight','ArrowUp','ArrowDown','KeyF'].includes(code); }
    down(code) { const changed = !this.keys.has(code); if (this.accepts(code)) this.keys.add(code); return changed; }
    up(code) { return this.keys.delete(code); }
    snapshot(speed = 10) {
      const keys = new Set([...this.keys, ...this.pointers.values()]);
      return {
        keys: [...new Set([...keys].map(code => keyMap[code]).filter(Boolean))],
        fire: this.fire || keys.has('KeyF'), scope: this.scope,
        look_x: ((keys.has('ArrowRight') ? 1 : 0) - (keys.has('ArrowLeft') ? 1 : 0)) * speed,
        look_y: ((keys.has('ArrowDown') ? 1 : 0) - (keys.has('ArrowUp') ? 1 : 0)) * speed,
        dx: Math.max(-1000, Math.min(1000, this.dx)), dy: Math.max(-200, Math.min(200, this.dy)),
      };
    }
    sent() { this.dx = this.dy = 0; }
  }
  const api = { LatestFrame, Inputs };
  if (typeof module !== 'undefined' && module.exports) module.exports = api;
  else root.CounterDreamClient = api;
})(globalThis);
