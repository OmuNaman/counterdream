const {test} = require('node:test');
const assert = require('node:assert/strict');
const {LatestFrame,Inputs} = require('../counterdream/static/client-core.js');
const flush = () => new Promise(resolve=>setImmediate(resolve));
test('held movement survives mouse sends; releases and blur clear every input', () => {
  const input = new Inputs();
  input.down('KeyW'); input.down('KeyF'); input.down('ArrowRight'); input.dx=5;
  assert.deepEqual(input.snapshot().keys,['w']);
  assert.equal(input.snapshot().fire,true);
  assert.equal(input.snapshot().look_x,10);
  input.sent();
  assert.equal(input.snapshot().dx,0);
  assert.equal(input.snapshot().look_x,10);
  assert.equal(input.down('KeyW'),false);
  input.up('KeyW'); assert.deepEqual(input.snapshot().keys,[]);
  input.pointers.set(3,'KeyA'); input.clear();
  assert.deepEqual(input.snapshot(),{keys:[],fire:false,scope:false,look_x:0,look_y:0,dx:0,dy:0});
});
test('touch controls support simultaneous movement and fire, including independent release', () => {
  const input = new Inputs();
  input.pointers.set(1,'KeyW'); input.pointers.set(2,'KeyF');
  assert.equal(input.snapshot().fire,true); assert.deepEqual(input.snapshot().keys,['w']);
  input.pointers.delete(2); assert.equal(input.snapshot().fire,false); assert.deepEqual(input.snapshot().keys,['w']);
});
test('a burst never creates an unbounded decode queue or mismatches pixels and metadata', async () => {
  const decoded=[], drawn=[], scheduled=[], closed=[];
  let release;
  const queue=new LatestFrame({decode:blob=>{decoded.push(blob); return new Promise(r=>{release=()=>r({id:blob,close:()=>closed.push(blob)});});},
    schedule:cb=>scheduled.push(cb),draw:(bitmap,meta)=>drawn.push([bitmap.id,meta.frame]),error:assert.fail});
  for(let frame=1;frame<=100;frame++)queue.push(frame,{frame});
  assert.deepEqual(decoded,[1]);
  release();await flush();
  assert.deepEqual(decoded,[1,100]);
  release();await flush();scheduled.shift()();
  assert.deepEqual(drawn,[[100,100]]);
  assert.deepEqual(closed,[1,100]);
  assert.equal(queue.dropped,99);
});
test('reset cannot paint a late frame from the previous world', async () => {
  let release;
  const drawn=[],closed=[],scheduled=[];
  const queue=new LatestFrame({decode:blob=>new Promise(r=>{release=()=>r({id:blob,close:()=>closed.push(blob)});}),
    schedule:cb=>scheduled.push(cb),draw:b=>drawn.push(b.id),error:assert.fail});
  queue.push('old',{});queue.clear();queue.push('reset',{});
  release();await flush();release();await flush();scheduled.shift()();
  assert.deepEqual(drawn,['reset']);assert.deepEqual(closed,['old','reset']);
});
test('a failed image decode does not freeze the rest of the stream', async () => {
  const errors=[],drawn=[],scheduled=[];
  const queue=new LatestFrame({decode:async blob=>{if(blob==='bad')throw Error('decode');return {id:blob,close(){}};},
    schedule:cb=>scheduled.push(cb),draw:b=>drawn.push(b.id),error:e=>errors.push(e.message)});
  queue.push('bad',{});queue.push('good',{});await flush();scheduled.shift()();
  assert.deepEqual(drawn,['good']);assert.deepEqual(errors,['decode']);
});
