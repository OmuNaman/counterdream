const {test}=require('node:test');
const assert=require('node:assert/strict');
const fs=require('node:fs');
const vm=require('node:vm');
function app() {
  const elements=new Map(),events={},docEvents={},sent=[];
  function element(id) {
    if (!elements.has(id)) elements.set(id,{
      value:({quality:'4',pace:'24','turn-speed':'10',spawn:'0'})[id]||'',textContent:'',disabled:false,
      style:{},classList:{toggle(){},add(){},remove(){},contains(){return false;}},dataset:{},
      focus(){},replaceChildren(){},matches(){return false;},addEventListener(){},
      getContext(){return {drawImage(){}};},setAttribute(){}
    });
    return elements.get(id);
  }
  const scope={CounterDreamClient:require('../counterdream/static/client-core.js'),
    document:{getElementById:element,querySelector:()=>element('live'),querySelectorAll:()=>[],
      addEventListener:(name,fn)=>docEvents[name]=fn,createElement:()=>({})},
    window:{addEventListener:(name,fn)=>events[name]=fn},performance:{now:()=>100},
    WebSocket:{OPEN:1},setTimeout:()=>1,clearTimeout(){},setInterval(){},
    fetch:async()=>({json:async()=>({streaming:true,spawns:['One'],budget_frames:100,generated_frames:0})}),
    requestAnimationFrame(){},console,send:raw=>sent.push(JSON.parse(raw))};
  vm.createContext(scope);
  vm.runInContext(fs.readFileSync(require.resolve('../counterdream/static/app.js'),'utf8'),scope);
  vm.runInContext('ws={readyState:1,bufferedAmount:0,send}; playing=true; streaming=true; streamReady=true;',scope);
  const key=(type,code,repeat=false)=>events[type]({code,repeat,target:element('viewport'),preventDefault(){}});
  return {scope,sent,key,events,element};
}
test('real keyboard handlers send press and release immediately without waiting for a heartbeat',()=>{
  const a=app();a.key('keydown','KeyW');
  assert.equal(a.sent.length,1);assert.deepEqual(a.sent[0].keys,['w']);
  a.key('keydown','KeyW',true);assert.equal(a.sent.length,1);
  a.key('keyup','KeyW');assert.equal(a.sent.length,2);assert.deepEqual(a.sent[1].keys,[]);
  assert.equal(a.sent[1].input_id,2);
});
test('changing fire while looking does not add extra mouse rotation impulses',()=>{
  const a=app();a.key('keydown','ArrowRight');a.key('keydown','KeyF');a.key('keyup','KeyF');
  assert.deepEqual(a.sent.map(x=>x.look_x),[10,10,10]);
  assert.deepEqual(a.sent.map(x=>x.dx),[0,0,0]);
  assert.deepEqual(a.sent.map(x=>x.fire),[false,true,false]);
});
test('losing focus pauses the server and clears held keyboard state',()=>{
  const a=app();a.key('keydown','KeyW');a.key('keydown','KeyF');a.events.blur();
  assert.equal(a.sent.at(-1).type,'pause');assert.equal(a.element('input-local').textContent,'PAUSED');
  vm.runInContext('resume()',a.scope);
  assert.deepEqual(a.sent.at(-1).keys,[]);assert.equal(a.sent.at(-1).fire,false);
});
