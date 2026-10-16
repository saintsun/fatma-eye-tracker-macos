"""The control panel that opens in a browser.

Binds to 127.0.0.1 only. Shows the tracker's live state and runs exactly the
same actions as the keyboard shortcuts, so calibration, the pointer and the
tuning can all be driven without the preview window having focus.

The HTTP thread never touches the tracker. It reads a snapshot the runtime
loop publishes once per frame, and it queues actions that the loop drains
on its next frame. Everything that mutates the tracker therefore happens on
the one thread that owns OpenCV and MediaPipe.
"""

from __future__ import annotations

import json
import socket
import threading
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from .i18n import DEFAULT_LANGUAGE, LANGUAGES, TEXT

DEFAULT_PORT = 8766
#: Ports tried in order, so a second instance does not simply fail.
PORT_RANGE = tuple(range(DEFAULT_PORT, DEFAULT_PORT + 8))

PAGE_TEMPLATE = """<!doctype html><html lang="__LANG__"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Fatma: Eye Tracker</title><style>
@import url("https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@500&display=swap");
:root{
  --bg:#070910; --bg2:#0b0f1a;
  --card:rgba(18,23,36,.72); --line:rgba(120,160,255,.14);
  --fg:#e9eefb; --dim:#7e8aa6;
  --neon:#00e5ff; --neon2:#b14bff; --ok:#00ffa3; --warn:#ffcc45; --err:#ff4d6d;
  --glow:0 0 18px rgba(0,229,255,.35);
}
*{box-sizing:border-box}
html,body{height:100%}
body{margin:0;padding:22px 18px 40px;color:var(--fg);
  font:14px/1.55 Inter,-apple-system,BlinkMacSystemFont,sans-serif;
  background:
    radial-gradient(900px 500px at 12% -8%, rgba(0,229,255,.12), transparent 60%),
    radial-gradient(760px 460px at 92% 4%, rgba(177,75,255,.12), transparent 62%),
    linear-gradient(180deg,var(--bg) 0%,var(--bg2) 100%);
  background-attachment:fixed}
.nav,.chips,section[data-p]{max-width:920px;margin-left:auto;margin-right:auto}
button{min-width:0}
::-webkit-scrollbar{width:9px;height:9px}
::-webkit-scrollbar-thumb{background:rgba(120,160,255,.2);border-radius:9px}
::-webkit-scrollbar-thumb:hover{background:rgba(0,229,255,.35)}

.nav{display:flex;align-items:center;gap:14px;flex-wrap:wrap;
  background:var(--card);border:1px solid var(--line);border-radius:999px;
  padding:9px 10px 9px 16px;margin-bottom:12px;backdrop-filter:blur(14px);
  box-shadow:0 10px 30px rgba(0,0,0,.35)}
.brand{display:flex;align-items:center;gap:10px;flex:0 0 auto}
.brand .nm{font-weight:700;font-size:14px;letter-spacing:.13em;
  text-transform:uppercase;
  background:linear-gradient(92deg,var(--neon),var(--neon2));
  -webkit-background-clip:text;background-clip:text;color:transparent;
  filter:drop-shadow(0 0 10px rgba(0,229,255,.4))}
.brand .ver{font-size:10.5px;color:var(--dim);font-family:"JetBrains Mono",monospace;
  padding:2px 7px;border:1px solid var(--line);border-radius:999px}
.tabs{display:flex;gap:5px;flex:1;justify-content:center;flex-wrap:wrap}
.tabs button{flex:0 0 auto;padding:7px 15px;border-radius:999px;
  background:transparent;border:1px solid transparent;color:var(--dim);
  font-size:13px;font-weight:500;cursor:pointer;font-family:inherit}
.tabs button:hover{color:var(--fg);background:rgba(255,255,255,.05)}
.tabs button.on{background:#fff;color:#0a0d16;font-weight:650;
  box-shadow:0 0 20px rgba(255,255,255,.22)}
.navsel{flex:0 0 auto;background:rgba(255,255,255,.05);border:1px solid var(--line);border-radius:999px;color:var(--fg);padding:6px 9px;font-size:12px;font-family:inherit;cursor:pointer}
.navsel:focus{outline:none;border-color:rgba(0,229,255,.55)}
.navst{flex:0 0 auto;font-size:15px;line-height:1;min-width:24px;text-align:center}
/* A pill that wraps becomes a tall blob: below this width the brand and the
   state keep the top row and the tabs drop to a full-width row of their own,
   each stretched to an equal share. */
@media (max-width:620px){
  .nav{border-radius:18px;padding:11px 13px;gap:10px}
  .brand{flex:1 1 auto}
  .navst{order:2}
  .tabs{order:3;flex:1 1 100%;justify-content:stretch;gap:4px}
  .tabs button{flex:1 1 0;padding:8px 6px;text-align:center}
}

.chips{display:flex;gap:7px;flex-wrap:wrap;margin-bottom:16px}
.chip{background:rgba(255,255,255,.04);border:1px solid var(--line);
  border-radius:20px;padding:4px 11px;font-size:11px;color:var(--dim);
  font-family:"JetBrains Mono",monospace;letter-spacing:.02em;white-space:nowrap}
.chip b{color:var(--fg);font-weight:600}
.chip.ok{border-color:rgba(0,255,163,.4);color:var(--ok)}
.chip.warn{border-color:rgba(255,204,69,.45);color:var(--warn)}
.chip.err{border-color:rgba(255,77,109,.45);color:var(--err)}

.card{background:var(--card);border:1px solid var(--line);border-radius:15px;
  padding:16px 17px;margin-bottom:13px;backdrop-filter:blur(14px);
  box-shadow:0 10px 30px rgba(0,0,0,.3)}
.card h2{margin:0 0 4px;font-size:13px;font-weight:600;letter-spacing:.06em;
  text-transform:uppercase;color:var(--fg)}
.card .sub{margin:0 0 13px;font-size:12.5px;color:var(--dim);line-height:1.5}
.g2{display:grid;grid-template-columns:1fr 1fr;gap:13px}
@media (max-width:880px){.g2{grid-template-columns:1fr}}
.row{display:flex;gap:8px;flex-wrap:wrap}

button.act{background:rgba(255,255,255,.05);border:1px solid var(--line);
  border-radius:10px;padding:9px 13px;font-size:12.5px;font-family:inherit;
  font-weight:500;cursor:pointer;color:var(--fg)}
button.act:hover{border-color:rgba(0,229,255,.45);box-shadow:var(--glow)}
button.act.on{background:rgba(0,255,163,.14);border-color:var(--ok);color:var(--ok)}
button.act.danger:hover{border-color:var(--err);box-shadow:0 0 18px rgba(255,77,109,.35)}
button.act.big{padding:13px 18px;font-size:14px;font-weight:600}
button.act[disabled]{opacity:.4;cursor:not-allowed;box-shadow:none;
  border-color:var(--line)}

.kv{display:flex;justify-content:space-between;gap:12px;padding:7px 0;
  border-bottom:1px solid rgba(120,160,255,.08);font-size:12.5px}
.kv:last-child{border-bottom:none}
.kv span:first-child{color:var(--dim)}
.kv span:last-child{font-family:"JetBrains Mono",ui-monospace,monospace;
  font-size:12px}

.seg{display:flex;flex-wrap:wrap;gap:4px;background:rgba(0,0,0,.28);border:1px solid var(--line);
  border-radius:11px;padding:3px}
.seg button{flex:1 0 auto;min-width:40px;padding:7px 10px;border-radius:8px;background:transparent;
  border:none;color:var(--dim);font-size:12.5px;font-family:inherit;
  font-weight:500;cursor:pointer}
.seg button:hover{color:var(--fg)}
.seg button.on{background:rgba(0,229,255,.16);color:var(--neon)}
label.fld{display:block;font-size:11.5px;color:var(--dim);margin:12px 0 6px;
  letter-spacing:.04em;text-transform:uppercase}
label.fld:first-child{margin-top:0}
.rng{display:flex;align-items:center;gap:11px}
.rng input[type=range]{flex:1;-webkit-appearance:none;height:4px;border-radius:3px;
  background:rgba(120,160,255,.22);outline:none}
.rng input[type=range]::-webkit-slider-thumb{-webkit-appearance:none;width:14px;
  height:14px;border-radius:50%;background:#fff;cursor:pointer;
  box-shadow:0 0 10px rgba(0,229,255,.6)}
.rng .vv{font-family:"JetBrains Mono",monospace;color:var(--dim);font-size:12px;
  min-width:46px;text-align:right}

.bar{height:6px;border-radius:4px;background:rgba(120,160,255,.16);
  overflow:hidden;margin-top:9px}
.bar i{display:block;height:100%;border-radius:4px;
  background:linear-gradient(92deg,var(--neon),var(--neon2));transition:width .2s}
.big{font-family:"JetBrains Mono",monospace;font-size:26px;font-weight:500;
  line-height:1.1}
.big small{font-size:12px;color:var(--dim);font-family:Inter,sans-serif;
  font-weight:400}

.toasts{position:fixed;top:16px;right:18px;z-index:120;display:flex;
  flex-direction:column;gap:9px;align-items:flex-end;pointer-events:none}
.toast{position:relative;max-width:340px;padding:10px 14px 10px 32px;
  border-radius:13px;font-size:12.5px;line-height:1.45;color:var(--fg);
  background:rgba(13,19,33,.95);border:1px solid var(--line);
  box-shadow:0 14px 34px rgba(0,0,0,.55);backdrop-filter:blur(12px);opacity:1}
.toast::before{content:"";position:absolute;left:13px;top:50%;margin-top:-4px;
  width:8px;height:8px;border-radius:50%;background:var(--neon);
  box-shadow:0 0 10px var(--neon)}
.toast.ok{border-color:rgba(0,255,163,.5);color:#b4ffd9}
.toast.ok::before{background:var(--ok);box-shadow:0 0 12px var(--ok)}
.toast.err{border-color:rgba(255,77,109,.5);color:#ffb3c0}
.toast.err::before{background:var(--err);box-shadow:0 0 10px var(--err)}
@media (max-width:760px){.toasts{left:16px;right:18px;align-items:stretch}
  .toast{max-width:100%}}
.off{opacity:.45;pointer-events:none}
.hero{display:flex;align-items:center;gap:16px;flex-wrap:wrap}
.hero .state{flex:1;min-width:200px}
.hero .state b{display:block;font-size:17px;font-weight:650;margin-bottom:3px}
.hero .state span{font-size:12.5px;color:var(--dim)}
button.power{padding:15px 26px;font-size:15px;font-weight:650;border-radius:13px;
  background:rgba(0,255,163,.14);border:1px solid var(--ok);color:var(--ok);
  font-family:inherit;cursor:pointer;box-shadow:0 0 22px rgba(0,255,163,.18)}
button.power:hover{box-shadow:0 0 30px rgba(0,255,163,.35)}
button.power.stop{background:rgba(255,77,109,.13);border-color:var(--err);
  color:var(--err);box-shadow:0 0 22px rgba(255,77,109,.16)}
button.power.stop:hover{box-shadow:0 0 30px rgba(255,77,109,.35)}
button.power[disabled]{opacity:.5;cursor:wait;box-shadow:none}
.err{margin-bottom:13px;padding:13px 15px;border-radius:13px;font-size:13px;
  line-height:1.5;color:#ffb3c0;border:1px solid rgba(255,77,109,.5);
  background:linear-gradient(180deg,rgba(255,77,109,.12),var(--card))}
.err b{display:block;margin-bottom:4px;color:#ff8da0}
.idle .card.needs-tracking{opacity:.45}
</style></head><body>
<div class="toasts" id="toasts"></div>

<div class="nav">
  <div class="brand"><span class="nm">Fatma</span><span class="ver" id="ver">—</span></div>
  <div class="tabs">
    <button data-t="kontrol" class="on" data-i18n="p.tab.control"></button>
    <button data-t="hassasiyet" data-i18n="p.tab.precision"></button>
    <button data-t="ayarlar" data-i18n="p.tab.settings"></button>
  </div>
  <select class="navsel" id="lang"><option value="en">EN</option><option value="tr">TR</option></select>
  <div class="navst" id="navst">⚪️</div>
</div>

<div class="chips" id="chips"></div>

<section data-p="kontrol">
  <div class="err" id="errbox" hidden></div>

  <div class="card">
    <div class="hero">
      <div class="state">
        <b id="statetitle">—</b>
        <span id="statesub"></span>
      </div>
      <button class="power" id="bpower" data-i18n="p.power.start"></button>
    </div>
  </div>

  <div class="card needs-tracking">
    <h2 data-i18n="p.pointer.title"></h2>
    <p class="sub" id="mousesub" data-i18n="p.pointer.sub_free"></p>
    <div class="row">
      <button class="act big" id="bmouse" data-i18n="p.pointer.on"></button>
      <button class="act" id="banchor" data-i18n="p.pointer.anchor"></button>
      <button class="act" id="bdrift" data-i18n="p.pointer.reset"></button>
    </div>
  </div>

  <div class="card needs-tracking">
    <h2 data-i18n="p.calib.title"></h2>
    <p class="sub" id="calibsub">—</p>
    <div class="row">
      <button class="act big" id="bcalib" data-i18n="p.calib.start"></button>
      <button class="act" id="bconfirm" disabled data-i18n="p.calib.confirm"></button>
      <button class="act danger" id="bcancel" disabled data-i18n="p.calib.cancel"></button>
    </div>
    <div class="bar"><i id="calibbar" style="width:0%"></i></div>
  </div>

  <div class="card needs-tracking">
    <h2 data-i18n="p.status.title"></h2>
    <div id="status"></div>
  </div>
</section>

<section data-p="hassasiyet" hidden>
  <div class="g2">
    <div class="card">
      <h2 data-i18n="p.acc.title"></h2>
      <p class="sub" data-i18n="p.acc.sub"></p>
      <div class="big" id="qmean">—</div>
      <div id="qdetail"></div>
    </div>
    <div class="card">
      <h2 data-i18n="p.jitter.title"></h2>
      <p class="sub" data-i18n="p.jitter.sub"></p>
      <div class="big" id="jrms">—</div>
      <div id="jdetail"></div>
    </div>
  </div>
  <div class="card">
    <h2 data-i18n="p.span.title"></h2>
    <p class="sub" data-i18n="p.span.sub"></p>
    <div class="big" id="spanv">—</div>
    <div id="spandetail"></div>
  </div>

  <div class="card">
    <h2 data-i18n="p.advice.title"></h2>
    <p class="sub" data-i18n="p.advice.sub"></p>
    <div id="advice"></div>
  </div>
</section>

<section data-p="ayarlar" hidden>
  <div class="g2">
    <div class="card">
      <h2 data-i18n="p.set.tracking"></h2>
      <label class="fld" data-i18n="p.set.which_eye"></label>
      <div class="seg" id="segeye"></div>
      <p class="sub" style="margin:7px 0 0" data-i18n="p.set.eye_sub"></p>
      <label class="fld" data-i18n="p.set.reference"></label>
      <div class="seg" id="segref"></div>
      <p class="sub" style="margin:7px 0 0" data-i18n="p.set.ref_sub"></p>
    </div>
    <div class="card">
      <h2 data-i18n="p.set.cursor"></h2>
      <label class="fld" data-i18n="p.set.smoothing"></label>
      <div class="seg" id="segsm"></div>
      <label class="fld" data-i18n="p.set.deadzone"></label>
      <div class="rng">
        <input type="range" id="rdead" min="0" max="40" step="1">
        <span class="vv" id="vdead">—</span>
      </div>
      <label class="fld" data-i18n="p.set.gestures"></label>
      <div class="row">
        <button class="act" id="bgest">—</button>
      </div>
      <p class="sub" style="margin:9px 0 0" data-i18n="p.set.gest_sub"></p>
    </div>
  </div>
  <div class="card">
    <h2 data-i18n="p.set.nextcal"></h2>
    <label class="fld" data-i18n="p.set.targets"></label>
    <div class="seg" id="seggrid"></div>
    <p class="sub" style="margin:9px 0 0" data-i18n="p.set.grid_sub"></p>
  </div>
  <div class="card">
    <h2 data-i18n="p.set.hotkey"></h2>
    <p class="sub" data-i18n="p.set.hotkey_sub"></p>
    <div class="row"><button class="act" id="bhk">—</button></div>
    <label class="fld" data-i18n="p.set.hotkey_key"></label>
    <div class="seg" id="seghk"></div>
    <div id="hkdetail"></div>
  </div>

  <div class="card">
    <h2 data-i18n="p.set.app"></h2>
    <div class="row"><button class="act danger" id="bquit" data-i18n="p.set.quit"></button></div>
  </div>
</section>

<script>
const STR=__STRINGS__;
let LANG='__LANG__';
const T=(k,f)=>{let v=(STR[LANG]&&STR[LANG][k])||STR.en[k]||k;
  if(f) for(const n in f) v=v.replaceAll('{'+n+'}',f[n]);
  return v.replace(/[{][^}]*[}]/g,'');};
function applyStatic(){document.querySelectorAll('[data-i18n]')
  .forEach(e=>e.textContent=T(e.dataset.i18n));}
const $=s=>document.querySelector(s), $$=s=>[...document.querySelectorAll(s)];
let S={}, busy=false;

function toast(msg,kind){
  const t=document.createElement('div');
  t.className='toast '+(kind||'');t.textContent=msg;
  $('#toasts').appendChild(t);
  setTimeout(()=>t.remove(),2600);
}
async function act(action,value){
  if(busy) return; busy=true;
  try{
    const r=await fetch('/api/action',{method:'POST',
      headers:{'Content-Type':'application/json'},
      body:JSON.stringify({action,value})});
    const j=await r.json();
    if(j.ok){ if(j.message) toast(j.message,'ok'); }
    else toast(j.error||T('p.toast.failed'),'err');
    await poll();
  }catch(e){ toast(T('p.toast.offline'),'err'); }
  finally{ busy=false; }
}
$$('.tabs button').forEach(b=>b.onclick=()=>{
  $$('.tabs button').forEach(x=>x.classList.toggle('on',x===b));
  $$('section[data-p]').forEach(s=>s.hidden = s.dataset.p!==b.dataset.t);
});

function seg(host,items,current,action){
  const el=$(host); if(!el) return;
  if(el.dataset.sig === items.map(i=>i[0]).join()+'|'+current) return;
  el.dataset.sig = items.map(i=>i[0]).join()+'|'+current;
  el.innerHTML='';
  items.forEach(([val,label])=>{
    const b=document.createElement('button');
    b.textContent=label;
    b.className = String(val)===String(current) ? 'on' : '';
    b.onclick=()=>act(action,val);
    el.appendChild(b);
  });
}
function kv(host,rows){
  $(host).innerHTML = rows.map(([k,v])=>
    `<div class="kv"><span>${k}</span><span>${v}</span></div>`).join('');
}

function render(){
  $('#ver').textContent='v'+(S.version||'?');
  const tr=!!S.tracking;
  $('#navst').textContent = !S.running ? '⚪️' : !tr ? '⏸'
    : S.calibrating ? '🎯' : S.mouse_on ? '🔴' : '😴';
  document.body.classList.toggle('idle', !tr);

  const eb=$('#errbox');
  if(S.error){ eb.hidden=false;
    eb.innerHTML='<b>'+T('p.error.title')+'</b>'+String(S.error)
      .replace(/[<>&]/g,c=>({'<':'&lt;','>':'&gt;','&':'&amp;'}[c])); }
  else eb.hidden=true;

  const bp=$('#bpower');
  bp.textContent = T(tr?'p.power.stop':'p.power.start');
  bp.classList.toggle('stop', tr);
  $('#statetitle').textContent = T(tr
    ? (S.calibrating?'p.state.calibrating':'p.state.on') : 'p.state.off');
  $('#statesub').textContent = T(tr
    ? (S.face?'p.state.face':'p.state.noface') : 'p.state.off_sub');

  const cam=(S.camera||[0,0]).join('×'), scr=(S.screen||[0,0]).join('×');
  const lowres = tr && (S.camera||[0])[0] < 1280;
  $('#chips').innerHTML=[
    `<span class="chip"><b>${tr?(S.fps||0).toFixed(0):'—'}</b> fps</span>`,
    `<span class="chip ${lowres?'warn':''}">${T('p.chip.camera')} <b>${tr?cam:'—'}</b></span>`,
    `<span class="chip">${T('p.chip.screen')} <b>${scr}</b></span>`,
    `<span class="chip">${T('p.chip.pointer')} <b>${S.pointer||'?'}</b></span>`,
    `<span class="chip ${S.face?'ok':(tr?'warn':'')}">${T(tr?(S.face?'p.chip.face_yes':'p.chip.face_no'):'p.chip.camera_off')}</span>`,
    `<span class="chip ${S.calibrated?'ok':'err'}">${T(S.calibrated?'p.chip.calibrated':'p.chip.uncalibrated')}</span>`,
  ].join('');

  const bm=$('#bmouse');
  bm.textContent = T(S.mouse_on?'p.pointer.off':'p.pointer.on');
  bm.classList.toggle('on', !!S.mouse_on);
  bm.disabled = !tr || (!S.calibrated && !S.mouse_on);
  $('#mousesub').textContent = T(!S.calibrated ? 'p.pointer.sub_needcal'
    : S.mouse_on ? 'p.pointer.sub_tracking' : 'p.pointer.sub_free');
  $('#banchor').disabled = !tr || !S.calibrated;
  $('#bdrift').disabled = !tr || !S.calibrated;

  const c=S.calibrating;
  $('#bcalib').disabled=c||!tr; $('#bconfirm').disabled=!c; $('#bcancel').disabled=!c;
  $('#calibbar').style.width = c ? (100*S.calib_step/S.calib_total)+'%' : '0%';
  $('#calibsub').textContent = c
    ? T('p.calib.running',{step:S.calib_step+1,total:S.calib_total})
    : T(S.calibrated?'p.calib.done':'p.calib.none');

  const onoff=v=>T(v?'hud.on':'hud.off').toLowerCase();
  kv('#status',[
    [T('p.status.pointer'), onoff(S.mouse_on)],
    [T('p.status.drift'), `dx=${(S.drift||[0,0])[0].toFixed(0)} dy=${(S.drift||[0,0])[1].toFixed(0)}`],
    [T('p.status.eye'), `${S.eye||'?'} / ${S.reference||'?'}`],
    [T('p.status.smoothing'), `${(S.smoothing||{}).index+1}/${(S.smoothing||{}).total} (cutoff ${(S.smoothing||{}).cutoff})`],
    [T('p.status.deadzone'), (S.deadzone||0)+' px'],
    [T('p.status.gestures'), onoff(S.gestures)],
  ]);

  const q=S.quality;
  $('#qmean').innerHTML = q
    ? `${q.mean_px.toFixed(0)} <small>${T('p.acc.mean')} · ${q.grade}</small>`
    : `— <small>${T('p.acc.none')}</small>`;
  kv('#qdetail', q
    ? [[T('p.acc.worst'), `${q.worst_px.toFixed(0)} px (#${q.worst_index+1})`],
       [T('p.acc.points'), q.points]]
    : [[T('p.span.state'), T('p.acc.pending')]]);

  const j=S.jitter||{};
  $('#jrms').innerHTML = j.rms!=null
    ? `${j.rms.toFixed(0)} <small>px${j.fixating?' · '+T('p.jitter.holding'):''}</small>`
    : '—';
  kv('#jdetail',[[T('p.jitter.best'), j.best!=null?j.best.toFixed(0)+' px':'—']]);

  const sp=S.span;
  $('#spanv').innerHTML = sp
    ? `${sp.x.toFixed(3)} / ${sp.y.toFixed(3)} <small>${T('p.span.axes')}${sp.ok?'':' · '+T('p.span.narrow')}</small>`
    : `— <small>${T('p.acc.none')}</small>`;
  kv('#spandetail', sp
    ? (sp.ok ? [[T('p.span.state'),T('p.span.ok')]]
             : [[T('p.span.state'),T('p.span.bad')],[T('p.span.what'),sp.advice]])
    : [[T('p.span.state'), T('p.acc.pending')]]);

  const tips=[];
  if(lowres) tips.push([T('p.advice.lowres'),T('p.advice.lowres_sub')]);
  if(sp && !sp.ok) tips.push([T('p.advice.span'),T('p.advice.span_sub')]);
  if(q && q.mean_px>120) tips.push([T('p.advice.calib'),T('p.advice.calib_sub')]);
  if(j.best!=null && j.best>60) tips.push([T('p.advice.jitter'),T('p.advice.jitter_sub')]);
  if(S.eye && S.eye!=='auto') tips.push([T('p.advice.oneeye'),T('p.advice.oneeye_sub')]);
  kv('#advice', tips.length?tips:[[T('p.advice.ok'),T('p.advice.ok_sub')]]);

  seg('#segeye',[['auto',T('p.set.eye_auto')],['left',T('p.set.eye_left')],
                 ['right',T('p.set.eye_right')],['both',T('p.set.eye_both')]],S.eye,'eye');
  seg('#segref',[['anchors','anchors (6)'],['corners','corners (2)']],S.reference,'reference');
  const sm=S.smoothing||{index:0,total:4};
  seg('#segsm',[[0,T('p.set.sm0')],[1,T('p.set.sm1')],[2,T('p.set.sm2')],
                [3,T('p.set.sm3')]],sm.index,'smoothing');
  seg('#seggrid',[[9,T('p.set.t9')],[13,T('p.set.t13')]],S.grid,'grid');

  const hk=S.hotkey||{};
  const bh=$('#bhk');
  bh.textContent=T(hk.enabled?'p.set.hotkey_on':'p.set.hotkey_off');
  bh.classList.toggle('on', !!hk.active);
  const clean=new Set(hk.clean||[]);
  seg('#seghk',(hk.keys||[]).map(k=>[k,
        (k==='period'?'.':k.toUpperCase())+(clean.has(k)?'':' ·')]),
      hk.key,'hotkey');
  const hkrows=[];
  if(hk.description) hkrows.push([T('p.set.hotkey'), hk.description]);
  if(hk.enabled && !hk.active) hkrows.push([T('p.error.title'), T('p.set.hotkey_failed')]);
  hkrows.push([T('p.set.hotkey_fired'), hk.fired_ago!=null
    ? T('p.set.hotkey_ago',{n:Math.round(hk.fired_ago)})
    : T('p.set.hotkey_never')]);
  const clash=(hk.conflicts||{})[hk.key];
  hkrows.push([clash?'⚠':'✓', clash || T('p.set.hotkey_clean')]);
  kv('#hkdetail', hkrows);

  const rd=$('#rdead');
  if(document.activeElement!==rd) rd.value=S.deadzone||0;
  $('#vdead').textContent=(S.deadzone||0)+' px';
  const bg=$('#bgest');
  bg.textContent=T(S.gestures?'p.set.gest_on':'p.set.gest_off');
  bg.classList.toggle('on',!!S.gestures);
  document.body.classList.toggle('off', !S.running);
}

$('#bpower').onclick=()=>{ $('#bpower').disabled=true; act(S.tracking?'stop':'start'); };
$('#bmouse').onclick=()=>act('mouse',!S.mouse_on);
$('#banchor').onclick=()=>act('anchor');
$('#bdrift').onclick=()=>act('reset_drift');
$('#bcalib').onclick=()=>act('calibrate');
$('#bconfirm').onclick=()=>act('confirm');
$('#bcancel').onclick=()=>act('cancel');
$('#bgest').onclick=()=>act('gestures',!S.gestures);
$('#bhk').onclick=()=>act('hotkey_enabled',!(S.hotkey||{}).enabled);
$('#bquit').onclick=()=>{ if(confirm(T('p.set.quit_confirm'))) act('quit'); };
$('#rdead').oninput=e=>$('#vdead').textContent=e.target.value+' px';
$('#rdead').onchange=e=>act('deadzone',parseFloat(e.target.value));

async function poll(){
  try{
    const r=await fetch('/api/state');
    S=await r.json(); $('#bpower').disabled=false;
    if(S.lang && S.lang!==LANG){LANG=S.lang;$('#lang').value=LANG;
      document.documentElement.lang=LANG;applyStatic();}
    render();
  }catch(e){ $('#navst').textContent='⚪️'; }
}
$('#lang').onchange=e=>{LANG=e.target.value;
  document.documentElement.lang=LANG;applyStatic();render();
  act('language',LANG);};
applyStatic();
poll(); setInterval(poll,700);
</script></body></html>"""


def render_page(lang: str = DEFAULT_LANGUAGE) -> str:
    """Fill the page template with the string table and the initial language.

    The whole table goes to the browser, so switching language is instant
    and needs no round trip — and there is exactly one place the wording
    lives, shared with the overlay drawn on the camera frames.
    """
    if lang not in LANGUAGES:
        lang = DEFAULT_LANGUAGE
    # The lang attribute is not decoration: CSS text-transform follows the
    # document language, and Turkish uppercases "i" to "I with a dot". An
    # English page left marked Turkish renders "POINTER" as "POINTER" with
    # a dotted I.
    return (PAGE_TEMPLATE
            .replace("__STRINGS__", json.dumps(TEXT, ensure_ascii=False))
            .replace("__LANG__", lang))


#: Kept for tests and for anything that just wants the default rendering.
PAGE = render_page()


class PanelBridge:
    """Shared state between the runtime loop and the HTTP threads.

    The loop publishes a snapshot and drains actions; the HTTP handlers read
    the snapshot and queue actions. Nothing else crosses the boundary, so
    the tracker keeps running on a single thread.
    """

    #: Actions the panel may request, mapped to the type of their value.
    ACTIONS = {
        "start": None,
        "stop": None,
        "mouse": bool,
        "calibrate": None,
        "confirm": None,
        "cancel": None,
        "anchor": None,
        "reset_drift": None,
        "smoothing": int,
        "eye": str,
        "reference": str,
        "grid": int,
        "deadzone": float,
        "gestures": bool,
        "language": str,
        "hotkey": str,
        "hotkey_enabled": bool,
        "quit": None,
    }

    def __init__(self, max_queue: int = 32) -> None:
        self._lock = threading.Lock()
        self._state: dict = {"running": False}
        self._queue: deque[tuple[str, object]] = deque(maxlen=max_queue)

    # --- runtime loop side ---------------------------------------------

    def publish(self, state: dict) -> None:
        """Replace the snapshot the panel shows. Called once per frame."""
        with self._lock:
            self._state = state

    def drain(self) -> list[tuple[str, object]]:
        """Take every queued action. Called once per frame."""
        with self._lock:
            actions = list(self._queue)
            self._queue.clear()
        return actions

    # --- HTTP side -----------------------------------------------------

    def snapshot(self) -> dict:
        with self._lock:
            return dict(self._state)

    def submit(self, action: str, value=None) -> tuple[bool, str]:
        """Queue an action. Returns (accepted, message)."""
        if action not in self.ACTIONS:
            return False, f"bilinmeyen eylem: {action}"
        kind = self.ACTIONS[action]
        if kind is not None:
            if value is None:
                return False, f"{action} bir değer bekliyor"
            try:
                value = kind(value)
            except (TypeError, ValueError):
                return False, f"{action} için geçersiz değer"
        with self._lock:
            self._queue.append((action, value))
        return True, ""


def _free_port(candidates=PORT_RANGE) -> int | None:
    for port in candidates:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                sock.bind(("127.0.0.1", port))
            except OSError:
                continue
            return port
    return None


def make_handler(bridge: PanelBridge):
    class _Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *args):
            # The tracker's stdout is the user's status display; a request
            # log line per poll would bury it.
            pass

        def _send(self, body: bytes, content_type: str, status: int = 200):
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            # The panel is local-only and holds no secrets, but there is no
            # reason for another page to be able to read or drive it.
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def _json(self, payload: dict, status: int = 200):
            self._send(json.dumps(payload).encode("utf-8"),
                       "application/json; charset=utf-8", status)

        def do_GET(self):  # noqa: N802
            if self.path.startswith("/api/state"):
                self._json(bridge.snapshot())
                return
            if self.path in ("/", "/index.html"):
                lang = bridge.snapshot().get("lang", DEFAULT_LANGUAGE)
                self._send(render_page(lang).encode("utf-8"),
                           "text/html; charset=utf-8")
                return
            self._json({"error": "not found"}, 404)

        def do_POST(self):  # noqa: N802
            if not self.path.startswith("/api/action"):
                self._json({"error": "not found"}, 404)
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                payload = json.loads(self.rfile.read(length) or b"{}")
            except (ValueError, TypeError):
                self._json({"ok": False, "error": "bozuk istek"}, 400)
                return
            if not isinstance(payload, dict):
                self._json({"ok": False, "error": "bozuk istek"}, 400)
                return
            ok, message = bridge.submit(payload.get("action"),
                                        payload.get("value"))
            self._json({"ok": ok, "error": message} if not ok
                       else {"ok": True, "message": message})

    return _Handler


class Panel:
    """Lifecycle for the control panel's HTTP server."""

    def __init__(self, bridge: PanelBridge, port: int | None = None) -> None:
        self.bridge = bridge
        self.requested_port = port
        self.port: int | None = None
        self.server: ThreadingHTTPServer | None = None
        self._thread: threading.Thread | None = None

    @property
    def url(self) -> str | None:
        return f"http://127.0.0.1:{self.port}" if self.port else None

    def start(self) -> bool:
        """Bind and serve. Returns False if no port was free."""
        candidates = (self.requested_port,) if self.requested_port else PORT_RANGE
        port = _free_port(candidates)
        if port is None:
            return False
        try:
            self.server = ThreadingHTTPServer(("127.0.0.1", port),
                                              make_handler(self.bridge))
        except OSError:
            return False
        self.server.daemon_threads = True
        self.port = port
        self._thread = threading.Thread(
            target=self.server.serve_forever, daemon=True, name="fatma-panel"
        )
        self._thread.start()
        return True

    def stop(self) -> None:
        """Shut the server down. Safe to call when it never started."""
        if self.server is not None:
            self.server.shutdown()
            self.server.server_close()
            self.server = None
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None
        self.port = None
