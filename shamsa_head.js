/*!
 * Shamsa — أفتار منصة «حرف مصر»
 * وردة نجمية مرسومة على Canvas: بتتنفس، وبتسمع، وبتفكر، وبتنبض مع الكلام.
 *
 * الاستخدام:
 *   const s = new Shamsa(canvasElement, { skin: 'copper' });
 *   s.setState('idle' | 'listen' | 'think' | 'talk');
 *   s.setSkin('copper' | 'khayamia' | 'sadaf' | 'clay' | 'faience');
 *   s.playVisemes([{ t: 0, id: 2 }, { t: 80, id: 19 }], startPerfMs); // t بالمللي ثانية من بداية الصوت
 *   s.playTimeline(seq, startPerfMs);   // [{t, shape}] من textToTimeline
 *   s.attachAnalyser(analyserNode);     // مزامنة بعلو الصوت (الخطة البديلة)
 *   s.stopSpeaking();
 *   s.destroy();
 */
(function (global) {
  'use strict';

  var SKINS = {
    copper:   { rim: ['#F0C893', '#CF8A52', '#7E4A22'], band: '#B8743E', bandInk: '#5A2F12', petal: '#E8B57F', petal2: '#B8743E', star: '#8E5428', star2: '#F3D3A8', core: '#FFE7BF', glow: '#EDB98A', motif: 'dots',   line: '#5A2F12' },
    khayamia: { rim: ['#2F4AA0', '#1E3A8A', '#14266A'], band: '#C0392B', bandInk: '#F3EADB', petal: '#E3B23C', petal2: '#C0392B', star: '#1E3A8A', star2: '#F3EADB', core: '#FFF4D6', glow: '#E3B23C', motif: 'stitch', line: '#F3EADB' },
    sadaf:    { rim: ['#6B4A2E', '#4A2F1B', '#2E1D10'], band: '#3A2414', bandInk: '#E9E4F2', petal: '#E9E4F2', petal2: '#B9C2D9', star: '#2E1D10', star2: '#F4F0FA', core: '#FFFFFF', glow: '#C9D6F2', motif: 'pearl',  line: '#E9E4F2' },
    clay:     { rim: ['#D9A574', '#B5653A', '#7A3E20'], band: '#C26E3C', bandInk: '#F6E6D0', petal: '#E8C29A', petal2: '#B5653A', star: '#7A3E20', star2: '#F6E6D0', core: '#FFE9CF', glow: '#E3955A', motif: 'brush',  line: '#F6E6D0' },
    faience:  { rim: ['#7FD6CB', '#36B3A2', '#1A6E66'], band: '#2A9486', bandInk: '#0E2F5A', petal: '#A8E6DD', petal2: '#36B3A2', star: '#0E2F5A', star2: '#D8F4EF', core: '#F0FFFC', glow: '#7FD6CB', motif: 'stitch', line: '#0E2F5A' }
  };

  /* عيلة الحرفة في المنصة → خامة الشمسة */
  var FAMILY_SKIN = { maaden: 'copper', fadda: 'sadaf', khashab: 'sadaf', tatriz: 'khayamia', nasig: 'khayamia', fukhar: 'clay', khous: 'clay', hagar: 'clay', zogag: 'faience', bahr: 'faience' };

  /* درجة الفتحة لكل شكل */
  var OPEN = { rest: 0, M: 0.06, F: 0.22, S: 0.3, U: 0.38, L: 0.45, E: 0.55, O: 0.72, A: 1 };

  /* Azure visemeId (0..21) → شكل */
  var AZURE_MAP = { 0: 'rest', 1: 'A', 2: 'A', 9: 'A', 11: 'A', 4: 'E', 5: 'E', 6: 'E', 3: 'O', 8: 'O', 10: 'O', 7: 'U',
    12: 'S', 13: 'S', 15: 'S', 16: 'S', 17: 'S', 19: 'S', 20: 'S', 14: 'L', 18: 'F', 21: 'M' };

  function hexA(h, a) {
    var n = parseInt(h.slice(1), 16);
    return 'rgba(' + (n >> 16) + ',' + ((n >> 8) & 255) + ',' + (n & 255) + ',' + Math.max(0, Math.min(1, a)).toFixed(3) + ')';
  }

  /* نص عربي → جدول أشكال تقريبي (للصوت اللي ملوش visemes) */
  function letterShape(ch) {
    if ('اآأإىةه'.indexOf(ch) >= 0) return 'A';
    if (ch === 'ي') return 'E';
    if (ch === 'و') return 'U';
    if ('بم'.indexOf(ch) >= 0) return 'M';
    if (ch === 'ف') return 'F';
    if (ch === 'ل') return 'L';
    if (/[ء-ي]/.test(ch)) return 'S';
    return null;
  }
  function textToTimeline(text, msPerLetter) {
    var ms = msPerLetter || 82, out = [], t = 0, flip = 0;
    for (var i = 0; i < text.length; i++) {
      var ch = text[i];
      if (/\s/.test(ch)) { out.push({ t: t, shape: 'rest', i: i }); t += ms * 0.8; continue; }
      if (/[.،,!؟?:]/.test(ch)) { out.push({ t: t, shape: 'rest', i: i }); t += ms * 4; continue; }
      var s = letterShape(ch); if (!s) continue;
      out.push({ t: t, shape: s, i: i }); t += ms;
      if ('SLMF'.indexOf(s) >= 0) {
        var nx = text[i + 1] || ' ';
        if ('اوي'.indexOf(nx) < 0 && /[ء-ي]/.test(nx)) { out.push({ t: t, shape: flip++ % 2 ? 'E' : 'A', i: i }); t += ms * 0.7; }
      }
    }
    out.push({ t: t, shape: 'rest', i: text.length });
    return { seq: out, duration: t };
  }

  function Shamsa(canvas, opts) {
    opts = opts || {};
    this.cv = canvas;
    this.ctx = canvas.getContext('2d');
    this.skin = SKINS[opts.skin] ? opts.skin : 'copper';
    this.state = 'idle';
    this.detail = opts.detail !== false;           // false = نسخة مبسطة للزرار الصغير
    this.reduced = !!(global.matchMedia && global.matchMedia('(prefers-reduced-motion: reduce)').matches);
    this.open = 0; this.target = 0; this.shape = 'rest';
    this.rot = 0; this.rot2 = 0; this.ripT = 0; this.morph = 0;
    this.look = { x: 0, y: 0 };
    this.seq = null; this.seqStart = 0; this.seqK = 0;
    this.analyser = null; this.buf = new Float32Array(1024);
    this.onShape = opts.onShape || null;            // callback(shape, charIndex)
    this.last = performance.now();
    this._raf = 0; this._visible = true;
    this._resize = this.resize.bind(this);
    this._loop = this.loop.bind(this);
    this.resize();
    global.addEventListener('resize', this._resize);
    var self = this;
    if ('IntersectionObserver' in global) {
      this._io = new IntersectionObserver(function (e) { self._visible = e[0].isIntersecting; });
      this._io.observe(canvas);
    }
    this._raf = requestAnimationFrame(this._loop);
  }

  Shamsa.SKINS = SKINS;
  Shamsa.FAMILY_SKIN = FAMILY_SKIN;
  Shamsa.AZURE_MAP = AZURE_MAP;
  Shamsa.OPEN = OPEN;
  Shamsa.textToTimeline = textToTimeline;

  Shamsa.prototype.resize = function () {
    var r = this.cv.getBoundingClientRect(), dpr = Math.min(2, global.devicePixelRatio || 1);
    var px = Math.max(40, Math.round((r.width || 120) * dpr));
    if (this.cv.width !== px) { this.cv.width = px; this.cv.height = px; }
  };
  Shamsa.prototype.setSkin = function (k) { if (SKINS[k] && k !== this.skin) { this.skin = k; this.morph = 1; } };
  Shamsa.prototype.setSkinForFamily = function (fam) { this.setSkin(FAMILY_SKIN[fam] || 'copper'); };
  Shamsa.prototype.setState = function (s) { this.state = s; if (s !== 'talk') { this.seq = null; this.shape = 'rest'; } };
  Shamsa.prototype.setOpen = function (v) { this.target = v; };

  Shamsa.prototype.playVisemes = function (events, startPerfMs) {
    this.playTimeline(events.map(function (e) { return { t: e.t, shape: AZURE_MAP[e.id] || 'rest', i: e.i }; }), startPerfMs);
  };
  Shamsa.prototype.playTimeline = function (seq, startPerfMs) {
    this.seq = seq.slice().sort(function (a, b) { return a.t - b.t; });
    this.seqStart = startPerfMs == null ? performance.now() : startPerfMs;
    this.seqK = 0; this.state = 'talk';
  };
  /* لما الفيزيمات بتوصل واحدة واحدة أثناء التوليد (Azure):
     beginStream() → pushViseme() كذا مرة → startClock() أول ما الصوت يبدأ */
  Shamsa.prototype.beginStream = function () { this.seq = []; this.seqK = 0; this.seqStart = Infinity; this.state = 'talk'; };
  Shamsa.prototype.pushViseme = function (tMs, id) {
    if (!this.seq) { this.seq = []; this.seqK = 0; }
    this.seq.push({ t: tMs, shape: AZURE_MAP[id] || 'rest' });
  };
  Shamsa.prototype.startClock = function (perfMs) { this.seqStart = perfMs == null ? performance.now() : perfMs; this.seqK = 0; this.state = 'talk'; };
  Shamsa.prototype.attachAnalyser = function (an) { this.analyser = an; this.state = an ? 'talk' : 'idle'; };
  Shamsa.prototype.stopSpeaking = function () { this.seq = null; this.analyser = null; this.shape = 'rest'; this.target = 0; if (this.state === 'talk') this.state = 'idle'; };
  Shamsa.prototype.destroy = function () {
    cancelAnimationFrame(this._raf); global.removeEventListener('resize', this._resize);
    if (this._io) this._io.disconnect();
  };

  Shamsa.prototype.loop = function (now) {
    this._raf = requestAnimationFrame(this._loop);
    if (!this._visible || document.hidden) { this.last = now; return; }
    var dt = Math.min(0.05, (now - this.last) / 1000); this.last = now;

    /* الهدف من الجدول الزمني أو من علو الصوت */
    if (this.seq && this.seq.length) {
      var el = now - this.seqStart;
      while (this.seqK < this.seq.length - 1 && this.seq[this.seqK + 1].t <= el) this.seqK++;
      var it = this.seq[this.seqK];
      if (it && it.t <= el) {
        if (it.shape !== this.shape && this.onShape) this.onShape(it.shape, it.i);
        this.shape = it.shape; this.target = OPEN[it.shape];
      }
    } else if (this.analyser) {
      this.analyser.getFloatTimeDomainData(this.buf);
      var q = 0; for (var i = 0; i < this.buf.length; i++) q += this.buf[i] * this.buf[i];
      var rms = Math.sqrt(q / this.buf.length);
      this.shape = rms < 0.015 ? 'rest' : rms < 0.035 ? 'M' : rms < 0.06 ? 'S' : rms < 0.09 ? 'E' : rms < 0.13 ? 'O' : 'A';
      this.target = OPEN[this.shape];
    } else if (this.state !== 'talk') this.target = 0;
    this.open += (this.target - this.open) * 0.3;
    this.draw(now / 1000, dt);
  };

  Shamsa.prototype.draw = function (t, dt) {
    var x = this.ctx, W = this.cv.width, s = SKINS[this.skin], st = this.state, open = this.open, R = this.reduced, full = this.detail;
    var k = W / 900; /* كل الرسم بمقاس 900 وبيتصغّر */
    var spd = st === 'think' ? 0.9 : st === 'talk' ? 0.18 : 0.06;
    this.rot += dt * spd * (R ? 0.3 : 1);
    this.rot2 -= dt * (st === 'think' ? 1.6 : 0.04) * (R ? 0.3 : 1);
    this.morph = Math.max(0, this.morph - dt * 2);
    var breathe = R ? 0 : Math.sin(t * 1.5) * 0.012;

    x.setTransform(1, 0, 0, 1, 0, 0); x.clearRect(0, 0, W, W);
    x.setTransform(k, 0, 0, k, W / 2, W / 2);
    var sc = 1 + breathe + (st === 'talk' ? open * 0.03 : 0); x.scale(sc, sc);

    /* 1 الهالة */
    var glowA = (st === 'talk' ? 0.35 + open * 0.65 : st === 'listen' ? 0.45 : 0.3) + this.morph * 0.4;
    var g = x.createRadialGradient(0, 0, 60, 0, 0, 430);
    g.addColorStop(0, hexA(s.glow, glowA * 0.55)); g.addColorStop(0.6, hexA(s.glow, glowA * 0.12)); g.addColorStop(1, hexA(s.glow, 0));
    x.fillStyle = g; x.beginPath(); x.arc(0, 0, 430, 0, 7); x.fill();

    if (st === 'listen') {
      this.ripT += dt;
      for (var r0 = 0; r0 < 3; r0++) {
        var p = (this.ripT * 0.6 + r0 / 3) % 1;
        x.strokeStyle = hexA('#36B3A2', (1 - p) * 0.6); x.lineWidth = 3;
        x.beginPath(); x.arc(0, 0, 420 - p * 170, 0, 7); x.stroke();
      }
    }

    /* 2 الإطار بالبُنط */
    g = x.createRadialGradient(-90, -110, 40, 0, 0, 360);
    g.addColorStop(0, s.rim[0]); g.addColorStop(0.55, s.rim[1]); g.addColorStop(1, s.rim[2]);
    x.fillStyle = g; x.beginPath(); x.arc(0, 0, 340, 0, 7); x.fill();
    x.strokeStyle = hexA(s.line, 0.45); x.lineWidth = 3;
    x.beginPath(); x.arc(0, 0, 326, 0, 7); x.stroke(); x.beginPath(); x.arc(0, 0, 290, 0, 7); x.stroke();
    var dots = full ? 48 : 24;
    for (var i = 0; i < dots; i++) {
      var a = i / dots * Math.PI * 2 + this.rot * 0.5, px = Math.cos(a) * 308, py = Math.sin(a) * 308, wave = 0;
      if (st === 'talk') { var ph = ((a - t * 3) % (Math.PI * 2) + Math.PI * 2) % (Math.PI * 2); wave = Math.max(0, 1 - ph / 1.2) * open; }
      x.fillStyle = hexA(s.line, 0.55 + wave * 0.45); x.beginPath(); x.arc(px, py, (full ? 6 : 9) + wave * 4, 0, 7); x.fill();
    }

    /* 3 حلقة النقشة */
    x.save(); x.rotate(this.rot2);
    x.fillStyle = s.band; x.beginPath(); x.arc(0, 0, 286, 0, 7); x.arc(0, 0, 214, 0, 7, true); x.fill();
    var units = full ? 24 : 12;
    for (var u = 0; u < units; u++) {
      x.save(); x.rotate(u / units * Math.PI * 2); x.translate(0, -250);
      if (s.motif === 'pearl') {
        var pg = x.createLinearGradient(-16, -20, 16, 20); pg.addColorStop(0, '#FFFFFF'); pg.addColorStop(0.5, '#C9D6F2'); pg.addColorStop(1, '#F2D9F0');
        x.fillStyle = pg; x.beginPath(); x.moveTo(0, -26); x.lineTo(14, 0); x.lineTo(0, 26); x.lineTo(-14, 0); x.fill();
      } else if (s.motif === 'brush') {
        x.strokeStyle = s.bandInk; x.lineWidth = 4; x.lineCap = 'round';
        x.beginPath(); x.moveTo(-14, 10); x.quadraticCurveTo(0, -26, 14, 10); x.stroke();
        x.fillStyle = s.bandInk; x.beginPath(); x.arc(0, 16, 3, 0, 7); x.fill();
      } else {
        x.fillStyle = s.bandInk; x.globalAlpha = 0.85;
        x.beginPath(); x.moveTo(0, -24); x.lineTo(18, 0); x.lineTo(0, 24); x.lineTo(-18, 0); x.closePath(); x.fill();
        x.globalAlpha = 1; x.fillStyle = s.band;
        x.beginPath(); x.moveTo(0, -11); x.lineTo(8, 0); x.lineTo(0, 11); x.lineTo(-8, 0); x.fill();
      }
      x.restore();
    }
    x.restore();

    /* 4 البتلات */
    var push = open * 26 + (st === 'listen' ? 6 : 0);
    x.save(); x.rotate(this.rot);
    for (var b = 0; b < 16; b++) {
      x.save(); x.rotate(b / 16 * Math.PI * 2 + Math.PI / 16); x.translate(0, -(70 + push * 0.6));
      var len = 138 + push, wid = b % 2 ? 26 : 34;
      x.fillStyle = b % 2 ? s.petal2 : s.petal;
      x.beginPath(); x.moveTo(0, 0); x.quadraticCurveTo(wid, -len * 0.5, 0, -len); x.quadraticCurveTo(-wid, -len * 0.5, 0, 0); x.closePath(); x.fill();
      if (full) { x.strokeStyle = hexA(s.star, 0.35); x.lineWidth = 2; x.stroke(); }
      x.restore();
    }
    x.restore();

    /* 5 النجمة الثمانية */
    x.save(); x.rotate(-this.rot * 0.6);
    var ss = 1 + open * 0.16 + this.morph * 0.08; x.scale(ss, ss);
    star8(x, 150, 0.62); x.fillStyle = s.star; x.fill(); x.strokeStyle = s.star2; x.lineWidth = 5; x.stroke();
    if (full) { x.save(); x.rotate(Math.PI / 8); star8(x, 112, 0.62); x.strokeStyle = hexA(s.star2, 0.55); x.lineWidth = 2.5; x.stroke(); x.restore(); }
    x.restore();

    /* 6 القلب (العين) */
    var L = this.look;
    if (st === 'listen') { L.x += (-14 - L.x) * 0.08; L.y += (4 - L.y) * 0.08; }
    else if (st === 'think') { L.x += (Math.cos(t * 2.2) * 10 - L.x) * 0.15; L.y += (-10 + Math.sin(t * 2.2) * 4 - L.y) * 0.15; }
    else { L.x += (Math.sin(t * 0.5) * 5 - L.x) * 0.05; L.y += (Math.cos(t * 0.4) * 3 - L.y) * 0.05; }
    var cr = 52 + open * 14;
    g = x.createRadialGradient(L.x, L.y, 4, 0, 0, cr + 30);
    g.addColorStop(0, s.core); g.addColorStop(0.55, hexA(s.glow, 0.9)); g.addColorStop(1, hexA(s.glow, 0));
    x.fillStyle = g; x.beginPath(); x.arc(0, 0, cr + 30, 0, 7); x.fill();
    x.fillStyle = s.core; x.beginPath(); x.arc(L.x * 0.6, L.y * 0.6, 18 + open * 10, 0, 7); x.fill();

    if (st === 'think') for (var d = 0; d < 3; d++) {
      var aa = t * 2.4 + d * 2.1; x.fillStyle = hexA(s.core, 0.9);
      x.beginPath(); x.arc(Math.cos(aa) * 196, Math.sin(aa) * 196, 9, 0, 7); x.fill();
    }
  };

  function star8(x, r, inner) {
    x.beginPath();
    for (var i = 0; i < 16; i++) { var rr = i % 2 ? r * inner : r, a = i * Math.PI / 8 - Math.PI / 2; x.lineTo(Math.cos(a) * rr, Math.sin(a) * rr); }
    x.closePath();
  }

  global.Shamsa = Shamsa;
})(window);


(function(){
  var S=null, AC=null, cur=null, lastPayload=null;
  var LABEL={idle:'شمسة جاهزة ✨', think:'شمسة بتدوّر في بيانات التوثيق…', talk:'شمسة بتتكلم 🔊'};
  function label(st){ var el=document.getElementById('shamsa-status'); if(el) el.textContent=LABEL[st]||''; }
  function boot(){
    var cv=document.getElementById('shamsa-cv');
    if(!cv){ return; }
    if(S && S.cv===cv) return;
    if(S) S.destroy();
    S=new window.Shamsa(cv,{skin:'copper'}); label('idle');
    cv.onclick=function(){ if(lastPayload) window.shamsaSpeak(lastPayload, true); };
  }
  setInterval(boot, 500);
  function ctx(){ try{ AC=AC||new (window.AudioContext||window.webkitAudioContext)(); if(AC.state==='suspended') AC.resume(); }catch(e){} return AC; }
  function stop(){ if(cur){ try{cur.pause();}catch(e){} cur=null; } if(window.speechSynthesis) speechSynthesis.cancel(); if(S){ S.stopSpeaking(); S.setState('idle'); } label('idle'); }
  window.shamsaThink=function(){ ctx(); stop(); boot(); if(S) S.setState('think'); label('think'); };
  window.shamsaStop=stop;
  window.shamsaSpeak=function(p, replay){
    boot(); if(!S) return; if(!p){ S.setState('idle'); label('idle'); return; }
    var d; try{ d=typeof p==='string'?JSON.parse(p):p; }catch(e){ S.setState('idle'); label('idle'); return; }
    if(!d || !d.text){ S.setState('idle'); label('idle'); return; }
    lastPayload=d; stop(); if(d.skin) S.setSkin(d.skin);
    var voiceOn=d.voice!==false;
    if(voiceOn && d.audio){
      var a=new Audio('data:'+(d.mime||'audio/mpeg')+';base64,'+d.audio); cur=a;
      var c=ctx();
      try{ var src=c.createMediaElementSource(a), an=c.createAnalyser(); an.fftSize=2048; src.connect(an); an.connect(c.destination); S.attachAnalyser(an); }
      catch(e){ var tl=window.Shamsa.textToTimeline(d.text); S.playTimeline(tl.seq); }
      label('talk');
      a.onended=function(){ if(cur===a) stop(); };
      a.play().catch(function(){ var tl=window.Shamsa.textToTimeline(d.text); S.playTimeline(tl.seq); setTimeout(stop, tl.duration+300); });
    } else {
      // من غير صوت: شمسة بتتحرك على قد طول الكلام
      var tl=window.Shamsa.textToTimeline(d.text.slice(0,400)); S.playTimeline(tl.seq); label('talk');
      setTimeout(function(){ if(S && S.state==='talk') stop(); }, tl.duration+300);
    }
  };
})();
