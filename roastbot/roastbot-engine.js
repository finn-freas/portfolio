"use strict";

const P2 = [];
for (let i = 0; i <= 80; i++) P2.push(Math.pow(2, i));

function rshr(v, s) {
  if (s <= 0) return v * P2[-s];
  return Math.floor((v + P2[s - 1]) / P2[s]);
}
function sat16(v) { return v < -32768 ? -32768 : v > 32767 ? 32767 : v; }
function floorShr(v, s) { return Math.floor(v / P2[s]); }
function pyRound(x) {
  const f = Math.floor(x), d = x - f;
  if (d > 0.5) return f + 1;
  if (d < 0.5) return f;
  return f % 2 === 0 ? f : f + 1;
}
function bitLength(n) {
  let b = 0;
  while (n >= 1) { n = Math.floor(n / 2); b++; }
  return b;
}

function loadModel(meta, buf) {
  const view = (name) => {
    const e = meta.index[name];
    const T = { int8: Int8Array, uint8: Uint8Array, int16: Int16Array, int32: Int32Array }[e.dtype];
    return new T(buf, e.offset, e.nbytes / T.BYTES_PER_ELEMENT);
  };
  const unpack4 = (name) => {
    const e = meta.index[name];
    const n = e.shape.reduce((a, b) => a * b, 1);
    const src = new Uint8Array(buf, e.offset, e.nbytes);
    const out = new Int8Array(n);
    for (let i = 0; i < n; i++) {
      const b = src[i >> 1];
      const v = (i & 1) ? (b >> 4) : (b & 15);
      out[i] = v >= 8 ? v - 16 : v;
    }
    return out;
  };
  const c = meta.cfg;
  const layers = [];
  for (let l = 0; l < c.n_layers; l++) {
    const L = {};
    for (const k of ["wq", "wk", "wv", "wo", "w1", "w3", "w2"]) {
      const shape = meta.index[`L${l}.${k}.q4`].shape;
      L[k] = { rows: shape[0], cols: shape[1], q4: unpack4(`L${l}.${k}.q4`), g8: view(`L${l}.${k}.g8`),
               m: view(`L${l}.${k}.m`), s: view(`L${l}.${k}.s`) };
    }
    layers.push(L);
  }
  const p = { cfg: c, fmt: meta.fmt, eps: meta.eps, f_fin: meta.f_fin, kv8: meta.kv8, K: meta.consts, layers,
              log2_lut: meta.log2_lut, chat: meta.chat };
  for (const k of ["table", "emb_m", "cls_m", "gfin", "rsqrt", "exp2", "sigmoid", "gumbel", "emb_s", "cls_s",
                   "rope_cos", "rope_sin", "w_base", "emoji_bias", "emoji_mask", "common"]) p[k] = view(k);
  c.head_size = c.dim / c.n_heads;
  c.kv_dim = c.n_kv_heads * c.head_size;
  return p;
}

function matvec4(M, x, out) {
  const { rows, cols, q4, g8, m, s } = M;
  const ng = cols / 32;
  for (let r = 0; r < rows; r++) {
    let acc = 0;
    const base = r * cols;
    for (let g = 0; g < ng; g++) {
      let sub = 0;
      const o = base + g * 32, xo = g * 32;
      for (let j = 0; j < 32; j++) sub += q4[o + j] * x[xo + j];
      acc += g8[r * ng + g] * sub;
    }
    out[r] = sat16(floorShr(acc * m[r] + P2[s[r] - 1], s[r]));
  }
  return out;
}

class IntModel {
  constructor(p) {
    this.p = p;
    const c = p.cfg;
    this.kc = new Int32Array(c.n_layers * c.seq_len * c.kv_dim);
    this.vc = new Int32Array(c.n_layers * c.seq_len * c.kv_dim);
    this.x = new Float64Array(c.dim);
    this.u = new Float64Array(c.dim);
    this.q = new Float64Array(c.dim);
    this.k = new Float64Array(c.kv_dim);
    this.v = new Float64Array(c.kv_dim);
    this.att = new Float64Array(c.dim);
    this.o = new Float64Array(c.dim);
    this.h1 = new Float64Array(c.hidden);
    this.h3 = new Float64Array(c.hidden);
    this.hb = new Float64Array(c.hidden);
    this.logits = new Float64Array(c.vocab);
    this.tt = new Float64Array(c.seq_len);
    this.e = new Float64Array(c.seq_len);
  }
  reset() { this.kc.fill(0); this.vc.fill(0); }

  rmsnorm(x, out) {
    const p = this.p, K = p.K;
    let ss = p.eps;
    for (let i = 0; i < x.length; i++) ss += x[i] * x[i];
    let sh = Math.max(0, bitLength(ss) - K.RSQ_BITS);
    sh += sh & 1;
    const t = p.rsqrt[floorShr(ss, sh) - (1 << (K.RSQ_BITS - 2))];
    const s = K.RSQ_K - 15 + sh / 2;
    for (let i = 0; i < x.length; i++) out[i] = sat16(rshr(x[i] * t, s));
    return out;
  }

  rope(v, pos) {
    const p = this.p, n = p.cfg.head_size / 2, F = p.K.F_ROPE;
    const cos = p.rope_cos, sin = p.rope_sin, off = pos * n;
    for (let i = 0; i < v.length / 2; i++) {
      const c = cos[off + (i % n)], s = sin[off + (i % n)];
      const a = v[2 * i], b = v[2 * i + 1];
      v[2 * i] = sat16(rshr(a * c - b * s, F));
      v[2 * i + 1] = sat16(rshr(a * s + b * c, F));
    }
  }

  forward(tok, pos, base = 0) {
    const p = this.p, c = p.cfg, f = p.fmt, K = p.K;
    const hs = c.head_size, rep = c.n_heads / c.n_kv_heads, kvd = c.kv_dim, D = c.dim;
    const tShift = f.q + f.k - K.F_T, hShift = 2 * f.h - f.hb;
    const x = this.x, u = this.u, q = this.q, k = this.k, v = this.v, att = this.att, o = this.o;
    const em = p.emb_m[tok], es = p.emb_s[tok];
    for (let i = 0; i < D; i++) x[i] = sat16(rshr(p.table[tok * D + i] * em, es));
    for (let l = 0; l < c.n_layers; l++) {
      const L = p.layers[l];
      this.rmsnorm(x, u);
      matvec4(L.wq, u, q); this.rope(q, pos);
      matvec4(L.wk, u, k); this.rope(k, pos);
      matvec4(L.wv, u, v);
      let sk = 0, sv = 0;
      if (p.kv8) {
        sk = p.kv8[l][0]; sv = p.kv8[l][1];
        for (let i = 0; i < kvd; i++) {
          k[i] = Math.min(127, Math.max(-128, rshr(k[i], sk)));
          v[i] = Math.min(127, Math.max(-128, rshr(v[i], sv)));
        }
      }
      const lb = l * c.seq_len * kvd;
      for (let i = 0; i < kvd; i++) { this.kc[lb + pos * kvd + i] = k[i]; this.vc[lb + pos * kvd + i] = v[i]; }
      const shift = tShift - sk;
      for (let h = 0; h < c.n_heads; h++) {
        const kvOff = Math.floor(h / rep) * hs, qOff = h * hs;
        const tt = this.tt, e = this.e;
        let tmax = -Infinity;
        for (let j = base; j <= pos; j++) {
          let dot = 0;
          const ko = lb + j * kvd + kvOff;
          for (let i = 0; i < hs; i++) dot += this.kc[ko + i] * q[qOff + i];
          let t = rshr(dot, shift);
          if (t < -(1 << 17)) t = -(1 << 17); else if (t > (1 << 17) - 1) t = (1 << 17) - 1;
          tt[j] = t;
          if (t > tmax) tmax = t;
        }
        let total = 0;
        for (let j = base; j <= pos; j++) {
          const d = tmax - tt[j], ip = d >> 8, fp = d & 255;
          const ev = ip >= 16 ? 0 : (p.exp2[fp] >> Math.min(ip, 15));
          e[j] = ev; total += ev;
        }
        const half = Math.floor(total / 2);
        for (let i = 0; i < hs; i++) {
          let out = 0;
          for (let j = base; j <= pos; j++) if (e[j]) out += e[j] * this.vc[lb + j * kvd + kvOff + i];
          const qq = Math.floor((Math.abs(out) * P2[sv] + half) / total);
          att[qOff + i] = out < 0 ? -qq : qq;
        }
      }
      for (let i = 0; i < D; i++) att[i] = sat16(att[i]);
      matvec4(L.wo, att, o);
      for (let i = 0; i < D; i++) x[i] = sat16(x[i] + o[i]);
      this.rmsnorm(x, u);
      matvec4(L.w1, u, this.h1); matvec4(L.w3, u, this.h3);
      const SIG_N = K.SIG_N, SIG_STEP = K.SIG_STEP, F_SIG = K.F_SIG;
      for (let i = 0; i < c.hidden; i++) {
        const h1 = this.h1[i];
        let idx = floorShr(h1, SIG_STEP) + SIG_N / 2;
        idx = idx < 0 ? 0 : idx > SIG_N - 1 ? SIG_N - 1 : idx;
        const s = sat16(rshr(h1 * p.sigmoid[idx], F_SIG));
        this.hb[i] = sat16(rshr(s * this.h3[i], hShift));
      }
      matvec4(L.w2, this.hb, o);
      for (let i = 0; i < D; i++) x[i] = sat16(x[i] + o[i]);
    }
    this.rmsnorm(x, u);
    const ugs = K.F_U + p.f_fin - f.ug;
    for (let i = 0; i < D; i++) u[i] = sat16(rshr(u[i] * p.gfin[i], ugs));
    const lg = this.logits;
    for (let r = 0; r < c.vocab; r++) {
      let acc = 0;
      const b = r * D;
      for (let i = 0; i < D; i++) acc += p.table[b + i] * u[i];
      lg[r] = sat16(floorShr(acc * p.cls_m[r] + P2[p.cls_s[r] - 1], p.cls_s[r]));
    }
    return lg;
  }
}

class XorShift {
  constructor(seed) { this.x = (seed >>> 0) || 0x9E3779B9; }
  next() {
    let x = this.x;
    x = (x ^ (x << 13)) >>> 0;
    x = (x ^ (x >>> 17)) >>> 0;
    x = (x ^ (x << 5)) >>> 0;
    this.x = x;
    return x;
  }
}

function makeRng(seed) {
  let x = seed >>> 0;
  x = (x ^ (x >>> 16)) >>> 0;
  x = Math.imul(x, 0x85EBCA6B) >>> 0;
  x = (x ^ (x >>> 13)) >>> 0;
  x = Math.imul(x, 0xC2B2AE35) >>> 0;
  x = (x ^ (x >>> 16)) >>> 0;
  return new XorShift(x);
}

function classMass(p, lg, mask) {
  const K = p.K, f = p.fmt.logit, n = lg.length;
  let mx = -Infinity;
  for (let i = 0; i < n; i++) if (lg[i] > mx) mx = lg[i];
  let em = 0, total = 0;
  const lim = 15 + K.MASS_BITS;
  for (let i = 0; i < n; i++) {
    const d = floorShr((mx - lg[i]) * K.LOG2E_Q8, f);
    const ip = d >> 8, fp = d & 255;
    const e = ip > lim ? 0 : floorShr(p.exp2[fp] * P2[K.MASS_BITS], Math.min(ip, lim));
    total += e;
    if (mask[i]) em += e;
  }
  return [em, total];
}

function pick(p, lgIn, invT, cutQ, rng, mask, gainQ, bias) {
  const K = p.K, n = lgIn.length;
  const [em, total] = classMass(p, lgIn, mask);
  const u = rng.next() >>> 16;
  const left = (BigInt(u) * BigInt(total)) << BigInt(K.GAIN_BITS);
  const right = (BigInt(em) * BigInt(gainQ)) << 16n;
  const keepMask = left < right;
  const lg = new Float64Array(n);
  for (let i = 0; i < n; i++) lg[i] = lgIn[i] + (keepMask && bias ? bias[i] : 0);
  let kmax = -Infinity;
  for (let i = 0; i < n; i++) {
    const keep = keepMask ? mask[i] : !mask[i];
    if (keep && lg[i] > kmax) kmax = lg[i];
  }
  let best = -1, bz = -Infinity;
  const sh = 32 - K.G_BITS;
  for (let i = 0; i < n; i++) {
    const draw = rng.next() >>> sh;
    const keep = keepMask ? mask[i] : !mask[i];
    if (!keep || lg[i] < kmax - cutQ) continue;
    const z = floorShr(lg[i] * invT, K.INV_T_BITS) + p.gumbel[draw];
    if (z > bz) { bz = z; best = i; }
  }
  return best < 0 ? 0 : best;
}

function log2q8(p, total) {
  let q = 0;
  while (total >= 512) { total = Math.floor(total / 2); q++; }
  return (q + 8 - 31) * 256 + p.log2_lut[total & 255];
}

function tokenScore(p, lg, tok, total, wq) {
  let mx = -Infinity;
  for (let i = 0; i < lg.length; i++) if (lg[i] > mx) mx = lg[i];
  const d = floorShr((mx - lg[tok]) * p.K.LOG2E_Q8, p.fmt.logit);
  return -d - log2q8(p, total) + wq[tok];
}

const WB = "(?<![\\p{L}\\p{N}_])", WE = "(?![\\p{L}\\p{N}_])";
const MOM = new RegExp(WB + "(your|ur|yo|you're|you’re) (mom|mama|momma|moms)" + WE, "u");
const AGGRO = new RegExp(WB + "(fuck|f\\*ck|fk|stfu|shut (up|it)|hate|stupid|dumb|trash|suck|ugly|annoying|bitch|loser|idiot|" +
  "cringe|mid|lame|weak|garbage|go away|stop talking|screw|unplug|fry|beat you|not funny|" +
  "(just|only|a|little|lil|dumb|stupid) (a )?bot|toaster|calculator|ratio|my mom|my mama|my moms|" +
  "ur mom|your mom|yo mama)" + WE, "iu");
const BLOCK = new RegExp("n[i1!]gg|f[a@]gg?[o0]t|" + WB + "f[a@]g" + WE + "|retard|tr[a@]nn|ch[i1]nk|" + WB + "sp[i1]c" + WE +
  "|k[i1]ke|dyke|g[o0][o0]k|wetback|beaner|kill (yo)?ur ?self|suicid|" + WB + "kys" + WE + "|" + WB + "unalive|" + WB +
  "(gay|lesbian|homo|queer|trans|black|white|asian|chinese|mexican|indian|arab|jew|muslim|christian|african|latin|" +
  "immigrant|autis|down syndrome|disabled|rape)[\\p{L}\\p{N}_]*", "iu");

class Person {
  constructor(p, tok) {
    this.p = p; this.tok = tok; this.C = p.chat;
    this.m = new IntModel(p);
    this.EMOJI = p.emoji_mask;
    this.wBase = Float64Array.from(p.w_base);
    this.wq = Float64Array.from(p.w_base);
    this.bias = Float64Array.from(p.emoji_bias);
    this.reset();
  }
  reset() {
    this.said = []; this.botMsgs = 0; this.botEmoji = 0; this.botMass = [];
    this.m.reset(); this.base = 0; this.hist = []; this.pos = 0; this.last = null;
  }
  emojiGain(emoji) {
    const C = this.C, scale = Math.pow(2 * Math.max(0, emoji), 2);
    if (!C.emoji_rate || scale === 0) return C.emoji_corr * scale;
    const target = Math.min(0.9, C.emoji_rate * scale);
    let need = (target * (this.botMsgs + C.EMOJI_HORIZON) - this.botEmoji) / C.EMOJI_HORIZON;
    if (need <= 0) return C.EMOJI_GAIN_MIN;
    need = Math.min(need, 0.9);
    const prior = C.emoji_cold || -Math.log(1 - target) / (C.emoji_corr * scale);
    const m = this.botMass.slice(-C.EMOJI_RECENT);
    for (let i = 0; i < Math.max(0, C.EMOJI_PRIOR - this.botMass.length); i++) m.push(prior);
    let lo = Math.log(C.EMOJI_GAIN_MIN), hi = Math.log(C.EMOJI_GAIN_MAX);
    for (let it = 0; it < 30; it++) {
      const mid = (lo + hi) / 2, g = Math.exp(mid);
      let s = 0;
      for (const v of m) s += 1 - Math.exp(-g * v);
      if (s / m.length < need) lo = mid; else hi = mid;
    }
    return Math.exp((lo + hi) / 2);
  }
  repeatTable() {
    const C = this.C, w = Float64Array.from(this.wBase), pen = C.REPEAT, q = Math.LOG2E * 256;
    if (pen) {
      const recent = this.said.slice(-C.REPEAT_N).reverse();
      recent.forEach((toks, age) => {
        const f = (C.REPEAT_N - age) / C.REPEAT_N;
        for (const t of new Set(toks)) {
          if (this.p.common[t] || t <= 3) continue;
          w[t] = Math.min(w[t], this.wBase[t] - pyRound(q * f * (this.EMOJI[t] ? C.REPEAT_EMOJI : pen)));
        }
      });
    }
    for (let i = 0; i < w.length; i++) w[i] = Math.min(32767, Math.max(-32768, w[i]));
    this.wq = w;
  }
  feed(ids) {
    for (const t of ids) { this.last = this.m.forward(t, this.pos, this.base); this.hist.push(t); this.pos++; }
  }
  settings(invT, cutQ, gainQ, rng) { this.invT = invT; this.cutQ = cutQ; this.gainQ = gainQ; this.rng = rng; }
  rewind(pos) { this.pos = pos; if (this.hist.length > pos) this.hist.length = pos; }
  skip(toks) { for (const t of toks) this.hist.push(t); this.pos += toks.length; }
  window(base) { this.base = base; }
  scoreAfter(t, nxt) {
    this.feed([t]);
    const [em, total] = classMass(this.p, this.last, this.EMOJI);
    return [nxt, em, total, tokenScore(this.p, this.last, nxt, total, this.wq)];
  }
  step(t) {
    this.feed([t]);
    const [em, total] = classMass(this.p, this.last, this.EMOJI);
    const g = this.EMOJI[t] ? (1 << this.p.K.GAIN_BITS) : this.gainQ;
    const nxt = pick(this.p, this.last, this.invT, this.cutQ, this.rng, this.EMOJI, g, this.bias);
    return [nxt, em, total, tokenScore(this.p, this.last, nxt, total, this.wq)];
  }
  bestReset() { this.best = null; this.cand = 0; this.acc = [0, 0]; }
  score(s) { this.acc[0] += s; this.acc[1] += 1; }
  candEnd(keep) {
    const [s, n] = this.acc;
    if (keep && n && (this.best === null || s * this.best[1] > this.best[0] * n)) this.best = [s, n, this.cand];
    this.cand++;
    this.acc = [0, 0];
    return this.best ? this.best[2] : 0;
  }
  makeRoom(need) {
    const C = this.C;
    if (this.pos + need <= C.NULL_BASE) return;
    const h = this.hist;
    const cut = [];
    for (let i = 0; i < h.length; i++) if ((h[i] === C.YOU || h[i] === C.HIM) && h.length - i <= C.KEEP) cut.push(i);
    const keep = cut.length ? h.slice(cut[0]) : [];
    this.m.reset(); this.hist = []; this.pos = 0;
    this.feed([C.BOS, ...keep]);
  }
}

function loops(p, toks) {
  const C = p.chat, w = toks.filter((t) => t !== C.HIM && !p.emoji_mask[t]);
  const tri = new Set();
  let n = 0;
  for (let i = 0; i + 2 < w.length; i++) { tri.add(w[i] + "," + w[i + 1] + "," + w[i + 2]); n++; }
  if (tri.size < n) return true;
  for (let i = 0; i + 2 < w.length; i++) if (w[i] === w[i + 1] && w[i + 1] === w[i + 2]) return true;
  return false;
}

function blocked(tok, toks) { return BLOCK.test(tok.decode(toks)); }
function hasMom(tok, toks) { return MOM.test(tok.decode(toks).toLowerCase()); }
function momBad(person, tok, toks, text) {
  const said = (t) => MOM.test(tok.decode(t).toLowerCase());
  return said(toks) && (!AGGRO.test(text.split("’").join("'")) || person.said.slice(-person.C.MOM_GAP).some(said));
}

function replyBest(person, tok, text, temp, cut, rng, emoji, nBest, budget) {
  const p = person.p, C = p.chat, E = person.EMOJI;
  emoji = emoji == null ? C.EMOJI_LEVEL : emoji;
  budget = budget == null ? C.BUDGET : budget;
  nBest = nBest == null ? C.N_BEST : nBest;
  const ids = [...(person.pos === 0 ? [C.BOS] : []), C.YOU, ...tok.encode(text), C.HIM];
  person.makeRoom(ids.length + C.MAX_REPLY);
  person.repeatTable();
  person.settings(Math.max(1, pyRound(256 / temp)), pyRound(cut * P2[p.fmt.logit]),
                  pyRound(person.emojiGain(emoji) * P2[p.K.GAIN_BITS]), rng);
  person.feed(ids.slice(0, -1));
  const p0 = person.pos;
  person.bestReset();
  const cands = [], moms = [], means = new Map(), sums = new Map();
  let used = 0, last = -1, win = 0;
  const noMom = !AGGRO.test(text.split("’").join("'")) || person.said.slice(-C.MOM_GAP).some((r) => hasMom(tok, r));
  const starts = C.MOM_START;
  const force = noMom || !starts || !starts.length ? [] : tok.encode(starts[person.botMsgs % starts.length]);
  for (let k = 0; k < nBest + C.BEST_EXTRA; k++) {
    const lim = means.size || !k ? budget : budget + C.BEST_EXTRA_BUDGET;
    if ((k >= nBest && means.size) || (k && used > lim - C.BEST_RESERVE)) break;
    person.rewind(p0);
    const queue = k === 0 ? force.slice() : [];
    let [nxt, em, total, sc] = queue.length ? person.scoreAfter(C.HIM, queue.shift()) : person.step(C.HIM);
    const fed = [], msgs = [], mass = [], scored = [];
    let cur = [], n = 0, prev = C.HIM, curMass = 0, ok = true;
    while (n < C.MAX_REPLY) {
      if (!E[prev]) curMass += em / Math.max(total, 1);
      person.score(sc);
      scored.push(nxt);
      n++;
      if (used + n > lim) { ok = false; break; }
      if (nxt === C.YOU || nxt === C.BOS || nxt === C.PAD) break;
      prev = nxt;
      fed.push(nxt);
      if (noMom && hasMom(tok, fed.slice(-4))) { ok = false; break; }
      let following = null;
      if (n < C.MAX_REPLY) following = queue.length ? person.scoreAfter(nxt, queue.shift()) : person.step(nxt);
      else person.feed([nxt]);
      if (nxt === C.HIM) { msgs.push(cur); mass.push(curMass); cur = []; curMass = 0; }
      else cur.push(nxt);
      if (n < C.MAX_REPLY) [nxt, em, total, sc] = following;
    }
    msgs.push(cur); mass.push(curMass);
    used += n;
    last = k;
    let good = ok && !loops(p, fed) && !momBad(person, tok, fed, text) && !blocked(tok, fed);
    if (good && person.acc[1] && hasMom(tok, fed)) moms.push([person.acc[0] / person.acc[1], k]);
    if (k === 0 && force.length) good = false;
    if (good && person.acc[1]) {
      means.set(k, person.acc[0] / person.acc[1]);
      let w = 0;
      for (const t of scored) w += person.wq[t];
      sums.set(k, [person.acc[0] - w, scored]);
    }
    win = person.candEnd(good);
    cands.push([fed, msgs, mass, ok]);
  }
  const top = [...means.keys()].sort((a, b) => means.get(b) - means.get(a)).slice(0, C.MMI_TOP);
  const moved = top.length > 1;
  if (moved) {
    person.window(C.NULL_BASE);
    person.rewind(C.NULL_BASE);
    person.feed([C.BOS, C.YOU]);
    const mi = new Map();
    for (const c of top) {
      const [s, toks] = sums.get(c);
      person.rewind(C.NULL_BASE + 2);
      let s0 = 0;
      const prevs = [C.HIM, ...toks];
      for (let i = 0; i < toks.length; i++) s0 += person.scoreAfter(prevs[i], toks[i])[3] - person.wq[toks[i]];
      mi.set(c, (s - C.MMI_LAM * s0) / toks.length);
    }
    person.window(0);
    win = top[0];
    for (const c of top) if (mi.get(c) > mi.get(win)) win = c;
  }
  if (moms.length && means.has(win)) {
    let bm = moms[0];
    for (const mm of moms) if (mm[0] > bm[0] || (mm[0] === bm[0] && mm[1] > bm[1])) bm = mm;
    if (bm[0] >= means.get(win) - C.MOM_SLACK) win = bm[1];
  }
  if (win !== last) { person.rewind(p0); person.feed([C.HIM, ...cands[win][0]]); }
  else if (moved) { person.rewind(p0); person.skip([C.HIM, ...cands[win][0]]); }
  const [, msgsW, massW] = cands[win];
  person.said.push(msgsW.flat());
  msgsW.forEach((m, i) => { if (m.length) person.botMass.push(massW[i]); });
  const kept = msgsW.filter((m) => m.length);
  person.botMsgs += kept.length;
  person.botEmoji += kept.filter((m) => m.some((t) => E[t])).length;
  return { texts: kept.map((m) => tok.decode(m).trim()), used, win };
}

if (typeof module !== "undefined") {
  module.exports = { loadModel, IntModel, XorShift, makeRng, classMass, pick, tokenScore, Person, replyBest, pyRound };
}
