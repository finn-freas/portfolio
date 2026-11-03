"use strict";

function bytesToUnicode() {
  const bs = [];
  for (let i = 33; i <= 126; i++) bs.push(i);
  for (let i = 161; i <= 172; i++) bs.push(i);
  for (let i = 174; i <= 255; i++) bs.push(i);
  const cs = bs.slice();
  let n = 0;
  for (let b = 0; b < 256; b++) if (!bs.includes(b)) { bs.push(b); cs.push(256 + n); n++; }
  const enc = new Array(256), dec = new Map();
  bs.forEach((b, i) => { enc[b] = String.fromCodePoint(cs[i]); dec.set(String.fromCodePoint(cs[i]), b); });
  return [enc, dec];
}

const PRE = /'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+/gu;

class Tokenizer {
  constructor(json) {
    const m = json.model;
    this.vocab = new Map(Object.entries(m.vocab));
    this.idToTok = [];
    for (const [t, i] of this.vocab) this.idToTok[i] = t;
    this.ranks = new Map();
    m.merges.forEach((mg, i) => this.ranks.set(Array.isArray(mg) ? mg.join(" ") : mg, i));
    this.added = json.added_tokens.slice().sort((a, b) => b.content.length - a.content.length);
    this.special = new Set(json.added_tokens.filter((a) => a.special).map((a) => a.id));
    for (const a of json.added_tokens) this.idToTok[a.id] = a.content;
    [this.byteEnc, this.byteDec] = bytesToUnicode();
    this.utf8 = new TextEncoder();
    this.cache = new Map();
  }

  bpe(word) {
    if (this.cache.has(word)) return this.cache.get(word);
    let parts = Array.from(word);
    while (parts.length > 1) {
      let best = -1, bestRank = Infinity;
      for (let i = 0; i < parts.length - 1; i++) {
        const r = this.ranks.get(parts[i] + " " + parts[i + 1]);
        if (r !== undefined && r < bestRank) { bestRank = r; best = i; }
      }
      if (best < 0) break;
      const a = parts[best], b = parts[best + 1], out = [];
      for (let i = 0; i < parts.length; i++) {
        if (i < parts.length - 1 && parts[i] === a && parts[i + 1] === b) { out.push(a + b); i++; }
        else out.push(parts[i]);
      }
      parts = out;
    }
    const ids = [];
    for (const p of parts) {
      const id = this.vocab.get(p);
      if (id !== undefined) ids.push(id);
    }
    this.cache.set(word, ids);
    return ids;
  }

  encodeSegment(text, out) {
    if (!text.startsWith(" ")) text = " " + text;
    for (const m of text.matchAll(PRE)) {
      let w = "";
      for (const b of this.utf8.encode(m[0])) w += this.byteEnc[b];
      for (const id of this.bpe(w)) out.push(id);
    }
  }

  encode(text) {
    const out = [];
    let seg = "", i = 0;
    while (i < text.length) {
      let hit = null;
      for (const a of this.added) if (text.startsWith(a.content, i)) { hit = a; break; }
      if (hit) {
        if (seg) { this.encodeSegment(seg, out); seg = ""; }
        out.push(hit.id);
        i += hit.content.length;
      } else {
        seg += text[i];
        i++;
      }
    }
    if (seg) this.encodeSegment(seg, out);
    return out;
  }

  decode(ids) {
    const bytes = [];
    for (const id of ids) {
      if (this.special.has(id)) continue;
      const t = this.idToTok[id];
      if (t === undefined) continue;
      const bs = [];
      let ok = true;
      for (const ch of t) {
        const b = this.byteDec.get(ch);
        if (b === undefined) { ok = false; break; }
        bs.push(b);
      }
      if (ok) bytes.push(...bs); else bytes.push(...this.utf8.encode(t));
    }
    return new TextDecoder("utf-8").decode(new Uint8Array(bytes));
  }
}

if (typeof module !== "undefined") module.exports = { Tokenizer };
