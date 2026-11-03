"use strict";
importScripts("roastbot-engine.js", "roastbot-tokenizer.js");

let person = null, tok = null, rng = null, model = null, seed = null;

async function fetchWithProgress(url, total, onFrac) {
  const res = await fetch(url);
  if (!res.ok) throw new Error(url + " " + res.status);
  if (!res.body || !res.body.getReader) return await res.arrayBuffer();
  const reader = res.body.getReader(), chunks = [];
  let n = 0;
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    chunks.push(value);
    n += value.length;
    onFrac(Math.min(1, n / total));
  }
  const out = new Uint8Array(n);
  let o = 0;
  for (const c of chunks) { out.set(c, o); o += c.length; }
  return out.buffer;
}

function newRng() {
  return makeRng(seed == null ? (Math.random() * 4294967296) >>> 0 : seed);
}

async function load() {
  const [meta, tj] = await Promise.all([fetch("meta.json").then((r) => r.json()), fetch("tokenizer_chat.json").then((r) => r.json())]);
  const buf = await fetchWithProgress("model.bin", meta.size, (frac) => postMessage({ type: "progress", frac }));
  model = loadModel(meta, buf);
  tok = new Tokenizer(tj);
  person = new Person(model, tok);
  rng = newRng();
  postMessage({ type: "ready", seed });
}

let ready = null;

onmessage = async (ev) => {
  const msg = ev.data;
  if (msg.type === "init") {
    seed = msg.seed == null ? null : msg.seed >>> 0;
    ready = load().catch((e) => postMessage({ type: "error", message: String(e && e.message || e) }));
    return;
  }
  await ready;
  if (!person) return;
  if (msg.type === "reset") {
    person.reset();
    postMessage({ type: "reset" });
  } else if (msg.type === "say") {
    const t0 = performance.now();
    try {
      const C = model.chat;
      const out = replyBest(person, tok, msg.text, C.TEMP, C.CUT, rng, null, null, null);
      const texts = out.texts.map((t) => t.split("�").join("").trim()).filter(Boolean);
      postMessage({ type: "reply", texts: texts.length ? texts : ["..."], ms: performance.now() - t0, tokens: out.used });
    } catch (e) {
      postMessage({ type: "error", message: String(e && e.message || e) });
    }
  }
};
