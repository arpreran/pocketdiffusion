/* CLIP byte-level BPE tokenizer (the one Stable Diffusion's text encoder uses), dependency-free.
 * Mirrors transformers' CLIPTokenizer: lower-case, regex pre-split, GPT-2 byte->unicode mapping,
 * BPE with the merges ranks, "</w>" on the last symbol of every word, BOS/EOS, pad to 77.
 * Works in the browser (window.ClipTokenizer) and in Node (module.exports) for testing.
 */
(function (root) {
  "use strict";

  function bytesToUnicode() {
    const bs = [];
    for (let i = 33; i <= 126; i++) bs.push(i);
    for (let i = 161; i <= 172; i++) bs.push(i);
    for (let i = 174; i <= 255; i++) bs.push(i);
    const cs = bs.slice();
    let n = 0;
    for (let b = 0; b < 256; b++) {
      if (!bs.includes(b)) { bs.push(b); cs.push(256 + n); n++; }
    }
    const m = new Array(256);
    bs.forEach((b, i) => { m[b] = String.fromCharCode(cs[i]); });
    return m;
  }

  const PAT = /<\|startoftext\|>|<\|endoftext\|>|'s|'t|'re|'ve|'m|'ll|'d|[\p{L}]+|[\p{N}]|[^\s\p{L}\p{N}]+/giu;

  class ClipTokenizer {
    constructor(vocab, mergesText) {
      this.encoder = vocab;
      this.byteEncoder = bytesToUnicode();
      const lines = mergesText.trim().split("\n").slice(1, 49152 - 256 - 2 + 1);
      this.ranks = new Map();
      lines.forEach((l, i) => this.ranks.set(l.trim(), i));
      this.cache = new Map();
      this.bos = vocab["<|startoftext|>"];
      this.eos = vocab["<|endoftext|>"];
      this.utf8 = new TextEncoder();
    }

    static clean(text) {   // whitespace + lower case (no HTML unescaping: matches transformers without ftfy)
      return String(text).replace(/\s+/g, " ").trim().toLowerCase();
    }

    bpe(token) {
      if (this.cache.has(token)) return this.cache.get(token);
      let word = [...token];
      word[word.length - 1] += "</w>";
      if (word.length === 1) { this.cache.set(token, word); return word; }
      for (;;) {
        let best = null, bestRank = Infinity;
        for (let i = 0; i < word.length - 1; i++) {
          const r = this.ranks.get(word[i] + " " + word[i + 1]);
          if (r !== undefined && r < bestRank) { bestRank = r; best = i; }
        }
        if (best === null) break;
        const a = word[best], b = word[best + 1], out = [];
        for (let i = 0; i < word.length;) {
          if (i < word.length - 1 && word[i] === a && word[i + 1] === b) { out.push(a + b); i += 2; }
          else { out.push(word[i]); i += 1; }
        }
        word = out;
        if (word.length === 1) break;
      }
      this.cache.set(token, word);
      return word;
    }

    encode(text) {
      // SD-Turbo's tokenizer registers its pad token "!" as an added token, so transformers splits
      // every "!" out of the text and maps it straight to id 0 (never merged by BPE) - same here
      const ids = [];
      ClipTokenizer.clean(text).split("!").forEach((seg, k) => {
        if (k > 0) ids.push(this.encoder["!"]);
        for (const m of seg.matchAll(PAT)) {
          const piece = [...this.utf8.encode(m[0])].map((b) => this.byteEncoder[b]).join("");
          for (const t of this.bpe(piece)) {
            const id = this.encoder[t];
            if (id !== undefined) ids.push(id);
          }
        }
      });
      return ids;
    }

    /** [BOS, up to 75 tokens, EOS, pad...] (length 77). SD-Turbo pads with "!" (id 0). */
    inputIds(text, length = 77, padId = 0) {
      const body = this.encode(text).slice(0, length - 2);
      const ids = [this.bos, ...body, this.eos];
      while (ids.length < length) ids.push(padId);
      return { ids, nTokens: body.length, truncated: this.encode(text).length > length - 2 };
    }
  }

  if (typeof module !== "undefined" && module.exports) module.exports = { ClipTokenizer };
  else root.ClipTokenizer = ClipTokenizer;
})(typeof window !== "undefined" ? window : globalThis);
