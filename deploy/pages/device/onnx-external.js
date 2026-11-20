/* Split a single-file ONNX model into "graph + external data" without copying the weights.
 *
 * ONNX Runtime Web parses a single-file model inside its 4 GB WebAssembly heap, which fails for the
 * 1.6 GB SD-Turbo UNet. With external data the weights are streamed to the GPU instead. This module
 * rewrites only the protobuf structure: every large initializer's raw_data field is replaced by an
 * external_data reference (location / offset / length) that points INTO THE ORIGINAL FILE BUFFER, so
 * the original download itself is passed to ONNX Runtime as the external-data file (zero copy).
 *
 * ModelProto.graph = field 7, GraphProto.initializer = field 5, TensorProto.raw_data = field 9,
 * TensorProto.external_data = field 13 (StringStringEntryProto key=1/value=2), data_location = 14.
 */
(function (root) {
  "use strict";
  const EXTERNAL = 1;

  function readVarint(b, p) {
    let v = 0, mul = 1, byte;
    do {
      byte = b[p++];
      v += (byte & 0x7f) * mul;
      mul *= 128;
    } while (byte & 0x80);
    return [v, p];
  }

  function varint(v) {                       // non-negative integers up to 2^53
    const out = [];
    do {
      let byte = v % 128;
      v = Math.floor(v / 128);
      if (v > 0) byte |= 0x80;
      out.push(byte);
    } while (v > 0);
    return out;
  }

  /** Iterate the fields of one message: yields {field, wire, start, valStart, end}. */
  function* fields(b, start, end) {
    let p = start;
    while (p < end) {
      const fStart = p;
      let key;
      [key, p] = readVarint(b, p);
      const field = Math.floor(key / 8), wire = key & 7;
      let valStart = p, valEnd;
      if (wire === 0) { [, valEnd] = readVarint(b, p); }
      else if (wire === 1) valEnd = p + 8;
      else if (wire === 5) valEnd = p + 4;
      else if (wire === 2) { let len; [len, p] = readVarint(b, p); valStart = p; valEnd = p + len; }
      else throw new Error("unsupported protobuf wire type " + wire);
      yield { field, wire, start: fStart, valStart, end: valEnd };
      p = valEnd;
    }
  }

  class Writer {
    constructor() { this.parts = []; this.size = 0; }
    raw(u8) { this.parts.push(u8); this.size += u8.length; }
    bytes(arr) { this.raw(Uint8Array.from(arr)); }
    lenField(field, u8) { this.bytes(varint(field * 8 + 2)); this.bytes(varint(u8.length)); this.raw(u8); }
    varField(field, v) { this.bytes(varint(field * 8)); this.bytes(varint(v)); }
    str(field, s) { this.lenField(field, new TextEncoder().encode(s)); }
    finish() {
      const out = new Uint8Array(this.size);
      let o = 0;
      for (const p of this.parts) { out.set(p, o); o += p.length; }
      return out;
    }
  }

  function entry(key, value) {
    const w = new Writer();
    w.str(1, key); w.str(2, value);
    return w.finish();
  }

  function rewriteTensor(b, start, end, location, minBytes, stats) {
    let raw = null;
    for (const f of fields(b, start, end)) if (f.field === 9 && f.wire === 2) raw = f;
    if (!raw || raw.end - raw.valStart < minBytes) return b.subarray(start, end);   // keep small tensors inline
    const w = new Writer();
    for (const f of fields(b, start, end)) {
      if (f.field === 9 || f.field === 13 || f.field === 14) continue;
      w.raw(b.subarray(f.start, f.end));
    }
    w.lenField(13, entry("location", location));
    w.lenField(13, entry("offset", String(raw.valStart)));
    w.lenField(13, entry("length", String(raw.end - raw.valStart)));
    w.varField(14, EXTERNAL);
    stats.tensors++; stats.bytes += raw.end - raw.valStart;
    return w.finish();
  }

  function rewriteGraph(b, start, end, location, minBytes, stats) {
    const w = new Writer();
    for (const f of fields(b, start, end)) {
      if (f.field === 5 && f.wire === 2) w.lenField(5, rewriteTensor(b, f.valStart, f.end, location, minBytes, stats));
      else w.raw(b.subarray(f.start, f.end));
    }
    return w.finish();
  }

  /** model: Uint8Array of a single-file .onnx. Returns {graph: Uint8Array (small), stats}. The original
   *  `model` buffer is the external-data file: pass it as {path: location, data: model}. */
  function splitExternal(model, location = "model.onnx.data", minBytes = 1024) {
    const stats = { tensors: 0, bytes: 0 };
    const w = new Writer();
    for (const f of fields(model, 0, model.length)) {
      if (f.field === 7 && f.wire === 2) w.lenField(7, rewriteGraph(model, f.valStart, f.end, location, minBytes, stats));
      else w.raw(model.subarray(f.start, f.end));
    }
    return { graph: w.finish(), stats };
  }

  const api = { splitExternal };
  if (typeof module !== "undefined" && module.exports) module.exports = api;
  else root.OnnxExternal = api;
})(typeof window !== "undefined" ? window : globalThis);
