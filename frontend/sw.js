/* PocketDiffusion service worker: makes the web app installable and keeps it usable offline.
 *  - app shell (page, script, styles, icons): network first, cached copy when offline
 *  - generated images (/media): cache first - a stored image never changes
 *  - API calls and the live event stream: always the network (never cached)
 */
"use strict";
const VERSION = "pd-v1.7";
const SHELL = ["/", "/static/styles.css?v=1.7", "/static/app.js?v=1.7", "/static/icons/icon-192.png",
  "/static/icons/icon-512.png", "/manifest.webmanifest"];
const MEDIA = "pd-media-v1";
const MEDIA_MAX = 300;

self.addEventListener("install", (e) => {
  e.waitUntil(caches.open(VERSION).then((c) => c.addAll(SHELL)).catch(() => null).then(() => self.skipWaiting()));
});

self.addEventListener("activate", (e) => {
  e.waitUntil((async () => {
    for (const k of await caches.keys()) if (k !== VERSION && k !== MEDIA) await caches.delete(k);
    await self.clients.claim();
  })());
});

const OFFLINE = `<!doctype html><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>PocketDiffusion - offline</title><style>body{margin:0;min-height:100vh;display:grid;place-items:center;
background:#0b0b14;color:#ecebf6;font:16px/1.5 system-ui,sans-serif;text-align:center;padding:16px}
a{color:#8b5cf6}</style><div><h1>PocketDiffusion is offline</h1><p>The image server is not reachable right now.<br>
Check your connection and <a href="/">try again</a>.</p></div>`;

async function trimMedia(cache) {
  const keys = await cache.keys();
  for (let i = 0; i < keys.length - MEDIA_MAX; i++) await cache.delete(keys[i]);
}

self.addEventListener("fetch", (e) => {
  const req = e.request;
  if (req.method !== "GET") return;
  const url = new URL(req.url);
  if (url.origin !== location.origin) return;
  if (url.pathname.startsWith("/api/")) return;                     // live data: network only

  if (url.pathname.startsWith("/media/")) {                          // immutable images
    e.respondWith((async () => {
      const cache = await caches.open(MEDIA);
      const hit = await cache.match(req);
      if (hit) return hit;
      const res = await fetch(req);
      if (res.ok) { cache.put(req, res.clone()); trimMedia(cache); }
      return res;
    })());
    return;
  }

  e.respondWith((async () => {                                       // app shell: network first
    try {
      const res = await fetch(req);
      if (res.ok && (url.pathname === "/" || url.pathname.startsWith("/static/"))) {
        const cache = await caches.open(VERSION);
        cache.put(url.pathname === "/" ? "/" : req, res.clone());
      }
      return res;
    } catch {
      const hit = await caches.match(url.pathname === "/" ? "/" : req);
      if (hit) return hit;
      if (req.mode === "navigate") return new Response(OFFLINE, { headers: { "Content-Type": "text/html" } });
      throw new Error("offline");
    }
  })());
});
