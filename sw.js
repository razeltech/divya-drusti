// DIVYA service worker - app shell + runtime cache for MediaPipe (CDN + models) so it works offline after first load
const VERSION = "divya-v13";
const SHELL = [
  "./",
  "./index.html",
  "./manifest.json",
  "./css/app.css?v=12",
  "./js/app.js?v=12",
  "./icons/icon-192.png",
  "./icons/icon-512.png"
];
const RUNTIME_HOSTS = ["cdn.jsdelivr.net", "storage.googleapis.com"];

self.addEventListener("install", e => {
  e.waitUntil(
    caches.open(VERSION).then(c => c.addAll(SHELL)).then(() => self.skipWaiting())
  );
});

self.addEventListener("activate", e => {
  e.waitUntil(
    caches.keys().then(keys => Promise.all(keys.filter(k => k !== VERSION).map(k => caches.delete(k))))
      .then(() => self.clients.claim())
  );
});

self.addEventListener("fetch", e => {
  const req = e.request;
  if (req.method !== "GET") return;
  const url = new URL(req.url);
  const same = url.origin === location.origin;
  if (!same && !RUNTIME_HOSTS.includes(url.hostname)) return;

  if (req.mode === "navigate") {
    // Network first for HTML navigation so updates arrive immediately
    e.respondWith(
      fetch(req)
        .then(r => {
          if (r && r.ok && !r.bodyUsed) {
            try {
              const copy = r.clone();
              caches.open(VERSION).then(c => c.put("./index.html", copy)).catch(() => {});
            } catch (err) {}
          }
          return r;
        })
        .catch(() => caches.match("./index.html"))
    );
    return;
  }

  e.respondWith(
    caches.match(req).then(hit => {
      if (hit) return hit;
      return fetch(req)
        .then(r => {
          if (!r || !r.ok) {
            // Fallback for any .glb requested at root
            if (r && r.status === 404 && url.pathname.endsWith(".glb")) {
              const file = url.pathname.split("/").pop();
              const alt = url.pathname.includes("/assets/") ? "./" + file : "./assets/" + file;
              return fetch(alt)
                .then(r2 => {
                  if (r2 && r2.ok && !r2.bodyUsed) {
                    try {
                      const copy2 = r2.clone();
                      caches.open(VERSION).then(c => c.put(req, copy2)).catch(() => {});
                    } catch (e) {}
                  }
                  return r2;
                })
                .catch(() => r);
            }
            return r;
          }
          if (!r.bodyUsed) {
            try {
              const copy = r.clone();
              caches.open(VERSION).then(c => c.put(req, copy)).catch(() => {});
            } catch (e) {}
          }
          return r;
        })
        .catch(() => caches.match(req));
    })
  );
});

self.addEventListener("message", e => {
  if (e.data && e.data.type === "SKIP_WAITING") {
    self.skipWaiting();
  }
});
