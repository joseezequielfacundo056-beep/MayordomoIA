// Service worker minimo: no cachea nada de forma agresiva (el chat necesita
// red sí o sí para hablar con Gemini), solo existe para que Chrome/Android
// consideren la app "instalable" como PWA de verdad. En iOS no hace falta
// (Safari no lo exige para agregar a inicio), pero no molesta tenerlo.
const CACHE = "mayordomo-shell-v1";
const SHELL = ["/static/icon-192.png", "/static/icon-512.png"];

self.addEventListener("install", (event) => {
  self.skipWaiting();
  event.waitUntil(caches.open(CACHE).then((c) => c.addAll(SHELL)).catch(() => {}));
});

self.addEventListener("activate", (event) => {
  event.waitUntil(self.clients.claim());
});

// network-first: si hay conexion, siempre trae lo mas nuevo (nunca queremos
// que quede pegado en una version vieja del chat); si no hay red, cae al
// cache minimo del shell (solo iconos, no la pagina ni las charlas).
self.addEventListener("fetch", (event) => {
  if (event.request.method !== "GET") return;
  event.respondWith(
    fetch(event.request).catch(() => caches.match(event.request))
  );
});
