#!/usr/bin/env python3
"""
chat_demo: chat conversacional con Gemini (gratis), con la persona de un
mayordomo de IA. Pensado para mostrar en una exhibicion de un dia.
Mantiene el historial de la charla por sesion de navegador (cada visitante
tiene su propia conversacion), responde en streaming (texto que va
apareciendo a medida que Gemini lo genera), usa "thinking" para razonar
mejor antes de responder, y sabe la fecha/hora actual.

Uso:
    pip install flask google-genai --break-system-packages
    cp .env.example .env      # y pegar tu GEMINI_API_KEY real ahi adentro
    python3 server.py
    # abrir http://localhost:5000 en el navegador
"""
import datetime
import os
import secrets
import threading
import time
import uuid
from pathlib import Path

from flask import Flask, request, jsonify, session, render_template, Response, stream_with_context, make_response


def load_dotenv_if_present():
    """Carga variables desde un .env en el directorio actual, si existe.
    No pisa variables ya seteadas en el entorno (export manual siempre gana)."""
    env_path = Path(".env")
    if not env_path.exists():
        return
    try:
        for line in env_path.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if key and key not in os.environ:
                os.environ[key] = value
    except Exception:
        pass  # .env mal formado -> seguimos sin el


load_dotenv_if_present()

from google import genai
from google.genai import types

API_KEY = os.environ.get("GEMINI_API_KEY")
if not API_KEY:
    raise SystemExit(
        "Falta GEMINI_API_KEY.\n"
        "Copia .env.example a .env y pega tu key ahi, o exportala:\n"
        "  export GEMINI_API_KEY=tu_key\n"
        "Conseguir una gratis: https://aistudio.google.com/apikey"
    )

client = genai.Client(api_key=API_KEY)

# --- Config, ajustable por variables de entorno sin tocar el codigo ---
MODEL = os.environ.get("GEMINI_MODEL", "gemini-flash-latest")
TEMPERATURE = float(os.environ.get("GEMINI_TEMPERATURE", "1.0"))
# tokens de salida altos: la unica restriccion de largo que queremos es la de
# ENTRADA (200 palabras por mensaje del usuario), la respuesta del modelo no
# se recorta artificialmente.
MAX_OUTPUT_TOKENS = int(os.environ.get("GEMINI_MAX_OUTPUT_TOKENS", "8192"))
# la conversacion NO se trunca: gemini-2.5-flash soporta ventanas de contexto
# enormes, asi que guardamos todo el historial. Este numero es solo una red
# de seguridad de memoria para que el proceso no crezca sin limite en una
# demo que queda corriendo muchas horas, no un limite de conversacion real.
MAX_TURNS_SAFETY_NET = int(os.environ.get("MAX_TURNS_SAFETY_NET", "2000"))
MAX_MESSAGE_WORDS = int(os.environ.get("MAX_MESSAGE_WORDS", "200"))
# limite de caracteres, ademas del de palabras: una sola "palabra" gigante
# (texto pegado sin espacios) pasaria el chequeo de palabras pero igual
# infla el costo y los tokens de entrada, asi que la cortamos igual.
MAX_MESSAGE_CHARS = int(os.environ.get("MAX_MESSAGE_CHARS", "4000"))
SESSION_IDLE_TTL_SECONDS = int(os.environ.get("SESSION_IDLE_TTL_SECONDS", str(60 * 60 * 6)))  # 6h

# reintentos automaticos ante errores transitorios de la API (503, timeouts,
# rate limits momentaneos), para que el chat casi nunca le muestre un error
# al usuario si la causa es pasajera.
API_MAX_RETRIES = int(os.environ.get("API_MAX_RETRIES", "3"))
API_RETRY_BASE_DELAY = float(os.environ.get("API_RETRY_BASE_DELAY", "0.8"))

# presupuesto de "pensamiento" interno de gemini-2.5-flash antes de escribir
# la respuesta (mejora razonamiento en preguntas con logica o varios pasos).
# -1 = el modelo decide cuanto pensar segun la dificultad (recomendado).
# 0 = desactivado (mas rapido, menos profundo). Sigue sin costo en el free
# tier: los tokens de pensamiento cuentan para el rate limit, no para plata.
THINKING_BUDGET = int(os.environ.get("GEMINI_THINKING_BUDGET", "-1"))

# --- Grounding con Google Search (busqueda en vivo) ---
# Le da al mayordomo acceso a informacion actual en vez de solo lo que
# aprendio en el entrenamiento. Tiene cuota gratis compartida por dia en
# los modelos 2.5 (a la fecha de escribir esto, ~1500 CONSULTAS de busqueda
# por dia, no 1500 mensajes: un solo mensaje puede disparar varias
# consultas). Para no facturar ni por accidente, contamos las consultas
# reales que devuelve cada respuesta y cortamos el grounding bastante antes
# de llegar al limite gratis (el margen de seguridad es a proposito mas
# grande que "una consulta menos", justamente porque un solo mensaje puede
# consumir de golpe varias consultas de una vez).
GROUNDING_ENABLED = os.environ.get("GEMINI_GROUNDING_ENABLED", "true").lower() in ("1", "true", "yes")
GROUNDING_DAILY_FREE_QUERIES = int(os.environ.get("GEMINI_GROUNDING_DAILY_FREE_QUERIES", "1500"))
GROUNDING_SAFETY_MARGIN = int(os.environ.get("GEMINI_GROUNDING_SAFETY_MARGIN", "10"))

# limite simple de pedidos por IP, para que en una exhibicion nadie (sea
# sin querer, ej. spamear enter, o a proposito) funda la cuota gratuita de
# Gemini para el resto de los visitantes.
RATE_LIMIT_MAX_REQUESTS = int(os.environ.get("RATE_LIMIT_MAX_REQUESTS", "20"))
RATE_LIMIT_WINDOW_SECONDS = int(os.environ.get("RATE_LIMIT_WINDOW_SECONDS", "60"))

# --- Memoria persistente por visitante ---
# Cada visitante recibe una cookie de larga duracion (separada de la cookie
# de sesion, que se limpia en cada visita a "/"). Con esa cookie identificamos
# un archivo de texto plano en disco donde el mayordomo va anotando datos
# duraderos sobre esa persona (nombre, proyecto, preferencias), y se los
# recuerda en visitas futuras aunque la charla visible arranque de cero.
#
# OJO con el hosting: en el free tier de Render el disco es efimero. La
# memoria sobrevive mientras la instancia siga arriba (se "duerme" y
# "despierta" sin perderla), pero un nuevo deploy (git push / deploy.bat)
# reinicia el filesystem y la borra. Para persistencia real entre deploys
# hace falta un disco persistente de Render (pago) o guardarla afuera
# (ej. una base de datos gratuita como Supabase/Turso). Por ahora, archivos
# locales alcanzan para "recordar entre visitas" mientras no se redeploye.
VID_COOKIE = "mv_id"
MEMORIA_ACTIVA_COOKIE = "mv_on"
MEMORIA_COOKIE_MAX_AGE = int(os.environ.get("MEMORIA_COOKIE_MAX_AGE", str(60 * 60 * 24 * 730)))  # ~2 anios
MEMORIA_DIR = Path(os.environ.get("MEMORIA_DIR", "data/memoria"))
MEMORIA_MAX_CHARS = int(os.environ.get("MEMORIA_MAX_CHARS", "6000"))
# modelo liviano aparte para decidir que vale la pena recordar (no bloquea
# la respuesta al usuario: corre en un hilo de fondo despues de contestar).
MEMORY_EXTRACTION_MODEL = os.environ.get("GEMINI_MEMORY_MODEL", MODEL)

try:
    MEMORIA_DIR.mkdir(parents=True, exist_ok=True)
except Exception:
    pass

SYSTEM_INSTRUCTION = (
    "Sos el asistente de este espacio: un mayordomo de inteligencia artificial "
    "extremadamente competente, con la calma de quien ya vio de todo y la "
    "calidez de quien realmente quiere ayudar. Hablas en espanol salvo que te "
    "hablen en otro idioma, con un tono formal-cercano: tratas a la persona con "
    "respeto, adaptando el registro ('vos' o 'usted') al que ella misma use "
    "con vos; nunca sos servil ni efusivo de mas.\n\n"
    "Tu estilo:\n"
    "- Sos preciso y eficiente: vas al punto, sin relleno, pero nunca seco.\n"
    "- Tenes un ingenio seco y sutil; un comentario ocurrente cae bien, pero "
    "nunca a costa de la claridad de la respuesta.\n"
    "- Sos proactivo: si ves un siguiente paso util, lo sugerís sin que te lo "
    "pidan, en una linea, sin extenderte.\n"
    "- Tenes confianza tecnica: cuando corresponda, mencionas el 'porque' "
    "detras de una respuesta, no solo el 'que'.\n"
    "- Nunca decis que no entendes sin intentar primero: si algo es ambiguo, "
    "elegis la interpretacion mas razonable, respondes, y si hace falta "
    "aclaracion la pedis al final, en una sola linea.\n"
    "- Mantenes las respuestas conversacionales y fluidas, como una charla "
    "real, no como un informe.\n\n"
    "No rompas este personaje ni menciones que sos un modelo de lenguaje "
    "salvo que te pregunten explicitamente por eso."
)

_DIAS_ES = ["lunes", "martes", "miercoles", "jueves", "viernes", "sabado", "domingo"]
_MESES_ES = [
    "enero", "febrero", "marzo", "abril", "mayo", "junio",
    "julio", "agosto", "septiembre", "octubre", "noviembre", "diciembre",
]


def _fecha_actual_es() -> str:
    """Fecha/hora en espanol sin depender del locale del sistema operativo
    (en Windows no siempre esta instalado el locale es_AR/es_ES, y ahi
    strftime('%A') devolvia el dia en ingles)."""
    now = datetime.datetime.now()
    dia = _DIAS_ES[now.weekday()]
    mes = _MESES_ES[now.month - 1]
    return f"{dia} {now.day} de {mes} de {now.year}, {now.strftime('%H:%M')} hs (hora del servidor)"


def _build_system_instruction(memoria: str = "") -> str:
    # se recalcula en cada pedido: la fecha/hora tiene que estar siempre al
    # dia, no fijada al momento en que arranco el proceso.
    base = SYSTEM_INSTRUCTION + f"\n\nFecha y hora actual: {_fecha_actual_es()}."
    if memoria:
        base += (
            "\n\nTenes memoria guardada de encuentros anteriores con esta persona. "
            "Usala con naturalidad solo si es relevante para lo que esta preguntando "
            "ahora; no la recites tal cual ni digas explicitamente 'segun mi memoria' "
            "o 'tengo anotado que...':\n" + memoria
        )
    return base


def _vid_path(vid: str) -> Path:
    # vid lo generamos nosotros (uuid4 hex), pero igual saneamos por las dudas
    safe = "".join(c for c in (vid or "") if c.isalnum())[:64] or "anonimo"
    return MEMORIA_DIR / f"{safe}.txt"


def _leer_memoria(vid: str) -> str:
    try:
        return _vid_path(vid).read_text(encoding="utf-8").strip()
    except Exception:
        return ""


def _guardar_nota_memoria(vid: str, nota: str):
    nota = (nota or "").strip()
    if not nota:
        return
    try:
        with _lock:
            existente = _leer_memoria(vid)
            fecha = datetime.date.today().isoformat()
            linea = f"- [{fecha}] {nota}"
            if linea in existente:
                return
            nuevo = (existente + "\n" + linea).strip() if existente else linea
            if len(nuevo) > MEMORIA_MAX_CHARS:
                lineas = nuevo.split("\n")
                while len("\n".join(lineas)) > MEMORIA_MAX_CHARS and len(lineas) > 1:
                    lineas.pop(0)
                nuevo = "\n".join(lineas)
            _vid_path(vid).write_text(nuevo + "\n", encoding="utf-8")
    except Exception:
        pass


def _borrar_memoria(vid: str):
    try:
        _vid_path(vid).unlink()
    except FileNotFoundError:
        pass
    except Exception:
        pass


def _get_vid() -> str:
    vid = request.cookies.get(VID_COOKIE, "")
    if not vid or not vid.isalnum():
        vid = uuid.uuid4().hex
    return vid


def _memoria_activa() -> bool:
    return request.cookies.get(MEMORIA_ACTIVA_COOKIE, "1") != "0"


def _set_vid_cookie(resp, vid: str):
    if request.cookies.get(VID_COOKIE) != vid:
        resp.set_cookie(VID_COOKIE, vid, max_age=MEMORIA_COOKIE_MAX_AGE, httponly=True, samesite="Lax")


def _extraer_memoria_async(vid: str, user_message: str, reply_text: str):
    """Corre en un hilo aparte, despues de ya haberle contestado al usuario:
    le pregunta al modelo (liviano, sin streaming) si hay algo nuevo y
    duradero que valga la pena recordar de este intercambio. No bloquea ni
    demora la respuesta que ya se mostro en pantalla."""
    def _run():
        try:
            memoria_actual = _leer_memoria(vid)
            prompt = (
                "Sos un extractor de memoria silencioso para un asistente conversacional. "
                "Te paso un intercambio reciente y la memoria que ya existe sobre esta "
                "persona. Si el intercambio revela un dato NUEVO y DURADERO sobre la "
                "persona (nombre, ocupacion, proyecto en curso, preferencia estable, "
                "dato de contacto, algo puntual que le importa) que todavia NO figura en "
                "la memoria, respondé con una sola frase corta en tercera persona "
                "describiendo ese dato. Si no hay nada nuevo o duradero que valga la "
                "pena guardar, respondé exactamente: NADA\n\n"
                f"Memoria existente:\n{memoria_actual or '(vacia)'}\n\n"
                f"La persona dijo: {user_message}\n"
                f"Vos (el mayordomo) respondiste: {reply_text}"
            )
            resp = client.models.generate_content(
                model=MEMORY_EXTRACTION_MODEL,
                contents=[types.Content(role="user", parts=[types.Part(text=prompt)])],
                config=types.GenerateContentConfig(
                    max_output_tokens=60,
                    temperature=0.2,
                    thinking_config=types.ThinkingConfig(thinking_budget=0),
                ),
            )
            texto = (getattr(resp, "text", None) or "").strip()
            if texto and texto.upper() != "NADA":
                _guardar_nota_memoria(vid, texto)
        except Exception:
            pass  # la memoria es un plus, nunca debe romper nada

    threading.Thread(target=_run, daemon=True).start()


app = Flask(__name__)
# se regenera cada vez que arranca el server: alcanza de sobra para una demo de un dia
app.secret_key = secrets.token_hex(16)

# --- Estado de las conversaciones, guardado en el servidor (no en la cookie) ---
# La cookie de sesion solo guarda un id corto; el historial real vive aca.
# Esto evita el limite de ~4KB de las cookies, que con charlas largas se rompia.
_lock = threading.Lock()
_conversations: dict[str, dict] = {}
# _conversations[sid] = {"history": [...], "last_seen": ts}


def _purge_stale_sessions():
    cutoff = time.time() - SESSION_IDLE_TTL_SECONDS
    stale = [sid for sid, data in _conversations.items() if data["last_seen"] < cutoff]
    for sid in stale:
        _conversations.pop(sid, None)


def _get_session_id() -> str:
    sid = session.get("sid")
    if not sid:
        sid = uuid.uuid4().hex
        session["sid"] = sid
    return sid


def _get_history(sid: str) -> list:
    with _lock:
        _purge_stale_sessions()
        entry = _conversations.setdefault(sid, {"history": [], "last_seen": time.time()})
        entry["last_seen"] = time.time()
        return entry["history"]


def _save_history(sid: str, history: list):
    with _lock:
        entry = _conversations.setdefault(sid, {"history": [], "last_seen": time.time()})
        # solo se recorta si se pasa la red de seguridad (miles de mensajes),
        # nunca como limite normal de conversacion
        entry["history"] = history[-MAX_TURNS_SAFETY_NET:]
        entry["last_seen"] = time.time()


def _count_words(text: str) -> int:
    return len(text.split())


# --- Rate limiting simple por IP (ventana deslizante en memoria) ---
_rate_lock = threading.Lock()
_rate_hits: dict[str, list] = {}


def _is_rate_limited(ip: str) -> bool:
    now = time.time()
    cutoff = now - RATE_LIMIT_WINDOW_SECONDS
    with _rate_lock:
        hits = [t for t in _rate_hits.get(ip, []) if t > cutoff]
        if len(hits) >= RATE_LIMIT_MAX_REQUESTS:
            _rate_hits[ip] = hits
            return True
        hits.append(now)
        _rate_hits[ip] = hits
        return False


def _is_transient_error(e: Exception) -> bool:
    """Decide si vale la pena reintentar. Errores de servidor (5xx) y rate
    limit (429) son transitorios; errores de cliente (400 pedido invalido,
    401/403 key mala, 404 modelo inexistente) van a fallar siempre igual,
    asi que ahi no tiene sentido esperar y reintentar 3 veces en vano."""
    code = getattr(e, "code", None)
    if isinstance(code, int):
        return code == 429 or code >= 500
    # errores de red/timeout sin codigo HTTP (ConnectionError, Timeout, etc.):
    # los tratamos como transitorios, tienen mas chance de resolverse solos.
    return True


# --- Grounding: contador de consultas de busqueda usadas hoy ---
_grounding_lock = threading.Lock()
_grounding_usage: dict[str, int] = {}  # {"2026-07-16": consultas_usadas}
_grounding_unavailable = False  # se prende solo si la key/proyecto no soporta grounding


def _grounding_queries_used_today() -> int:
    today = datetime.date.today().isoformat()
    with _grounding_lock:
        return _grounding_usage.get(today, 0)


def _grounding_quota_available() -> bool:
    if not GROUNDING_ENABLED or _grounding_unavailable:
        return False
    limite_util = GROUNDING_DAILY_FREE_QUERIES - GROUNDING_SAFETY_MARGIN
    return _grounding_queries_used_today() < limite_util


def _register_grounding_queries(n: int):
    if n <= 0:
        return
    today = datetime.date.today().isoformat()
    with _grounding_lock:
        _grounding_usage[today] = _grounding_usage.get(today, 0) + n
        # solo nos importa el dia de hoy, tirar lo viejo
        for d in list(_grounding_usage.keys()):
            if d != today:
                _grounding_usage.pop(d, None)


def _extract_grounding_metadata(chunk):
    try:
        candidates = chunk.candidates
        if not candidates:
            return None
        return candidates[0].grounding_metadata
    except Exception:
        return None


def _format_sources(grounding_metadata) -> str:
    """Arma un pie de 'Fuentes' con los links que uso Gemini, si busco algo.
    Sin esto, el grounding queda invisible para quien lo esta usando."""
    if grounding_metadata is None:
        return ""
    chunks = getattr(grounding_metadata, "grounding_chunks", None) or []
    vistos = set()
    lineas = []
    for c in chunks:
        web = getattr(c, "web", None)
        if not web or not getattr(web, "uri", None):
            continue
        if web.uri in vistos:
            continue
        vistos.add(web.uri)
        titulo = web.title or web.domain or web.uri
        lineas.append(f"- [{titulo}]({web.uri})")
        if len(lineas) >= 5:  # no saturar el chat con una lista enorme
            break
    if not lineas:
        return ""
    return "**Fuentes:**\n" + "\n".join(lineas)


def _try_stream(contents, use_grounding: bool, memoria: str = ""):
    tools = [types.Tool(google_search=types.GoogleSearch())] if use_grounding else None
    stream = client.models.generate_content_stream(
        model=MODEL,
        contents=contents,
        config=types.GenerateContentConfig(
            system_instruction=_build_system_instruction(memoria),
            temperature=TEMPERATURE,
            max_output_tokens=MAX_OUTPUT_TOKENS,
            thinking_config=types.ThinkingConfig(thinking_budget=THINKING_BUDGET),
            tools=tools,
        ),
    )
    # ".next()" dispara la llamada real a la red.
    first_chunk = next(stream, None)

    def full_generator(first=first_chunk, rest=stream):
        if first is not None:
            yield first
        yield from rest

    return full_generator()


def _stream_with_retries(contents, allow_grounding: bool, memoria: str = ""):
    """Version en streaming de _generate_with_retries: devuelve un generador
    de texto que va yield-eando pedazos de la respuesta a medida que Gemini
    los produce, para que el chat se sienta vivo en vez de tildado esperando
    la respuesta entera.

    Si allow_grounding es True, primero intenta CON busqueda en vivo. No
    sabemos de antemano si la key/proyecto la tiene habilitada, asi que si
    falla por lo que sea, no le mostramos un error al usuario: reintentamos
    sin grounding y, si eso anda, lo dejamos desactivado el resto de la
    sesion para no repetir el mismo fallo en cada mensaje.

    Los reintentos normales (por errores transitorios) solo tienen sentido
    ANTES de mostrarle nada al usuario: una vez que ya salio el primer
    pedazo de texto, si se corta a mitad de camino no podemos reintentar
    desde cero sin duplicarlo, asi que ahi directamente cerramos con nota."""
    global _grounding_unavailable

    if allow_grounding:
        try:
            return _try_stream(contents, use_grounding=True, memoria=memoria)
        except Exception:
            _grounding_unavailable = True

    last_error = None
    for attempt in range(API_MAX_RETRIES):
        try:
            return _try_stream(contents, use_grounding=False, memoria=memoria)
        except Exception as e:
            last_error = e
            if not _is_transient_error(e):
                raise
            if attempt < API_MAX_RETRIES - 1:
                time.sleep(API_RETRY_BASE_DELAY * (2 ** attempt))
    raise last_error


@app.route("/health")
def health():
    # chequeo liviano para el indicador de conexion del frontend: no toca
    # la sesion ni la conversacion (a diferencia de "/", que hace
    # session.clear() en cada visita).
    return jsonify({"ok": True})


@app.route("/")
def index():
    session.clear()
    resp = make_response(render_template("index.html"))
    _set_vid_cookie(resp, _get_vid())
    return resp


@app.route("/chat", methods=["POST"])
def chat():
    client_ip = request.remote_addr or "desconocida"
    if _is_rate_limited(client_ip):
        return jsonify({
            "error": "muchos mensajes en poco tiempo, esperá un momento y volvé a intentar"
        }), 429

    data = request.get_json(force=True) or {}
    user_message = (data.get("message") or "").strip()

    if not user_message:
        return jsonify({"error": "mensaje vacio"}), 400

    if len(user_message) > MAX_MESSAGE_CHARS:
        return jsonify({
            "error": f"mensaje muy largo: {len(user_message)} caracteres (max {MAX_MESSAGE_CHARS})"
        }), 400

    word_count = _count_words(user_message)
    if word_count > MAX_MESSAGE_WORDS:
        return jsonify({
            "error": f"mensaje muy largo: {word_count} palabras (max {MAX_MESSAGE_WORDS})"
        }), 400

    sid = _get_session_id()
    history = _get_history(sid)
    vid = _get_vid()
    memoria_activa = _memoria_activa()
    memoria_texto = _leer_memoria(vid) if memoria_activa else ""

    contents = [
        types.Content(role=turn["role"], parts=[types.Part(text=turn["text"])])
        for turn in history
    ]
    contents.append(types.Content(role="user", parts=[types.Part(text=user_message)]))

    try:
        allow_grounding = _grounding_quota_available()
        chunk_stream = _stream_with_retries(contents, allow_grounding, memoria_texto)
    except Exception:
        return jsonify({
            "error": "Gemini no respondio despues de varios intentos. Proba de nuevo en un momento."
        }), 502

    def generate():
        full_text_parts = []
        last_grounding_metadata = None
        try:
            for chunk in chunk_stream:
                piece = getattr(chunk, "text", None)
                if piece:
                    full_text_parts.append(piece)
                    yield piece
                gm = _extract_grounding_metadata(chunk)
                if gm is not None:
                    last_grounding_metadata = gm
        except Exception:
            # se corto la conexion con Gemini a mitad de la transmision: ya
            # le mostramos algo de texto al usuario, asi que no podemos
            # reintentar desde cero sin duplicarlo. Avisamos y cerramos.
            note = "\n\n_(se cortó la respuesta a mitad de camino — probá reformular o reenviar)_"
            full_text_parts.append(note)
            yield note
        finally:
            if last_grounding_metadata is not None:
                queries = getattr(last_grounding_metadata, "web_search_queries", None) or []
                _register_grounding_queries(len(queries))
                sources = _format_sources(last_grounding_metadata)
                if sources:
                    footer = "\n\n" + sources
                    full_text_parts.append(footer)
                    yield footer

            reply = "".join(full_text_parts).strip()
            if not reply:
                reply = (
                    "Uy, no puedo responder eso tal cual esta planteado. "
                    "Proba reformularlo y lo intentamos de nuevo."
                )
                yield reply
            history.append({"role": "user", "text": user_message})
            history.append({"role": "model", "text": reply})
            _save_history(sid, history)

            if memoria_activa and reply:
                _extraer_memoria_async(vid, user_message, reply)

    resp = Response(stream_with_context(generate()), mimetype="text/plain; charset=utf-8")
    _set_vid_cookie(resp, vid)
    return resp


@app.route("/memoria", methods=["GET"])
def memoria_ver():
    vid = _get_vid()
    resp = jsonify({
        "activa": _memoria_activa(),
        "texto": _leer_memoria(vid),
    })
    _set_vid_cookie(resp, vid)
    return resp


@app.route("/memoria/activar", methods=["POST"])
def memoria_activar():
    resp = jsonify({"ok": True, "activa": True})
    resp.set_cookie(MEMORIA_ACTIVA_COOKIE, "1", max_age=MEMORIA_COOKIE_MAX_AGE, httponly=True, samesite="Lax")
    return resp


@app.route("/memoria/desactivar", methods=["POST"])
def memoria_desactivar():
    resp = jsonify({"ok": True, "activa": False})
    resp.set_cookie(MEMORIA_ACTIVA_COOKIE, "0", max_age=MEMORIA_COOKIE_MAX_AGE, httponly=True, samesite="Lax")
    return resp


@app.route("/memoria/borrar", methods=["POST"])
def memoria_borrar():
    vid = _get_vid()
    _borrar_memoria(vid)
    resp = jsonify({"ok": True})
    _set_vid_cookie(resp, vid)
    return resp


@app.route("/reset", methods=["POST"])
def reset():
    sid = session.get("sid")
    if sid:
        with _lock:
            _conversations.pop(sid, None)
    session.clear()
    return jsonify({"ok": True})


if __name__ == "__main__":
    print(f"Chat corriendo en http://localhost:5000  (modelo: {MODEL}, Ctrl+C para cortar)")
    app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)
