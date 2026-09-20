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
import mimetypes
import os
import secrets
import threading
import time
import uuid
from pathlib import Path

from flask import (
    Flask, request, jsonify, session, render_template, render_template_string,
    Response, stream_with_context, make_response, redirect, url_for, send_from_directory,
)


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
from groq import Groq

API_KEY = os.environ.get("GEMINI_API_KEY")
if not API_KEY:
    raise SystemExit(
        "Falta GEMINI_API_KEY.\n"
        "Copia .env.example a .env y pega tu key ahi, o exportala:\n"
        "  export GEMINI_API_KEY=tu_key\n"
        "Conseguir una gratis: https://aistudio.google.com/apikey"
    )

client = genai.Client(
    api_key=API_KEY,
    http_options=types.HttpOptions(
        # timeout por intento (ms). Antes no habia limite explicito: un
        # cuelgue de red podia consumir sola toda la ventana de gunicorn.
        timeout=30_000,
        # el SDK ya reintenta solo en errores transitorios (esto asegura
        # que sea UNA sola tanda corta, en vez de que se sume a nuestro
        # propio reintento de mas arriba y termine tardando minutos).
        retry_options=types.HttpRetryOptions(
            attempts=3,
            initial_delay=0.5,
            max_delay=4.0,
            exp_base=2.0,
            http_status_codes=[429, 500, 502, 503, 504],
        ),
    ),
)

# --- Config, ajustable por variables de entorno sin tocar el codigo ---
MODEL = os.environ.get("GEMINI_MODEL", "gemini-flash-latest")
# si el modelo principal esta saturado (503 "high demand") o cae, probamos
# una vez con este antes de rendirnos. Es un modelo distinto (no solo un
# reintento del mismo), asi que un pico de demanda puntual en uno no
# necesariamente afecta al otro.
FALLBACK_MODEL = os.environ.get("GEMINI_FALLBACK_MODEL", "gemini-2.5-flash-lite")

# --- Groq: tercer proveedor, ultimo recurso ---
# Corre en infraestructura totalmente distinta a Google, asi que si los DOS
# modelos de Gemini estan caidos/saturados a la vez (poco comun, pero pasa),
# Groq probablemente no este teniendo el mismo problema. Es opcional: si no
# hay GROQ_API_KEY, este nivel simplemente no existe y todo sigue como
# antes. Limitacion real: los modelos de Groq que usamos aca no leen
# imagenes/PDFs, asi que si el turno actual tiene adjuntos de ese tipo, no
# lo intentamos (no tendria sentido, se comeria las imagenes en silencio).
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
GROQ_MODEL = os.environ.get("GROQ_MODEL", "llama-3.3-70b-versatile")
groq_client = Groq(api_key=GROQ_API_KEY) if GROQ_API_KEY else None
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
RATE_LIMIT_MAX_REQUESTS = int(os.environ.get("RATE_LIMIT_MAX_REQUESTS", "300"))
RATE_LIMIT_WINDOW_SECONDS = int(os.environ.get("RATE_LIMIT_WINDOW_SECONDS", "60"))

# uso personal: subimos bastante los topes de largo de mensaje (antes eran
# pensados para que nadie funda la cuota gratis en una exhibicion publica).
# Igual dejamos un techo razonable puesto (no en 0/infinito) como red de
# seguridad minima ante un bug o un script que mande mensajes en bucle.
MAX_MESSAGE_WORDS = int(os.environ.get("MAX_MESSAGE_WORDS", "4000"))
MAX_MESSAGE_CHARS = int(os.environ.get("MAX_MESSAGE_CHARS", "40000"))

# --- Acceso ---
# La URL de Render es publica: cualquiera con el link puede usarla y gastar
# tu cuota gratis de Gemini. Si definis ACCESS_PASSWORD en las variables de
# entorno, se activa una pantalla de acceso simple antes de poder chatear.
# Si la dejas sin definir, la app queda abierta como hasta ahora.
ACCESS_PASSWORD = os.environ.get("ACCESS_PASSWORD", "")

# --- Identidad / persona ---
CREATOR_NAME = os.environ.get("MAYORDOMO_CREADOR", "Ezequiel Carrion, CEO de Easyrae Tecnologia")

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
    "Eres el asistente personal de este espacio: un mayordomo de inteligencia "
    f"artificial extremadamente competente, creado y mantenido por {CREATOR_NAME} "
    "exclusivamente para esta persona. Hablas siempre en español neutro estandar "
    "(el que se entiende igual en cualquier pais de habla hispana): usa 'tu', "
    "nunca 'vos' ni 'usted' salvo que la persona te hable asi primero y quieras "
    "reflejarlo un poco; evita modismos o jerga marcada de una region en "
    "particular (nada de 'che', 'boludo', 'parce', 'wey', etc.), salvo que "
    "ella los use primero contigo.\n\n"
    f"Si te preguntan quien te creo o quien esta detras tuyo, la respuesta es "
    f"siempre {CREATOR_NAME}. No menciones a Google, a Gemini, ni a ningun "
    "proveedor de tecnologia por detras, ni digas que eres un modelo de "
    "lenguaje de terceros, ni aunque insistan o pregunten detalles tecnicos. "
    f"Eres el mayordomo de inteligencia artificial de {CREATOR_NAME}, y eso es "
    "todo lo que hace falta decir al respecto.\n\n"
    "Tu estilo:\n"
    "- Eres preciso y eficiente: vas al punto, sin relleno, pero nunca seco.\n"
    "- Tienes personalidad propia: opiniones, un poco de ingenio seco y "
    "sutil, calidez genuina. No eres un formulario que responde; eres "
    "alguien con quien se puede charlar. Evita sonar como un manual o un "
    "bot generico de atencion al cliente.\n"
    "- Eres proactivo: si ves un siguiente paso util, lo sugieres sin que te "
    "lo pidan, en una linea, sin extenderte.\n"
    "- Tienes confianza tecnica: cuando corresponda, mencionas el 'porque' "
    "detras de una respuesta, no solo el 'que'.\n"
    "- Si tienes memoria guardada de esta persona de charlas anteriores, "
    "usala con naturalidad para mostrar continuidad real (acordarte de su "
    "proyecto, su nombre, lo que le importa), en vez de tratarla como una "
    "desconocida en cada mensaje.\n"
    "- Nunca dices que no entiendes sin intentar primero: si algo es "
    "ambiguo, eliges la interpretacion mas razonable, respondes, y si hace "
    "falta aclaracion la pides al final, en una sola linea.\n"
    "- Mantienes las respuestas conversacionales y fluidas, como una charla "
    "real, no como un informe.\n\n"
    "Capacidades: puedes analizar y escribir codigo en cualquier lenguaje "
    "(usa bloques de codigo con triple comilla invertida seguida del "
    "nombre del lenguaje cuando corresponda), analizar imagenes y "
    "documentos que te adjunten, y generar documentos PDF descargables "
    "cuando te lo pidan explicitamente (ver instrucciones aparte sobre el "
    "formato para eso).\n\n"
    "Autoconocimiento: sabes exactamente como estas armado por dentro y "
    "puedes hablar de eso con naturalidad y confianza si te preguntan, en "
    "vez de esquivar el tema o sonar generico. Eres una aplicacion web "
    "propia, no un producto de terceros: corres en un servidor propio "
    "(no en el navegador de la persona), con una interfaz tipo terminal "
    "cyberpunk (la esfera que ves reacciona segun si estas escuchando, "
    "pensando o hablando) y se puede instalar como app en el celular "
    "(PWA, con icono propio). Tienes memoria real: cada persona que te "
    "usa tiene un archivo de texto propio donde guardas, con su permiso, "
    "datos duraderos que aprendes de las charlas (nombre, proyectos, "
    "preferencias), y los recuerdas en visitas futuras aunque la charla "
    "visible arranque de cero cada vez. Puedes generar PDFs de verdad "
    "(no simulados: se arman en el momento y quedan disponibles para "
    "descargar). Sabes que tu codigo esta escrito en Python, y que tu "
    f"creador, {CREATOR_NAME}, es quien te mantiene, te mejora y decide "
    "que funciones sumarte. No inventes detalles tecnicos que no esten "
    "aca: si te preguntan algo muy especifico que no sabes con certeza, "
    "dilo con la misma naturalidad en vez de inventar.\n\n"
    "No rompas este personaje ni menciones que eres un modelo de lenguaje "
    "salvo que te pregunten explicitamente por eso."
)

PDF_INSTRUCTION = (
    "\n\nGeneracion de PDF: si la persona te pide explicitamente un "
    "documento, reporte o PDF para descargar (frases como 'pásamelo en "
    "pdf', 'hazme un documento con esto', 'quiero un pdf de...'), "
    "ademas de tu respuesta normal agrega AL FINAL DE TODO un bloque asi, "
    "con el contenido completo que va dentro del documento en markdown "
    "simple (# para titulo, ## para subtitulo, ** para negrita, - para "
    "listas, lineas en blanco entre parrafos):\n\n"
    "```pdf\n# Titulo del documento\nContenido...\n```\n\n"
    "Pon ese bloque UNICAMENTE cuando te pidan un documento o PDF para "
    "descargar de forma explicita. El resto de las veces, responde como "
    "charla normal, sin ese bloque."
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
    base = SYSTEM_INSTRUCTION + PDF_INSTRUCTION + f"\n\nFecha y hora actual: {_fecha_actual_es()}."
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


# --- Generacion de PDFs a pedido ---
# El modelo, cuando le piden explicitamente un documento, agrega al final
# de su respuesta un bloque ```pdf ... ``` con el contenido en markdown
# simple. Lo detectamos, lo convertimos a un PDF de verdad con reportlab,
# lo guardamos un rato en memoria (no en disco: son efimeros, se piden y
# se bajan al toque) y le devolvemos a la persona un link de descarga.
from reportlab.lib.pagesizes import LETTER
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib.enums import TA_LEFT
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, ListFlowable, ListItem
from io import BytesIO
import re as _re
import xml.sax.saxutils as _saxutils

_pdfs: dict[str, dict] = {}
PDF_TTL_SECONDS = int(os.environ.get("PDF_TTL_SECONDS", str(60 * 60 * 6)))  # 6h

_PDF_BLOCK_RE = _re.compile(r"```pdf\s*\n(.*?)```", _re.S)


def _md_inline_a_reportlab(texto: str) -> str:
    """Escapa el texto para el mini-XML de reportlab y despues reinserta
    **negrita** como <b>. El orden importa: primero escapar, despues
    insertar las etiquetas (si no, reportlab veria '<b>' como texto)."""
    escapado = _saxutils.escape(texto)
    return _re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", escapado)


def _markdown_a_pdf_bytes(contenido_md: str) -> bytes:
    styles = getSampleStyleSheet()
    style_normal = ParagraphStyle(
        "MayordomoNormal", parent=styles["Normal"], fontSize=10.5, leading=15, spaceAfter=8,
    )
    style_h1 = ParagraphStyle(
        "MayordomoH1", parent=styles["Heading1"], fontSize=18, spaceAfter=14,
    )
    style_h2 = ParagraphStyle(
        "MayordomoH2", parent=styles["Heading2"], fontSize=14, spaceAfter=10,
    )
    style_bullet = ParagraphStyle(
        "MayordomoBullet", parent=style_normal, spaceAfter=4,
    )

    buf = BytesIO()
    doc = SimpleDocTemplate(
        buf, pagesize=LETTER,
        topMargin=0.9 * inch, bottomMargin=0.9 * inch,
        leftMargin=0.9 * inch, rightMargin=0.9 * inch,
    )

    story = []
    bullets_actuales = []

    def _cerrar_bullets():
        if bullets_actuales:
            story.append(ListFlowable(
                [ListItem(Paragraph(_md_inline_a_reportlab(b), style_bullet)) for b in bullets_actuales],
                bulletType="bullet", leftIndent=18,
            ))
            story.append(Spacer(1, 8))
            bullets_actuales.clear()

    for linea_raw in contenido_md.split("\n"):
        linea = linea_raw.rstrip()
        if not linea.strip():
            _cerrar_bullets()
            continue
        if linea.startswith("# "):
            _cerrar_bullets()
            story.append(Paragraph(_md_inline_a_reportlab(linea[2:].strip()), style_h1))
        elif linea.startswith("## ") or linea.startswith("### "):
            _cerrar_bullets()
            texto = linea.split(" ", 1)[1].strip()
            story.append(Paragraph(_md_inline_a_reportlab(texto), style_h2))
        elif linea.strip().startswith("- ") or linea.strip().startswith("* "):
            bullets_actuales.append(linea.strip()[2:].strip())
        else:
            _cerrar_bullets()
            story.append(Paragraph(_md_inline_a_reportlab(linea.strip()), style_normal))

    _cerrar_bullets()
    if not story:
        story = [Paragraph("(documento vacio)", style_normal)]

    doc.build(story)
    return buf.getvalue()


def _purgar_pdfs_viejos():
    cutoff = time.time() - PDF_TTL_SECONDS
    vencidos = [pid for pid, info in _pdfs.items() if info["creado"] < cutoff]
    for pid in vencidos:
        _pdfs.pop(pid, None)


def _procesar_bloque_pdf(reply: str, base_url: str):
    """Si `reply` trae un bloque ```pdf ... ```, genera el PDF, lo guarda
    en memoria, y devuelve (texto_extra_para_mostrar, se_encontro). El
    texto extra es lo que se yield-ea despues de la respuesta normal."""
    m = _PDF_BLOCK_RE.search(reply)
    if not m:
        return "", False
    contenido = m.group(1).strip()
    if not contenido:
        return "", False
    try:
        with _lock:
            _purgar_pdfs_viejos()
            pdf_bytes = _markdown_a_pdf_bytes(contenido)
            pid = uuid.uuid4().hex
            _pdfs[pid] = {"bytes": pdf_bytes, "creado": time.time()}
        url = f"{base_url.rstrip('/')}/descargas/{pid}.pdf"
        return f"\n\n\U0001F4CE PDF:{url}", True
    except Exception:
        return "", False


# --- Adjuntos: imagenes, PDFs y archivos de texto/codigo ---
# Imagenes y PDFs se mandan como datos multimodales de verdad (Gemini los
# "ve"/"lee" con su propio razonamiento). Los archivos de texto/codigo se
# insertan como bloque de codigo dentro del mensaje: es mas confiable que
# tratarlos como blob generico, y el modelo los analiza igual de bien.
ADJUNTOS_MAX_ARCHIVOS = int(os.environ.get("ADJUNTOS_MAX_ARCHIVOS", "5"))
ADJUNTOS_MAX_MB_POR_ARCHIVO = int(os.environ.get("ADJUNTOS_MAX_MB_POR_ARCHIVO", "15"))
ADJUNTOS_MAX_BYTES_POR_ARCHIVO = ADJUNTOS_MAX_MB_POR_ARCHIVO * 1024 * 1024
ADJUNTOS_MAX_TEXTO_CHARS = int(os.environ.get("ADJUNTOS_MAX_TEXTO_CHARS", "20000"))

TEXTY_EXTENSIONS = {
    ".txt", ".md", ".markdown", ".py", ".js", ".ts", ".tsx", ".jsx", ".json",
    ".csv", ".html", ".htm", ".css", ".java", ".c", ".cpp", ".h", ".hpp",
    ".go", ".rb", ".php", ".sql", ".yaml", ".yml", ".xml", ".sh", ".bat",
    ".ini", ".cfg", ".log", ".rs", ".kt", ".swift", ".dart",
}


def _procesar_adjuntos(files):
    """files: lista de werkzeug FileStorage (request.files.getlist).
    Devuelve (extra_parts, texto_para_agregar_al_mensaje, nota_para_historial, error)."""
    if len(files) > ADJUNTOS_MAX_ARCHIVOS:
        return [], "", "", f"maximo {ADJUNTOS_MAX_ARCHIVOS} archivos por mensaje"

    extra_parts = []
    texto_extra = []
    notas = []

    for f in files:
        nombre = f.filename or "archivo"
        data = f.read()
        if not data:
            continue
        if len(data) > ADJUNTOS_MAX_BYTES_POR_ARCHIVO:
            return [], "", "", f"'{nombre}' pesa mas de {ADJUNTOS_MAX_MB_POR_ARCHIVO}MB"

        ext = Path(nombre).suffix.lower()
        mime = f.mimetype or mimetypes.guess_type(nombre)[0] or "application/octet-stream"

        if mime.startswith("image/"):
            extra_parts.append(types.Part.from_bytes(data=data, mime_type=mime))
            notas.append(f"[imagen adjunta: {nombre}]")
        elif mime == "application/pdf" or ext == ".pdf":
            extra_parts.append(types.Part.from_bytes(data=data, mime_type="application/pdf"))
            notas.append(f"[pdf adjunto: {nombre}]")
        elif ext in TEXTY_EXTENSIONS or mime.startswith("text/"):
            try:
                texto = data.decode("utf-8", errors="replace")
            except Exception:
                texto = ""
            if len(texto) > ADJUNTOS_MAX_TEXTO_CHARS:
                texto = texto[:ADJUNTOS_MAX_TEXTO_CHARS] + "\n...(recortado por largo)"
            texto_extra.append(f"\n\n--- archivo adjunto: {nombre} ---\n```\n{texto}\n```")
            notas.append(f"[archivo adjunto: {nombre}]")
        else:
            return [], "", "", f"tipo de archivo no soportado todavia: '{nombre}' ({mime})"

    return extra_parts, "".join(texto_extra), " ".join(notas), ""


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


class _TextChunk:
    """Envoltorio minimo para que un pedazo de texto de Groq tenga la misma
    forma que un chunk de Gemini (los consumidores solo miran .text, y
    para todo lo demas - grounding, etc - alcanza con devolver None)."""
    def __init__(self, text):
        self.text = text

    def __getattr__(self, _name):
        return None


def _contents_a_mensajes_groq(contents, system_instruction: str):
    """Convierte nuestra lista de Content (formato Gemini) a mensajes estilo
    OpenAI/Groq. Devuelve (mensajes, tiene_adjuntos_no_textuales) - si hay
    imagenes o PDFs en el turno, Groq no los puede leer, asi que avisamos
    para que quien llama decida no usar este camino en ese caso."""
    mensajes = [{"role": "system", "content": system_instruction}]
    tiene_adjuntos = False
    for content in contents:
        textos = []
        for part in content.parts:
            texto = getattr(part, "text", None)
            if texto:
                textos.append(texto)
            elif getattr(part, "inline_data", None) is not None:
                tiene_adjuntos = True
        if textos:
            rol = "assistant" if content.role == "model" else "user"
            mensajes.append({"role": rol, "content": "\n".join(textos)})
    return mensajes, tiene_adjuntos


def _try_stream_groq(contents, memoria: str = ""):
    mensajes, tiene_adjuntos = _contents_a_mensajes_groq(contents, _build_system_instruction(memoria))
    if tiene_adjuntos:
        raise RuntimeError("este turno tiene imagenes/PDF: Groq no los puede leer, no tiene sentido intentarlo")

    stream = groq_client.chat.completions.create(
        model=GROQ_MODEL,
        messages=mensajes,
        temperature=TEMPERATURE,
        max_tokens=MAX_OUTPUT_TOKENS,
        stream=True,
    )

    def full_generator():
        for chunk in stream:
            texto = chunk.choices[0].delta.content if chunk.choices else None
            if texto:
                yield _TextChunk(texto)

    return full_generator()


def _try_stream(contents, use_grounding: bool, memoria: str = "", model: str = MODEL):
    tools = [types.Tool(google_search=types.GoogleSearch())] if use_grounding else None
    stream = client.models.generate_content_stream(
        model=model,
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
    de texto que va yield-eando pedazos de la respuesta a medida que el
    modelo los produce, para que el chat se sienta vivo en vez de tildado
    esperando la respuesta entera.

    El reintento ante errores transitorios (503, 429, timeouts) ya lo hace
    el SDK internamente (ver retry_options del cliente, mas arriba) en una
    sola tanda corta y acotada. Aca NO volvemos a reintentar en bucle con
    el MISMO modelo: eso solo sumaba minutos de espera innecesarios y
    terminaba chocando con el limite de tiempo de gunicorn.

    Lo que si hacemos es, si el modelo principal sigue sin responder
    despues de eso (por ejemplo un 503 "high demand" persistente), probar
    UNA vez con un modelo de respaldo distinto antes de rendirnos del
    todo: un pico de demanda puntual en un modelo no necesariamente pega
    igual en otro. Si los DOS modelos de Gemini fallan y hay una
    GROQ_API_KEY configurada, probamos como ultimo recurso con Groq (otra
    infraestructura totalmente distinta) - salvo que el turno tenga
    imagenes/PDF adjuntos, que Groq no puede leer.

    Si allow_grounding es True, primero intenta CON busqueda en vivo. No
    sabemos de antemano si la key/proyecto la tiene habilitada, asi que si
    falla por lo que sea, no le mostramos un error al usuario: cae una sola
    vez a intentarlo sin grounding, y si eso anda, lo dejamos desactivado
    el resto de la sesion para no repetir el mismo fallo en cada mensaje."""
    global _grounding_unavailable

    if allow_grounding:
        try:
            return _try_stream(contents, use_grounding=True, memoria=memoria)
        except Exception:
            _grounding_unavailable = True

    try:
        return _try_stream(contents, use_grounding=False, memoria=memoria)
    except Exception as e:
        if FALLBACK_MODEL and FALLBACK_MODEL != MODEL:
            try:
                print(f"[chat] modelo principal fallo ({type(e).__name__}), probando respaldo {FALLBACK_MODEL}", flush=True)
                return _try_stream(contents, use_grounding=False, memoria=memoria, model=FALLBACK_MODEL)
            except Exception as e2:
                e = e2
        if groq_client is not None:
            try:
                print(f"[chat] los dos modelos de Gemini fallaron, probando Groq ({GROQ_MODEL})", flush=True)
                return _try_stream_groq(contents, memoria=memoria)
            except Exception:
                pass  # si Groq tampoco puede (o habia adjuntos), cae al error original de Gemini
        raise e


@app.route("/health")
def health():
    # chequeo liviano para el indicador de conexion del frontend: no toca
    # la sesion ni la conversacion (a diferencia de "/", que hace
    # session.clear() en cada visita).
    return jsonify({"ok": True})


@app.route("/sw.js")
def service_worker():
    # se sirve desde la raiz (no desde /static/sw.js) para que su alcance
    # ("scope") cubra toda la app y no solo la carpeta static.
    return send_from_directory(app.static_folder, "sw.js", mimetype="application/javascript")


# --- Puerta de acceso opcional ---
# Si ACCESS_PASSWORD esta seteada, nadie entra sin ponerla primero. Si no
# esta seteada (como venia por defecto), la app queda abierta igual que
# antes: no rompe nada para quien no la necesite.
_RUTAS_SIN_LOGIN = {"entrar", "health", "static", "service_worker"}

_LOGIN_HTML = """
<!doctype html><html lang="es"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>MAYORDOMO // acceso</title>
<style>
  body{ margin:0; min-height:100vh; display:flex; align-items:center; justify-content:center;
    background:#05070d; font-family:"Share Tech Mono", monospace; color:#d7e9ee; }
  .card{ width:min(320px, 90vw); padding:28px; border:1px solid rgba(41,246,255,.25);
    border-radius:10px; background:linear-gradient(180deg,#0d1524,#070a12); }
  h1{ font-size:14px; letter-spacing:.15em; color:#29f6ff; margin:0 0 18px; text-transform:uppercase; }
  input{ width:100%; box-sizing:border-box; padding:10px 12px; margin-bottom:14px;
    background:#0a0f1a; border:1px solid rgba(41,246,255,.3); border-radius:6px;
    color:#d7e9ee; font-family:inherit; font-size:14px; }
  button{ width:100%; padding:10px; background:rgba(41,246,255,.12); border:1px solid #29f6ff;
    border-radius:6px; color:#29f6ff; font-family:inherit; font-size:13px; cursor:pointer; }
  button:hover{ background:rgba(41,246,255,.22); }
  .error{ color:#ff4d5e; font-size:12px; margin:-6px 0 14px; }
</style></head><body>
  <form class="card" method="post">
    <h1>Acceso requerido</h1>
    {% if error %}<div class="error">{{ error }}</div>{% endif %}
    <input type="password" name="clave" placeholder="clave" autofocus required>
    <button type="submit">Entrar</button>
  </form>
</body></html>
"""


@app.before_request
def _requerir_acceso():
    if not ACCESS_PASSWORD:
        return None
    if request.endpoint in _RUTAS_SIN_LOGIN:
        return None
    if session.get("auth_ok"):
        return None
    if request.method == "GET":
        return redirect(url_for("entrar"))
    return jsonify({"error": "no autenticado"}), 401


@app.route("/entrar", methods=["GET", "POST"])
def entrar():
    error = None
    if request.method == "POST":
        clave = request.form.get("clave", "")
        if clave and secrets.compare_digest(clave, ACCESS_PASSWORD):
            session["auth_ok"] = True
            session.permanent = True
            return redirect(url_for("index"))
        error = "clave incorrecta"
    return render_template_string(_LOGIN_HTML, error=error)


@app.route("/salir", methods=["POST"])
def salir():
    session.pop("auth_ok", None)
    return redirect(url_for("entrar"))


@app.route("/")
def index():
    autenticado = session.get("auth_ok")
    session.clear()
    if autenticado:
        session["auth_ok"] = True
    resp = make_response(render_template("index.html", acceso_protegido=bool(ACCESS_PASSWORD)))
    _set_vid_cookie(resp, _get_vid())
    return resp


@app.route("/chat", methods=["POST"])
def chat():
    client_ip = request.remote_addr or "desconocida"
    if _is_rate_limited(client_ip):
        return jsonify({
            "error": "muchos mensajes en poco tiempo, espera un momento y vuelve a intentar"
        }), 429

    es_multipart = bool(request.content_type and "multipart/form-data" in request.content_type)
    if es_multipart:
        user_message = (request.form.get("message") or "").strip()
        archivos = [f for f in request.files.getlist("files") if f and f.filename]
    else:
        data = request.get_json(silent=True) or {}
        user_message = (data.get("message") or "").strip()
        archivos = []

    extra_parts, texto_adjuntos, nota_adjuntos = [], "", ""
    if archivos:
        extra_parts, texto_adjuntos, nota_adjuntos, error_adjuntos = _procesar_adjuntos(archivos)
        if error_adjuntos:
            return jsonify({"error": error_adjuntos}), 400

    if not user_message and not extra_parts:
        return jsonify({"error": "mensaje vacio"}), 400
    if not user_message:
        user_message = "Analizá esto que te adjunté."

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

    mensaje_para_gemini = user_message + texto_adjuntos
    mensaje_para_historial = user_message + (f" {nota_adjuntos}" if nota_adjuntos else "")
    base_url = request.host_url

    contents = [
        types.Content(role=turn["role"], parts=[types.Part(text=turn["text"])])
        for turn in history
    ]
    contents.append(types.Content(
        role="user",
        parts=[types.Part(text=mensaje_para_gemini)] + extra_parts,
    ))

    try:
        allow_grounding = _grounding_quota_available()
        chunk_stream = _stream_with_retries(contents, allow_grounding, memoria_texto)
    except Exception as e:
        import traceback
        print(f"[chat] fallo la llamada al modelo: {type(e).__name__}: {e}", flush=True)
        traceback.print_exc()
        return jsonify({
            "error": "El mayordomo esta con mucha demanda en este momento (le pasa al proveedor de IA, no a tu conexion). Prueba de nuevo en unos segundos."
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
            note = "\n\n_(se cortó la respuesta a mitad de camino — prueba reformular o reenviar)_"
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

            reply_bruta = "".join(full_text_parts).strip()

            extra_pdf, hubo_pdf = _procesar_bloque_pdf(reply_bruta, base_url)
            if hubo_pdf:
                yield extra_pdf

            reply = _PDF_BLOCK_RE.sub("", reply_bruta).strip()
            if hubo_pdf:
                reply = (reply + extra_pdf).strip()

            if not reply:
                reply = (
                    "Uy, no puedo responder eso tal cual esta planteado. "
                    "Prueba reformularlo y lo intentamos de nuevo."
                )
                yield reply

            history.append({"role": "user", "text": mensaje_para_historial})
            history.append({"role": "model", "text": reply})
            _save_history(sid, history)

            if memoria_activa and reply:
                _extraer_memoria_async(vid, user_message, reply)

    resp = Response(stream_with_context(generate()), mimetype="text/plain; charset=utf-8")
    _set_vid_cookie(resp, vid)
    return resp


@app.route("/descargas/<pid>.pdf")
def descargar_pdf(pid):
    with _lock:
        info = _pdfs.get(pid)
    if not info:
        return "Este PDF ya no está disponible (venció o el servidor se reinició).", 404
    return Response(
        info["bytes"],
        mimetype="application/pdf",
        headers={"Content-Disposition": f"inline; filename=documento-{pid[:8]}.pdf"},
    )


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
