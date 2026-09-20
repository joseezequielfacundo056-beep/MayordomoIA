# gunicorn lee este archivo solo, sin tener que tocar el "Start Command"
# en Render.
#
# El default de gunicorn es matar el worker a los 30 segundos de silencio.
# Como el chat es streaming (la conexion queda abierta generando texto) y
# ahora ademas puede analizar imagenes/PDFs mas pesados, 30s se quedaba
# corto y cortaba respuestas legitimas a mitad de camino. Con los timeouts
# propios ya acotados del lado del cliente de Gemini (ver server.py), esto
# le da margen real sin permitir que un cuelgue se coma minutos enteros.
timeout = 90
graceful_timeout = 30
keepalive = 5
workers = 1
