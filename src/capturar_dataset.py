"""Captura del dataset propio con la plantilla de encuadre en pantalla.

Uso:
  uv run python -m src.capturar_dataset --persona p05 --clase falta_bata
  uv run python -m src.capturar_dataset --persona p05 --clase sin_epp --fuente 1

Teclas: ESPACIO inicia la tanda (cuenta regresiva de 3 s) | g mostrar/ocultar guia | q salir
Cada tanda guarda --fotogramas imagenes (60 por defecto), una cada --intervalo s (0.5 -> 30 s),
y para sola.

Salida: dataset/raw/<clase>/<persona>__<clase>_<fecha>_<nnn>.jpg
  El prefijo "<persona>__" es obligatorio: eda_limpieza lo usa para separar train/test por
  persona (sin fuga de datos). Use siempre el mismo codigo para la misma persona.

La plantilla (cuadros de cabeza, bata, manos y pies) se dibuja SOLO en pantalla:
las fotos se guardan limpias.
"""
import argparse
import re
import time
from pathlib import Path

import cv2
import numpy as np

from .common import RAW
from .guia_encuadre import aviso_aspecto, dibujar_guia

CLASES_VALIDAS = {
    "completo", "falta_cofia", "falta_guantes", "falta_bata", "sin_epp",
    "falta_cofia_guantes", "falta_cofia_bata", "falta_guantes_bata", "prueba",
}
QUE_LLEVA = {
    "completo": "cofia + guantes + bata",
    "falta_cofia": "guantes + bata (SIN cofia)",
    "falta_guantes": "cofia + bata (SIN guantes)",
    "falta_bata": "cofia + guantes (SIN bata)",
    "sin_epp": "NADA de EPP",
    "falta_cofia_guantes": "solo bata",
    "falta_cofia_bata": "solo guantes",
    "falta_guantes_bata": "solo cofia",
    "prueba": "prueba de camara (borrar despues)",
}
GUION = [  # (fraccion de la tanda hasta la que aplica, instruccion)
    (0.25, "De frente, manos AL FRENTE del pecho"),
    (0.50, "Girar despacio +-30 grados"),
    (0.75, "Acercarse hasta la marca de 1 m (cabeza siempre visible)"),
    (1.00, "Alejarse a la marca, manos a los lados"),
]


def guardar(img, ruta):
    """cv2.imwrite falla en silencio con tildes/ñ en la ruta (Windows); imencode no."""
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 95])
    if not ok:
        raise IOError(f"No se pudo codificar {ruta}")
    ruta.write_bytes(buf.tobytes())


def texto(img, t, y, esc, color=(255, 255, 255), x=12):
    g = max(1, int(round(img.shape[1] / 640)))
    cv2.putText(img, t, (x, y), cv2.FONT_HERSHEY_SIMPLEX, esc, (0, 0, 0), g + 3, cv2.LINE_AA)
    cv2.putText(img, t, (x, y), cv2.FONT_HERSHEY_SIMPLEX, esc, color, g, cv2.LINE_AA)


def componer_vista(frame, a, estado, n, mostrar_guia, cuenta=None):
    """Arma la imagen de PANTALLA (copia del fotograma + guia + estado)."""
    vista = frame.copy()
    h, w = vista.shape[:2]
    esc = max(0.45, w / 1280 * 0.6)
    if mostrar_guia:
        dibujar_guia(vista)
    paso = int(30 * esc / 0.6)
    y0 = h - paso * 3
    texto(vista, f"Persona {a.persona} | Clase {a.clase}: {QUE_LLEVA.get(a.clase, '')}", y0, esc,
          (0, 255, 255))
    if estado == "espera":
        texto(vista, "ESPACIO = iniciar tanda | g = guia | q = salir", y0 + paso, esc)
    elif estado == "cuenta":
        texto(vista, f"Comienza en {cuenta}...", y0 + paso, esc * 1.4, (0, 200, 255))
    elif estado == "grabando":
        avance = n / float(a.fotogramas)
        instr = next(t for lim, t in GUION if avance <= lim)
        texto(vista, f"GRABANDO {n}/{a.fotogramas} - {instr}", y0 + paso, esc, (0, 0, 255))
        bx0, bx1, by = 12, w - 12, y0 + paso * 2 - 8
        cv2.rectangle(vista, (bx0, by), (bx1, by + 12), (80, 80, 80), -1)
        cv2.rectangle(vista, (bx0, by), (bx0 + int((bx1 - bx0) * avance), by + 12), (0, 0, 255), -1)
        cv2.circle(vista, (w - 30, 30), 12, (0, 0, 255), -1)
    elif estado == "lista":
        texto(vista, f"Tanda lista: {n} fotos. ESPACIO = otra tanda | q = salir", y0 + paso, esc,
              (0, 255, 0))
    return vista


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--persona", required=True, help="codigo de la persona: p01, p02, ...")
    ap.add_argument("--clase", required=True, help=", ".join(sorted(CLASES_VALIDAS)))
    ap.add_argument("--fuente", type=int, default=0)
    ap.add_argument("--ancho", type=int, default=1280)
    ap.add_argument("--alto", type=int, default=720)
    ap.add_argument("--fotogramas", type=int, default=60)
    ap.add_argument("--intervalo", type=float, default=0.5, help="segundos entre fotos")
    ap.add_argument("--sin-guia", action="store_true", help="iniciar con la plantilla oculta")
    a = ap.parse_args()

    if "__" in a.persona or not re.fullmatch(r"[A-Za-z0-9_-]+", a.persona):
        raise SystemExit("--persona solo con letras/numeros (ej. p05) y sin '__'.")
    if not re.fullmatch(r"p\d{2,}", a.persona) and a.persona != "prueba":
        print(f"AVISO: '{a.persona}' no sigue el formato p01, p02... Verifique que sea el codigo correcto.")
    if a.clase not in CLASES_VALIDAS:
        raise SystemExit(f"Clase '{a.clase}' no valida. Use una de: {', '.join(sorted(CLASES_VALIDAS))}")

    destino = Path(RAW) / a.clase
    destino.mkdir(parents=True, exist_ok=True)
    previas = len(list(destino.glob(f"{a.persona}__*.jpg")))
    if previas:
        print(f"AVISO: ya hay {previas} fotos de {a.persona} en {a.clase}; las nuevas se agregan.")

    cap = cv2.VideoCapture(a.fuente)
    if not cap.isOpened():
        raise SystemExit(f"No se pudo abrir la fuente de video {a.fuente}. Cierre otras apps o pruebe --fuente 1")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, a.ancho)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, a.alto)
    ok, frame = cap.read()
    if not ok:
        raise SystemExit("La camara no entrega imagen.")
    aviso = aviso_aspecto(frame.shape[1], frame.shape[0])
    if aviso:
        print(aviso)
    print(f"Camara {frame.shape[1]}x{frame.shape[0]} | {a.persona} / {a.clase}: {QUE_LLEVA[a.clase]}")
    print("ESPACIO iniciar tanda | g guia | q salir")

    ventana = "Captura dataset (q para salir)"
    estado, n, mostrar_guia = "espera", 0, not a.sin_guia
    t_inicio_cuenta = t_ultima = 0.0
    sello = ""
    total = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            print("Se perdio la imagen de la camara.")
            break
        ahora = time.time()
        cuenta = None

        if estado == "cuenta":
            restante = 3 - int(ahora - t_inicio_cuenta)
            if restante <= 0:
                estado, n, t_ultima = "grabando", 0, 0.0
                sello = time.strftime("%Y%m%d_%H%M%S")
            else:
                cuenta = restante

        if estado == "grabando" and ahora - t_ultima >= a.intervalo:
            ruta = destino / f"{a.persona}__{a.clase}_{sello}_{n:03d}.jpg"
            guardar(frame, ruta)          # se guarda el fotograma LIMPIO, sin guia
            n += 1
            total += 1
            t_ultima = ahora
            if n >= a.fotogramas:
                estado = "lista"
                print(f"Tanda completa: {n} fotos en {destino}")

        cv2.imshow(ventana, componer_vista(frame, a, estado, n, mostrar_guia, cuenta))
        k = cv2.waitKey(1) & 0xFF
        if k == ord("q"):
            break
        if k == ord("g"):
            mostrar_guia = not mostrar_guia
        if k == ord(" ") and estado in ("espera", "lista"):
            estado, t_inicio_cuenta = "cuenta", ahora

    cap.release()
    cv2.destroyAllWindows()
    if estado == "grabando":
        print(f"Tanda interrumpida: se guardaron {n} de {a.fotogramas} fotos.")
    print(f"Total guardadas en esta sesion: {total} ({destino})")


if __name__ == "__main__":
    main()
