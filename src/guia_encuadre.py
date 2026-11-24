"""Plantilla de encuadre en vivo: cuadros de cabeza, bata, manos y marca de pies.

Sirve para que el estudiante se ubique siempre igual durante la captura del dataset propio.
Los cuadros salen de features.CAJAS (las mismas regiones que usa el modelo clasico), asi que
si recalibra CAJAS la plantilla se actualiza sola.

IMPORTANTE: la plantilla se dibuja SOLO en la vista de pantalla, nunca en la foto guardada
(si las lineas quedan en el dataset, el modelo las aprende).

Uso:
  uv run python -m src.guia_encuadre                      # camara 0
  uv run python -m src.guia_encuadre --fuente 1           # otra camara
  uv run python -m src.guia_encuadre --imagen foto.jpg --salida reports/guia.jpg   # sin camara
Teclas: g mostrar/ocultar guia | s guardar captura de pantalla (con guia) | q salir
"""
import argparse
import time
from pathlib import Path

import cv2
import numpy as np

from .features import CAJAS
from .predictores import ZONA_INSPECCION as ZONA  # misma zona que recorta la CNN
# Linea donde deben quedar los pies (sobre la marca amarilla del piso).
PIES_Y, PIES_X0, PIES_X1 = 0.885, 0.44, 0.63
ASPECTO_CALIBRADO = 16 / 9

ESTILO = {  # nombre en pantalla, color BGR
    "cabeza": ("CABEZA / COFIA", (255, 200, 0)),
    "torso": ("BATA (pecho)", (255, 0, 255)),
    "faldon": ("BATA (cadera)", (0, 255, 0)),
    "manos": ("MANOS", (0, 200, 255)),
}


def _px(caja, w, h):
    x0, y0, x1, y1 = caja
    return int(x0 * w), int(y0 * h), int(x1 * w) - 1, int(y1 * h) - 1


def _rect_punteado(img, p0, p1, color, grosor=2, paso=12):
    (x0, y0), (x1, y1) = p0, p1
    for x in range(x0, x1, paso * 2):
        cv2.line(img, (x, y0), (min(x + paso, x1), y0), color, grosor)
        cv2.line(img, (x, y1), (min(x + paso, x1), y1), color, grosor)
    for y in range(y0, y1, paso * 2):
        cv2.line(img, (x0, y), (x0, min(y + paso, y1)), color, grosor)
        cv2.line(img, (x1, y), (x1, min(y + paso, y1)), color, grosor)


def dibujar_guia(vista, texto_extra=None):
    """Dibuja la plantilla sobre `vista` (pase una COPIA del fotograma) y la devuelve."""
    h, w = vista.shape[:2]
    esc = max(0.45, w / 1280 * 0.6)
    g = max(1, int(round(w / 640)))

    # Oscurecer fuera de la zona de inspeccion
    zx0, zy0, zx1, zy1 = _px(ZONA, w, h)
    sombra = vista.copy()
    sombra[:] = (sombra * 0.45).astype(np.uint8)
    sombra[zy0:zy1, zx0:zx1] = vista[zy0:zy1, zx0:zx1]
    vista[:] = sombra
    _rect_punteado(vista, (zx0, zy0), (zx1, zy1), (255, 255, 255), g)

    # Cuadros de cada region
    for nombre, (etiqueta, color) in ESTILO.items():
        x0, y0, x1, y1 = _px(CAJAS[nombre], w, h)
        cv2.rectangle(vista, (x0, y0), (x1, y1), color, g)
        cv2.putText(vista, etiqueta, (x1 + 6, y0 + int(18 * esc / 0.6)),
                    cv2.FONT_HERSHEY_SIMPLEX, esc, color, g, cv2.LINE_AA)

    # Marca de pies
    py, px0, px1 = int(PIES_Y * h), int(PIES_X0 * w), int(PIES_X1 * w)
    cv2.line(vista, (px0, py), (px1, py), (0, 255, 255), g + 2)
    cv2.putText(vista, "PIES AQUI", (px1 + 6, py + 6), cv2.FONT_HERSHEY_SIMPLEX,
                esc, (0, 255, 255), g, cv2.LINE_AA)

    # Instrucciones
    lineas = ["Una sola persona en la zona clara", "Cabeza completa dentro del cuadro CABEZA",
              "No pasar de la marca de ~1 m"]
    if texto_extra:
        lineas.insert(0, texto_extra)
    for i, t in enumerate(lineas):
        y = int(28 * esc / 0.6) * (i + 1)
        cv2.putText(vista, t, (12, y), cv2.FONT_HERSHEY_SIMPLEX, esc, (0, 0, 0), g + 3, cv2.LINE_AA)
        cv2.putText(vista, t, (12, y), cv2.FONT_HERSHEY_SIMPLEX, esc, (255, 255, 255), g, cv2.LINE_AA)
    return vista


def aviso_aspecto(w, h):
    asp = w / float(h)
    if abs(asp - ASPECTO_CALIBRADO) > 0.05:
        return (f"AVISO: la camara entrega {w}x{h} (aspecto {asp:.2f}); la guia esta calibrada "
                f"para 16:9 (p. ej. 1280x720). Las regiones pueden quedar corridas.")
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fuente", type=int, default=0)
    ap.add_argument("--ancho", type=int, default=1280)
    ap.add_argument("--alto", type=int, default=720)
    ap.add_argument("--imagen", help="probar la guia sobre una foto en vez de la camara")
    ap.add_argument("--salida", help="con --imagen: guardar el resultado aqui en vez de mostrarlo")
    a = ap.parse_args()

    if a.imagen:
        img = cv2.imdecode(np.fromfile(a.imagen, np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            raise SystemExit(f"No se pudo leer {a.imagen}")
        aviso = aviso_aspecto(img.shape[1], img.shape[0])
        if aviso:
            print(aviso)
        vista = dibujar_guia(img.copy())
        if a.salida:
            Path(a.salida).parent.mkdir(parents=True, exist_ok=True)
            ok, buf = cv2.imencode(".jpg", vista)
            Path(a.salida).write_bytes(buf.tobytes())
            print(f"Guardado: {a.salida}")
        else:
            cv2.imshow("Guia de encuadre", vista)
            cv2.waitKey(0)
        return

    cap = cv2.VideoCapture(a.fuente)
    if not cap.isOpened():
        raise SystemExit(f"No se pudo abrir la camara {a.fuente}. Cierre otras apps o pruebe --fuente 1")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, a.ancho)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, a.alto)
    ok, frame = cap.read()
    if not ok:
        raise SystemExit("La camara no entrega imagen.")
    aviso = aviso_aspecto(frame.shape[1], frame.shape[0])
    if aviso:
        print(aviso)
    print(f"Camara: {frame.shape[1]}x{frame.shape[0]} | g guia | s captura | q salir")

    mostrar = True
    destino = Path("reports")
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        vista = dibujar_guia(frame.copy()) if mostrar else frame.copy()
        cv2.imshow("Guia de encuadre (q para salir)", vista)
        k = cv2.waitKey(1) & 0xFF
        if k == ord("q"):
            break
        if k == ord("g"):
            mostrar = not mostrar
        if k == ord("s"):
            destino.mkdir(exist_ok=True)
            ruta = destino / f"guia_{time.strftime('%Y%m%d_%H%M%S')}.jpg"
            ok2, buf = cv2.imencode(".jpg", vista)
            ruta.write_bytes(buf.tobytes())
            print(f"Captura de pantalla: {ruta}")
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
