"""Descriptores clasicos de OpenCV (Fase 1): vector numerico por imagen.

extract(img_bgr) -> lista de floats, siempre del mismo largo (N_FEATURES). Se usa igual al
entrenar (eda_limpieza.py) y en vivo (PredictorClasico): si cambia esta funcion, vuelva a
correr eda_limpieza y benchmark_fase1 para regenerar features.csv y clasico.joblib.

Version 3 (regiones fijas calibradas al encuadre de la camara).
  - v1 miraba la imagen completa reducida a 128x128: la cofia quedaba en pocos pixeles y el
    vector describia sobre todo ropa, fondo y otras personas al fondo.
  - v2 ubicaba las regiones a partir del rostro, pero el tapabocas impide detectar el rostro
    con Haar, y el detector de personas HOG elegia a la gente del fondo.
  - v3 aprovecha que la camara es fija y el estudiante se para sobre la marca amarilla:
    las regiones son fracciones fijas del cuadro (funcionan con cualquier resolucion).

Regiones (x0, y0, x1, y1 como fraccion del ancho/alto del cuadro):
  cabeza -> parte alta de la cabeza: cofia vs. cabello
  torso  -> pecho y abdomen: bata (azul o blanca) vs. ropa de calle
  faldon -> cadera y muslos: la bata llega hasta aqui; una camiseta blanca no
  manos  -> zona de las manos a los lados y frente al cuerpo: guantes vs. piel

Si mueve la camara o la marca del piso, recalibre CAJAS revisando reports/regiones/
(python -m src.features dataset/limpio/completo ...). Largo total: 4 x 40 + 2 = 162.
"""
import cv2
import numpy as np

CAJAS = {
    "cabeza": (0.475, 0.11, 0.585, 0.25),
    "torso": (0.465, 0.30, 0.605, 0.54),
    "faldon": (0.470, 0.52, 0.600, 0.68),
    "manos": (0.430, 0.42, 0.640, 0.62),
}
REGIONES = tuple(CAJAS)
H_BINS, S_BINS, V_BINS = 8, 2, 2
N_POR_REGION = H_BINS * S_BINS * V_BINS + 8
N_GLOBALES = 2
N_FEATURES = len(REGIONES) * N_POR_REGION + N_GLOBALES
LADO_ROI = 96


def _recorte(img, caja):
    h, w = img.shape[:2]
    x0, y0, x1, y1 = caja
    a, b, c, d = int(x0 * w), int(y0 * h), int(x1 * w), int(y1 * h)
    if c - a < 8 or d - b < 8:
        return None
    return img[b:d, a:c]


def _describir(roi, v_ref):
    """40 valores por region."""
    if roi is None:
        return [0.0] * N_POR_REGION
    roi = cv2.resize(roi, (LADO_ROI, LADO_ROI), interpolation=cv2.INTER_AREA)
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    ycc = cv2.cvtColor(roi, cv2.COLOR_BGR2YCrCb)
    gris = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    s, v = hsv[..., 1].astype(float), hsv[..., 2].astype(float)

    hist = cv2.calcHist([hsv], [0, 1, 2], None, [H_BINS, S_BINS, V_BINS],
                        [0, 180, 0, 256, 0, 256]).flatten()
    hist = hist / max(hist.sum(), 1.0)

    cr, cb = ycc[..., 1], ycc[..., 2]
    piel = float(((cr >= 135) & (cr <= 180) & (cb >= 85) & (cb <= 135)).mean())
    claro = float(((s < 60) & (v > 1.10 * v_ref)).mean())        # blanco/claro (bata, cofia)
    azul = float(((hsv[..., 0] >= 90) & (hsv[..., 0] <= 125) & (s >= 50)).mean())  # bata/cofia azul
    oscuro = float((v < 0.45 * v_ref).mean())                     # cabello, pantalon oscuro
    bordes = float((cv2.Canny(gris, 60, 140) > 0).mean())
    textura = float(np.log1p(cv2.Laplacian(gris, cv2.CV_64F).var()) / 10.0)
    extra = [piel, claro, azul, oscuro, bordes, textura,
             float(s.mean() / 255.0), float(v.mean() / max(v_ref, 1.0))]
    return list(hist) + extra


def extract(img_bgr):
    hsv_total = cv2.cvtColor(cv2.resize(img_bgr, (160, 90)), cv2.COLOR_BGR2HSV)
    v_ref = max(float(np.median(hsv_total[..., 2])), 30.0)   # luz general de la escena
    vec = []
    for nombre in REGIONES:
        vec.extend(_describir(_recorte(img_bgr, CAJAS[nombre]), v_ref))
    vec.extend([v_ref / 255.0, float(hsv_total[..., 1].mean() / 255.0)])
    vec = np.asarray(vec, dtype=float)
    assert vec.size == N_FEATURES, (vec.size, N_FEATURES)
    return vec.tolist()


COLORES = {"cabeza": (255, 200, 0), "torso": (255, 0, 255),
           "faldon": (0, 255, 0), "manos": (0, 200, 255)}


def regiones_debug(img_bgr):
    """Dibuja las regiones sobre una copia de la imagen (para revisar el encuadre)."""
    out = img_bgr.copy()
    h, w = out.shape[:2]
    for nombre, (x0, y0, x1, y1) in CAJAS.items():
        p0, p1 = (int(x0 * w), int(y0 * h)), (int(x1 * w) - 1, int(y1 * h) - 1)
        cv2.rectangle(out, p0, p1, COLORES[nombre], 2)
        cv2.putText(out, nombre, (p0[0] + 3, p0[1] + 16), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, COLORES[nombre], 2)
    return out


if __name__ == "__main__":
    # Revisar regiones: python -m src.features dataset/limpio/completo dataset/limpio/sin_epp
    import sys
    from pathlib import Path

    if len(sys.argv) < 2:
        raise SystemExit("Uso: python -m src.features <foto.jpg o carpeta> [...]")
    destino = Path("reports") / "regiones"
    destino.mkdir(parents=True, exist_ok=True)
    rutas = []
    for arg in sys.argv[1:]:
        p = Path(arg)
        if p.is_dir():
            fotos = sorted(f for f in p.iterdir() if f.suffix.lower() in (".jpg", ".jpeg", ".png")
                           and not f.name.startswith("._"))
            rutas += fotos[:: max(1, len(fotos) // 4)][:4]   # 4 fotos repartidas por carpeta
        elif p.is_file():
            rutas.append(p)
        else:
            print(f"No existe: {arg}")
    for ruta in rutas:
        img = cv2.imdecode(np.fromfile(str(ruta), np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            print(f"No se pudo leer {ruta}")
            continue
        ok, buf = cv2.imencode(".jpg", regiones_debug(img))
        salida = destino / f"{Path(ruta).parent.name}_{Path(ruta).stem}.jpg"
        salida.write_bytes(buf.tobytes())
        print(f"{ruta} -> {salida}")