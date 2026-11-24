"""Equilibra el brillo entre sesiones de captura para que la luz no delate la clase.

Problema: las fotos de una sesion salieron mas claras (~148) que las de otra (~121) y las
sesiones no se reparten igual entre clases (p. ej. 'completo' casi solo de la sesion oscura).
El modelo puede aprender "oscuro = completo" en vez de mirar el EPP.

Solucion: en CADA clase, la mitad de las fotos de cada sesion se llevan al brillo de la otra
sesion. Asi todas las clases quedan con fotos de ambos niveles y el brillo deja de servir para
adivinar la clase. Los originales NO se tocan: se escribe una copia.

Uso:
  uv run python -m src.ajustar_brillo                       # dataset/raw -> dataset/raw_ajustado
  uv run python -m src.ajustar_brillo --umbral 135 --fraccion 0.5

Limitacion (decirlo en el informe): solo corrige el brillo global. Otras diferencias entre
sesiones (balance de color, sombras, fondo) siguen ahi; recapturar sigue siendo lo ideal.
"""
import argparse
import hashlib
import shutil
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from .common import RAW, ROOT

EXT = {".jpg", ".jpeg", ".png", ".bmp"}


def leer(p):
    return cv2.imdecode(np.fromfile(str(p), np.uint8), cv2.IMREAD_COLOR)


def escribir(img, p):
    ok, buf = cv2.imencode(p.suffix if p.suffix.lower() in (".jpg", ".jpeg", ".png") else ".jpg", img,
                           [cv2.IMWRITE_JPEG_QUALITY, 95])
    if not ok:
        raise IOError(f"No se pudo codificar {p}")
    p.write_bytes(buf.tobytes())


def brillo(img):
    return float(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).mean())


def llevar_a(img, objetivo):
    """Escala la luminancia (canal L de LAB) para que el brillo medio quede en `objetivo`."""
    actual = max(brillo(img), 1.0)
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)
    lab[..., 0] = np.clip(lab[..., 0] * (objetivo / actual), 0, 255)
    out = cv2.cvtColor(lab.astype(np.uint8), cv2.COLOR_LAB2BGR)
    # una segunda correccion fina (la conversion LAB no es exactamente lineal en gris)
    b = brillo(out)
    if b > 1:
        out = cv2.convertScaleAbs(out, alpha=objetivo / b, beta=0)
    return out


def elegido(nombre, fraccion):
    """Seleccion reproducible (misma foto -> misma decision en cada corrida)."""
    h = int(hashlib.md5(nombre.encode("utf-8")).hexdigest()[:8], 16)
    return (h % 10000) / 10000.0 < fraccion


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default=str(RAW))
    ap.add_argument("--salida", default=str(ROOT / "dataset" / "raw_ajustado"))
    ap.add_argument("--umbral", type=float, default=135.0,
                    help="brillo que separa sesion oscura (<) y clara (>=)")
    ap.add_argument("--fraccion", type=float, default=0.5,
                    help="fraccion de cada sesion que se lleva al nivel de la otra")
    a = ap.parse_args()

    raw, salida = Path(a.raw), Path(a.salida)
    if salida.resolve() == raw.resolve():
        raise SystemExit("--salida no puede ser la misma carpeta que --raw (los originales no se tocan).")
    fotos = [f for f in sorted(raw.rglob("*")) if f.is_file() and f.suffix.lower() in EXT
             and not f.name.startswith("._") and f.parent.name != "prueba"]
    if not fotos:
        raise SystemExit(f"No hay fotos en {raw}")

    print(f"Leyendo {len(fotos)} fotos de {raw} ...")
    filas = []
    for f in fotos:
        img = leer(f)
        if img is None:
            continue
        filas.append({"ruta": f, "clase": f.parent.name, "brillo": brillo(img)})
    d = pd.DataFrame(filas)
    oscura, clara = d[d.brillo < a.umbral], d[d.brillo >= a.umbral]
    if oscura.empty or clara.empty:
        raise SystemExit(f"No se ven dos sesiones con --umbral {a.umbral} "
                         f"(brillo min {d.brillo.min():.0f}, max {d.brillo.max():.0f}). Ajuste --umbral.")
    nivel_oscuro, nivel_claro = oscura.brillo.median(), clara.brillo.median()
    print(f"Sesion oscura: {len(oscura)} fotos (mediana {nivel_oscuro:.0f}) | "
          f"sesion clara: {len(clara)} fotos (mediana {nivel_claro:.0f})")

    if salida.exists():
        shutil.rmtree(salida)
    cambiadas = 0
    nuevos = []
    for r in d.itertuples():
        dst = salida / r.ruta.relative_to(raw)
        dst.parent.mkdir(parents=True, exist_ok=True)
        es_clara = r.brillo >= a.umbral
        if elegido(r.ruta.name, a.fraccion):
            img = llevar_a(leer(r.ruta), nivel_oscuro if es_clara else nivel_claro)
            escribir(img, dst)
            nuevos.append(brillo(img))
            cambiadas += 1
        else:
            shutil.copy2(r.ruta, dst)
            nuevos.append(r.brillo)
    d["brillo_nuevo"] = nuevos

    resumen = d.groupby("clase").agg(
        fotos=("brillo", "size"),
        clara_antes=("brillo", lambda s: (s >= a.umbral).mean()),
        clara_despues=("brillo_nuevo", lambda s: (s >= a.umbral).mean()),
        brillo_antes=("brillo", "mean"),
        brillo_despues=("brillo_nuevo", "mean"),
    ).round(2)
    print(f"\nModificadas {cambiadas} de {len(d)} fotos. Fraccion de fotos 'claras' por clase:")
    print(resumen.to_string())
    print(f"\nCopia lista en {salida}. Los originales en {raw} no se modificaron.")
    print("Siguiente: uv run python -m src.eda_limpieza --raw " + salida.relative_to(ROOT).as_posix())


if __name__ == "__main__":
    main()
