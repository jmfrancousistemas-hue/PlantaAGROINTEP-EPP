"""EDA + pipeline de limpieza + extraccion de caracteristicas.

Uso: python -m src.eda_limpieza [--raw dataset/raw]
Entrada : dataset/raw/<clase>/*.jpg
Salidas : dataset/limpio/<clase>/, data/features.csv, reports/eda.png,
          reports/inventario_completo.csv

Tambien informa (sin procesarlas) las imagenes publicas de
dataset/publico/<elemento>/<si|no>/, que solo se usan para entrenar en la Fase 2.
"""
import argparse
import hashlib
import shutil
from pathlib import Path

import cv2
import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from . import features
from .common import CLEAN, DATA, IMG_EXT, RAW, REPORTS, ROOT

PUBLICO = ROOT / "dataset" / "publico"
MAX_CLASES_CORTE1 = 4


def es_imagen(f):
    """Imagen valida: extension conocida y no es un archivo oculto de macOS (._*)."""
    return f.is_file() and f.suffix.lower() in IMG_EXT and not f.name.startswith("._")


def leer_imagen(data):
    """Decodifica bytes a BGR de 3 canales (acepta gris y PNG con transparencia)."""
    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_UNCHANGED)
    if img is None:
        return None
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    elif img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    if img.dtype != np.uint8:
        img = cv2.convertScaleAbs(img, alpha=255.0 / max(float(img.max()), 1.0))
    return img


def guardar_jpg(img, dst):
    """cv2.imwrite falla en silencio con rutas con tildes en Windows; imencode no."""
    ok, buf = cv2.imencode(".jpg", img)
    if not ok:
        raise IOError(f"No se pudo codificar {dst}")
    dst.write_bytes(buf.tobytes())


def inventariar(raw, min_lado, umbral_blur):
    if not raw.exists():
        raise SystemExit(f"No existe la carpeta {raw}. Capture primero el dataset (paso 4).")
    filas, vistos = [], set()
    clases = sorted(p for p in raw.iterdir() if p.is_dir() and not p.name.startswith((".", "_")))
    if not clases:
        raise SystemExit(f"No hay carpetas de clase en {raw}")
    for cdir in clases:
        for f in sorted(cdir.rglob("*")):
            if not es_imagen(f):
                continue
            fila = dict(path=str(f), clase=cdir.name, motivo="ok",
                        grupo=f.stem.split("__")[0] if "__" in f.stem else None)
            try:
                data = f.read_bytes()
            except OSError:
                fila["motivo"] = "ilegible"
                filas.append(fila)
                continue
            img = leer_imagen(data)
            if img is None:
                fila["motivo"] = "corrupta"
                filas.append(fila)
                continue
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            h, w = gray.shape
            md5 = hashlib.md5(data).hexdigest()
            fila.update(alto=h, ancho=w, brillo=float(gray.mean()),
                        nitidez=float(cv2.Laplacian(gray, cv2.CV_64F).var()))
            if md5 in vistos:
                fila["motivo"] = "duplicada"
            elif min(h, w) < min_lado:
                fila["motivo"] = "muy_pequena"
            elif fila["nitidez"] < umbral_blur:
                fila["motivo"] = "borrosa"
            elif fila["brillo"] < 25:
                fila["motivo"] = "muy_oscura"
            elif fila["brillo"] > 240:
                fila["motivo"] = "sobreexpuesta"
            vistos.add(md5)
            filas.append(fila)
    inv = pd.DataFrame(filas)
    if inv.empty:
        raise SystemExit(f"No se encontraron imagenes en {raw}")
    return inv


def contar_publicos():
    """Cuenta dataset/publico/<elemento>/<si|no>/ (no se limpian ni van a features.csv)."""
    conteo = {}
    if PUBLICO.exists():
        for e in sorted(p for p in PUBLICO.iterdir() if p.is_dir()):
            for s in sorted(p for p in e.iterdir() if p.is_dir()):
                conteo[f"{e.name}/{s.name}"] = sum(1 for f in s.iterdir() if es_imagen(f))
    return conteo


def graficar(inv, limpio):
    fig, ax = plt.subplots(1, 4, figsize=(18, 4))
    inv.groupby("clase").size().plot.bar(ax=ax[0], title="Imagenes por clase (antes)")
    if len(limpio):
        limpio.groupby("clase").size().plot.bar(ax=ax[1], color="seagreen",
                                                title="Imagenes por clase (despues)")
    else:
        ax[1].set_title("Imagenes por clase (despues): ninguna")
    con_brillo = inv.dropna(subset=["brillo"]) if "brillo" in inv else inv.iloc[0:0]
    if len(con_brillo):
        con_brillo.boxplot(column="brillo", by="clase", ax=ax[2])
    ax[2].set_title("Brillo por clase")
    desc = inv[inv.motivo != "ok"].groupby("motivo").size()
    if len(desc):
        desc.plot.barh(ax=ax[3], color="indianred")
        ax[3].set_title("Descartadas por motivo")
    else:
        ax[3].axis("off")
        ax[3].text(0.5, 0.5, "Sin descartes", ha="center", va="center", fontsize=14)
        ax[3].set_title("Descartadas por motivo")
    plt.suptitle("")
    plt.tight_layout()
    plt.savefig(REPORTS / "eda.png", dpi=120)
    plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default=str(RAW))
    ap.add_argument("--min-lado", type=int, default=64)
    ap.add_argument("--umbral-blur", type=float, default=20.0)
    a = ap.parse_args()

    for d in (REPORTS, DATA):
        d.mkdir(parents=True, exist_ok=True)
    raw = Path(a.raw)

    inv = inventariar(raw, a.min_lado, a.umbral_blur)
    inv.to_csv(REPORTS / "inventario_completo.csv", index=False)
    limpio = inv[inv.motivo == "ok"].copy()
    if limpio.empty:
        graficar(inv, limpio)
        raise SystemExit("Todas las imagenes fueron descartadas. Revise reports/inventario_completo.csv")

    if CLEAN.exists():
        shutil.rmtree(CLEAN)
    filas = []
    for i, r in enumerate(limpio.itertuples()):
        dst = CLEAN / r.clase / f"{i:05d}.jpg"
        dst.parent.mkdir(parents=True, exist_ok=True)
        img = leer_imagen(Path(r.path).read_bytes())
        guardar_jpg(img, dst)
        g = r.grupo if isinstance(r.grupo, str) and r.grupo else f"sin_grupo_{i}"
        fila = {"path": dst.relative_to(ROOT).as_posix(), "clase": r.clase, "grupo": g}
        fila.update({f"f_{k}": v for k, v in enumerate(features.extract(img))})
        filas.append(fila)
    fd = pd.DataFrame(filas)
    fd.to_csv(DATA / "features.csv", index=False)
    graficar(inv, limpio)

    print(f"Origen: {raw}")
    print(f"Total: {len(inv)} | validas: {len(limpio)} | descartadas: {len(inv) - len(limpio)}")
    desc = inv[inv.motivo != "ok"].motivo.value_counts()
    print(desc.to_string() if len(desc) else "Sin descartes")

    conteo = limpio.clase.value_counts()
    print("\nImagenes por clase (limpias):\n" + conteo.to_string())
    if len(conteo) > MAX_CLASES_CORTE1:
        print(f"NOTA: {len(conteo)} clases. Para el primer corte (max. 3-4 clases) use "
              "--raw dataset/raw_corte1; para el modelo final esto es lo esperado.")
    if len(conteo) > 1 and conteo.max() / conteo.min() > 3:
        print("AVISO: desbalance > 3:1; considera mas imagenes de la clase minoritaria.")

    es_pub = fd.grupo.str.startswith("pub")
    es_sin = fd.grupo.str.startswith("sin_grupo")
    propios = fd[~es_pub & ~es_sin]
    print(f"\nPersonas/sesiones propias: {propios.grupo.nunique()} | publicas en raw: "
          f"{int(es_pub.sum())} imgs | sin grupo: {int(es_sin.sum())} imgs")
    if es_sin.any():
        print("AVISO: hay imagenes sin codigo de persona (nombre sin '__'); no se pueden "
              "separar por persona y pueden causar fuga de datos.")
    if propios.grupo.nunique() < 4:
        print("AVISO: menos de 4 personas propias; el test por persona sera poco confiable (ideal >= 6).")

    if len(propios):
        tabla = pd.crosstab(propios.grupo, propios.clase)
        print("\nImagenes por persona y clase:\n" + tabla.to_string())
        por_clase = (tabla > 0).sum()
        if por_clase.min() < 3:
            print("AVISO: alguna clase tiene menos de 3 personas distintas:\n" + por_clase.to_string())
        incompletas = tabla.index[(tabla == 0).any(axis=1)].tolist()
        if incompletas:
            print("AVISO: personas sin todas las clases (complete sus tandas): " + ", ".join(incompletas))

    pub = contar_publicos()
    if pub:
        print("\nPublicas por elemento (dataset/publico, solo entrenamiento Fase 2): "
              + ", ".join(f"{k}={v}" for k, v in pub.items()))
        vacias = [k for k, v in pub.items() if v == 0]
        if vacias:
            print("AVISO: carpetas publicas vacias: " + ", ".join(vacias))
    else:
        print("\nPublicas por elemento: ninguna (paso 6 opcional no ejecutado)")

    print("\nCaracteristicas: data/features.csv | Grafica: reports/eda.png")


if __name__ == "__main__":
    main()
