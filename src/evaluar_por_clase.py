"""Evalua el modelo multietiqueta ya entrenado POR CLASE (estado) y lo registra en MLflow.

No reentrena: carga models/epp_multi.keras + elementos.json y predice las fotos de las
personas de test, con el mismo recorte y las mismas reglas que el video en vivo y la API.

Uso:
  uv run python -m src.evaluar_por_clase                     # test p01,p04 (igual que el entrenamiento)
  uv run python -m src.evaluar_por_clase --test p01,p04 --nombre Final_320_por_clase

Por cada clase (completo, falta_cofia, ..., sin_epp) calcula:
  accuracy           fotos donde el modelo acerto los 3 elementos (presente/falta, umbral 0.5)
  decision_correcta  completo -> APROBADO; las demas -> DENEGADO (con las reglas de umbrales)
  falso_aprobado     fotos de esa clase que el sistema aprobaria sin cumplir
  incierto           fotos enviadas a revision del docente
Salidas: tabla en pantalla, reports/por_clase_multi.csv, reports/por_clase_multi.png y un run
en MLflow (experimento agrointep-fase2) con las metricas accuracy_<clase>, etc.
"""
import argparse
import time
from pathlib import Path

import cv2
import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .common import IMG_EXT, RAW, REPORTS, mlflow_uri
from .predictores import (UMBRAL_APROBAR, UMBRAL_DEFECTO, PredictorMulti, evaluar_elementos)

ORDEN = ["completo", "falta_cofia", "falta_guantes", "falta_bata", "falta_cofia_guantes",
         "falta_cofia_bata", "falta_guantes_bata", "sin_epp"]


def etiquetas(estado, elementos):
    """completo / sin_epp / falta_<e>[_<e2>] -> {elemento: 1 presente | 0 ausente} o None."""
    if estado == "completo":
        return {e: 1 for e in elementos}
    if estado == "sin_epp":
        return {e: 0 for e in elementos}
    if estado.startswith("falta_"):
        faltan = estado[6:].split("_")
        if all(f in elementos for f in faltan):
            return {e: 0 if e in faltan else 1 for e in elementos}
    return None


def estado_predicho(p, elementos):
    """Nombre de clase segun los elementos que el modelo da por ausentes (P < 0.5)."""
    faltan = [e for e in ["cofia", "guantes", "bata"] if e in elementos and p[e] < 0.5]
    if not faltan:
        return "completo"
    if len(faltan) == len(elementos):
        return "sin_epp"
    return "falta_" + "_".join(faltan)


def matriz(d, columna, columnas, titulo, ruta, normalizar=True):
    """Matriz de confusion: filas = clase real, columnas = `columna` (prediccion)."""
    filas = [c for c in ORDEN if c in set(d.clase)]
    m = pd.crosstab(d.clase, d[columna]).reindex(index=filas, columns=columnas, fill_value=0)
    if normalizar:
        m = m.div(m.sum(axis=1), axis=0)
    fig, ax = plt.subplots(figsize=(1.1 * len(columnas) + 3, 0.6 * len(filas) + 2))
    ax.imshow(m.values, cmap="Blues", vmin=0, vmax=1 if normalizar else None)
    for i in range(m.shape[0]):
        for j in range(m.shape[1]):
            v = m.values[i, j]
            ax.text(j, i, f"{v:.2f}" if normalizar else f"{int(v)}", ha="center", va="center",
                    color="white" if (v > 0.5 if normalizar else v > m.values.max() / 2) else "#0b2d5b",
                    fontsize=9)
    ax.set_xticks(range(len(columnas)), columnas, rotation=30, ha="right")
    ax.set_yticks(range(len(filas)), filas)
    ax.set_xlabel("Predicción del sistema")
    ax.set_ylabel("Clase real")
    ax.set_title(titulo)
    plt.tight_layout()
    plt.savefig(ruta, dpi=120)
    plt.close()
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test", default="p01,p04", help="personas de test separadas por coma")
    ap.add_argument("--nombre", default="Evaluacion_por_clase", help="nombre del run en MLflow")
    ap.add_argument("--sin-mlflow", action="store_true")
    a = ap.parse_args()
    personas = {p.strip() for p in a.test.split(",") if p.strip()}

    pred = PredictorMulti()
    elementos = pred.elementos
    req = tuple(elementos)  # se exigen todos los elementos del modelo
    print(f"Modelo: models/epp_multi.keras | elementos {elementos} | img {pred.img} | "
          f"recorte {pred.zona or 'NO'} | test {sorted(personas)}")

    filas = []
    t0 = time.time()
    for cdir in sorted(p for p in RAW.iterdir() if p.is_dir()):
        verdad = etiquetas(cdir.name, elementos)
        if verdad is None:
            continue
        for f in sorted(cdir.iterdir()):
            if f.suffix.lower() not in IMG_EXT or f.name.startswith("._"):
                continue
            if f.stem.split("__")[0] not in personas:
                continue
            img = cv2.imdecode(np.fromfile(str(f), np.uint8), cv2.IMREAD_COLOR)
            if img is None:
                continue
            p = pred.predecir(img)
            acierto = all((p[e] >= 0.5) == bool(verdad[e]) for e in elementos)
            cat = evaluar_elementos(p, UMBRAL_DEFECTO, UMBRAL_APROBAR, req)[0]
            cumple = all(verdad[e] == 1 for e in req)
            filas.append({"clase": cdir.name, "acierto": acierto, "decision": cat,
                          "predicho": estado_predicho(p, elementos),
                          "correcta": cat == ("aprobado" if cumple else "denegado"),
                          "falso_aprobado": (not cumple) and cat == "aprobado"})
    if not filas:
        raise SystemExit(f"No hay fotos de {sorted(personas)} en {RAW}")
    print(f"{len(filas)} fotos evaluadas en {time.time() - t0:.0f} s")

    d = pd.DataFrame(filas)
    tabla = d.groupby("clase").agg(
        imagenes=("acierto", "size"),
        accuracy=("acierto", "mean"),
        decision_correcta=("correcta", "mean"),
        falso_aprobado=("falso_aprobado", "mean"),
        incierto=("decision", lambda s: (s == "incierto").mean()),
    )
    tabla = tabla.reindex([c for c in ORDEN if c in tabla.index] +
                          [c for c in tabla.index if c not in ORDEN])
    total = pd.DataFrame([{"imagenes": len(d), "accuracy": d.acierto.mean(),
                           "decision_correcta": d.correcta.mean(),
                           "falso_aprobado": d.falso_aprobado.mean(),
                           "incierto": (d.decision == "incierto").mean()}], index=["TOTAL"])
    tabla = pd.concat([tabla, total]).round(3)
    tabla.index.name = "clase"

    REPORTS.mkdir(exist_ok=True)
    ruta_csv = REPORTS / "por_clase_multi.csv"
    tabla.to_csv(ruta_csv)

    fig, ax = plt.subplots(figsize=(10, 4.5))
    t = tabla.drop(index="TOTAL")
    x = np.arange(len(t))
    ax.bar(x - 0.2, t.accuracy, 0.4, label="Accuracy (3 elementos)", color="#1f6feb")
    ax.bar(x + 0.2, t.decision_correcta, 0.4, label="Decisión de ingreso correcta", color="#2ea043")
    for i, (v1, v2) in enumerate(zip(t.accuracy, t.decision_correcta)):
        ax.text(i - 0.2, v1 + 0.01, f"{v1:.2f}", ha="center", fontsize=8)
        ax.text(i + 0.2, v2 + 0.01, f"{v2:.2f}", ha="center", fontsize=8)
    ax.set_xticks(x, t.index, rotation=25, ha="right")
    ax.set_ylim(0, 1.1)
    ax.set_ylabel("Fracción de imágenes")
    ax.set_title(f"Resultado por clase - test {', '.join(sorted(personas))}")
    ax.legend(loc="lower right")
    plt.tight_layout()
    ruta_png = REPORTS / "por_clase_multi.png"
    plt.savefig(ruta_png, dpi=120)
    plt.close()

    pruebas = ", ".join(sorted(personas))
    ruta_cm = REPORTS / "cm_multi_por_clase.png"
    m1 = matriz(d, "predicho", [c for c in ORDEN if c != "falta_cofia_guantes_bata"],
                f"Modelo final - matriz de confusión por clase (test {pruebas})", ruta_cm)
    ruta_dec = REPORTS / "cm_multi_decision.png"
    m2 = matriz(d, "decision", ["aprobado", "denegado", "incierto"],
                f"Modelo final - decisión de ingreso por clase (test {pruebas})", ruta_dec)

    pd.set_option("display.width", 160)
    print("\nMatriz de confusion (filas = real, columnas = predicho, proporciones):")
    print(m1.round(2).to_string())
    print("\nDecision de ingreso por clase (proporciones):")
    print(m2.round(2).to_string())
    print("\n" + tabla.to_string())
    print("\naccuracy = acerto los 3 elementos | decision_correcta = completo->APROBADO, resto->DENEGADO")
    print(f"Guardado: {ruta_csv}, {ruta_png}, {ruta_cm} y {ruta_dec}")

    if not a.sin_mlflow:
        import mlflow
        mlflow.set_tracking_uri(mlflow_uri())
        mlflow.set_experiment("agrointep-fase2")
        with mlflow.start_run(run_name=a.nombre):
            mlflow.log_params({"tipo": "evaluacion_por_clase", "img": pred.img,
                               "recorte": pred.zona is not None,
                               "test_personas": ",".join(sorted(personas))})
            for clase, r in tabla.iterrows():
                c = clase.lower()
                mlflow.log_metrics({f"accuracy_{c}": r.accuracy,
                                    f"decision_correcta_{c}": r.decision_correcta,
                                    f"falso_aprobado_{c}": r.falso_aprobado,
                                    f"incierto_{c}": r.incierto})
            mlflow.log_artifact(str(ruta_csv))
            mlflow.log_artifact(str(ruta_png))
            mlflow.log_artifact(str(ruta_cm))
            mlflow.log_artifact(str(ruta_dec))
        print(f"Registrado en MLflow: experimento agrointep-fase2, run '{a.nombre}'")


if __name__ == "__main__":
    main()
