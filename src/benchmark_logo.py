"""Validacion "dejando una persona fuera" (Leave-One-Group-Out) para los modelos clasicos.

Uso: python -m src.benchmark_logo [--csv data/features.csv] [--sin-mlflow]

Por que: con 6 personas, un unico split (4 train / 2 test) da numeros que cambian mucho segun
que pareja quede en test. Aqui cada persona es el test una vez y se entrena con las demas;
se reporta media +- desviacion entre personas. No reemplaza a benchmark_fase1 (que guarda
models/clasico.joblib para el video en vivo): sirve para reportar resultados confiables.

Salidas: reports/logo_resumen.csv, reports/logo_por_persona.csv,
         reports/cm_logo_<modelo>.png y un run por modelo en MLflow (experimento "fase1_logo").
"""
import argparse
import warnings

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix, f1_score, recall_score
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from .common import DATA, REPORTS, ROOT

CLASE_OK = "completo"


def modelos():
    return {
        "LogisticRegression": make_pipeline(
            StandardScaler(), LogisticRegression(max_iter=3000, class_weight="balanced")),
        "RandomForest": RandomForestClassifier(
            n_estimators=300, class_weight="balanced", random_state=42, n_jobs=-1),
        "SVM_RBF": make_pipeline(
            StandardScaler(), SVC(kernel="rbf", C=1.0, gamma="scale", class_weight="balanced")),
    }


def evaluar(nombre, modelo, X, y, g, clases):
    filas, y_true_all, y_pred_all = [], [], []
    for tr, te in LeaveOneGroupOut().split(X, y, g):
        if len(np.unique(y[tr])) < 2:
            continue
        modelo.fit(X[tr], y[tr])
        pred = modelo.predict(X[te])
        presentes = sorted(np.unique(y[te]))
        fila = {"modelo": nombre, "persona_test": g[te][0], "n_test": len(te),
                "f1_macro": f1_score(y[te], pred, labels=presentes, average="macro", zero_division=0),
                "accuracy": float((pred == y[te]).mean())}
        no_ok = y[te] != CLASE_OK
        fila["falsos_aprobados_tasa"] = float((pred[no_ok] == CLASE_OK).mean()) if no_ok.any() else np.nan
        rec = recall_score(y[te], pred, labels=presentes, average=None, zero_division=0)
        for c in clases:
            fila[f"recall_{c}"] = float(rec[presentes.index(c)]) if c in presentes else np.nan
        filas.append(fila)
        y_true_all.extend(y[te])
        y_pred_all.extend(pred)
    return pd.DataFrame(filas), np.array(y_true_all), np.array(y_pred_all)


def guardar_cm(nombre, y_true, y_pred, clases):
    cm = confusion_matrix(y_true, y_pred, labels=clases, normalize="true")
    fig, ax = plt.subplots(figsize=(6, 5))
    ConfusionMatrixDisplay(cm, display_labels=clases).plot(ax=ax, cmap="Blues", values_format=".2f",
                                                           colorbar=False)
    ax.set_title(f"{nombre} - LOGO (todas las personas)")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    ruta = REPORTS / f"cm_logo_{nombre}.png"
    plt.savefig(ruta, dpi=120)
    plt.close()
    return ruta


def registrar_mlflow(nombre, resumen, ruta_cm, n_personas):
    try:
        import mlflow
        mlflow.set_tracking_uri(f"sqlite:///{(ROOT / 'mlflow.db').as_posix()}")
        mlflow.set_experiment("fase1_logo")
        with mlflow.start_run(run_name=f"{nombre}_LOGO"):
            mlflow.log_params({"modelo": nombre, "validacion": "LeaveOneGroupOut",
                               "n_personas": n_personas})
            mlflow.log_metrics({k: float(v) for k, v in resumen.items()
                                if isinstance(v, (int, float, np.floating)) and np.isfinite(v)})
            mlflow.log_artifact(str(ruta_cm))
    except Exception as e:  # MLflow es opcional: no debe tumbar el benchmark
        print(f"  (MLflow no registro {nombre}: {e})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=str(DATA / "features.csv"))
    ap.add_argument("--sin-mlflow", action="store_true")
    a = ap.parse_args()

    df = pd.read_csv(a.csv)
    df = df[~df.grupo.astype(str).str.startswith(("pub", "sin_grupo"))]
    cols = [c for c in df.columns if c.startswith("f_")]
    X, y, g = df[cols].to_numpy(float), df.clase.to_numpy(), df.grupo.astype(str).to_numpy()
    clases = sorted(np.unique(y))
    personas = sorted(np.unique(g))
    if len(personas) < 3:
        raise SystemExit(f"Hay {len(personas)} personas; LOGO necesita al menos 3.")
    print(f"LOGO con {len(personas)} personas ({', '.join(personas)}), {len(df)} imgs, "
          f"{len(cols)} caracteristicas, clases: {', '.join(clases)}")
    warnings.filterwarnings("ignore", category=UserWarning)

    resumenes, detalle = [], []
    for nombre, modelo in modelos().items():
        por_persona, yt, yp = evaluar(nombre, modelo, X, y, g, clases)
        detalle.append(por_persona)
        metricas = ["f1_macro", "accuracy", "falsos_aprobados_tasa"] + [f"recall_{c}" for c in clases]
        res = {"modelo": nombre}
        for m in metricas:
            res[f"{m}_media"] = por_persona[m].mean()
            res[f"{m}_std"] = por_persona[m].std()
        resumenes.append(res)
        ruta_cm = guardar_cm(nombre, yt, yp, clases)
        if not a.sin_mlflow:
            registrar_mlflow(nombre, res, ruta_cm, len(personas))

    resumen = pd.DataFrame(resumenes)
    por_persona = pd.concat(detalle, ignore_index=True)
    resumen.to_csv(REPORTS / "logo_resumen.csv", index=False)
    por_persona.to_csv(REPORTS / "logo_por_persona.csv", index=False)

    def fmt(m):
        return resumen.apply(lambda r: f"{r[m + '_media']:.3f} +- {r[m + '_std']:.3f}", axis=1)

    tabla = pd.DataFrame({"modelo": resumen.modelo, "F1 macro": fmt("f1_macro"),
                          "accuracy": fmt("accuracy"), "falsos aprobados": fmt("falsos_aprobados_tasa")})
    print("\nResumen (media +- desviacion entre personas):")
    print(tabla.to_string(index=False))
    rec = pd.DataFrame({"modelo": resumen.modelo, **{c: fmt(f"recall_{c}") for c in clases}})
    print("\nRecall por clase:")
    print(rec.to_string(index=False))
    print("\nF1 macro por persona de test:")
    print(por_persona.pivot(index="persona_test", columns="modelo", values="f1_macro").round(3).to_string())
    print(f"\nAzar con {len(clases)} clases: F1 ~ {1 / len(clases):.2f}. "
          "'falsos aprobados' = fraccion de personas SIN EPP completo clasificadas como completo.")
    print("Salidas: reports/logo_resumen.csv, reports/logo_por_persona.csv, reports/cm_logo_*.png")


if __name__ == "__main__":
    main()
