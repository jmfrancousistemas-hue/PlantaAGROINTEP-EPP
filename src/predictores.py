"""Predictores y logica de decision compartidos por el video en vivo y la API.

Cada predictor recibe un fotograma BGR (como lo entrega OpenCV) y devuelve un diccionario
{clase: probabilidad}. Las clases son las del entrenamiento: completo, falta_<epp>, sin_epp.

No importa nada pesado al cargar el modulo: scikit-learn / TensorFlow / requests solo se
importan cuando se crea el backend correspondiente.
"""
import json
import os
from collections import deque
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
MODELS = ROOT / "models"
UMBRAL_DEFECTO = 0.70  # confianza promedio < umbral -> INCIERTO (README)
UMBRAL_APROBAR = 0.80  # para APROBAR el ingreso se exige mas confianza que para denegarlo
IMG = 224
ELEMENTOS_TODOS = ("cofia", "guantes", "bata")

# Zona de inspeccion (x0, y0, x1, y1 como fraccion del cuadro): donde se para el estudiante,
# sobre la marca amarilla. La CNN por elemento ve SOLO esta zona (la persona se ve ~4 veces mas
# grande y la gente del fondo queda casi toda fuera). La misma zona se dibuja en la guia de
# captura (guia_encuadre.py). Si mueve la camara, recalibre aqui y reentrene.
ZONA_INSPECCION = (0.40, 0.06, 0.68, 0.94)


def recortar_zona(img, zona=ZONA_INSPECCION):
    """Recorta la zona de inspeccion de un fotograma completo de la camara."""
    h, w = img.shape[:2]
    x0, y0, x1, y1 = zona
    return img[int(y0 * h):int(y1 * h), int(x0 * w):int(x1 * w)]


def letterbox(img, lado):
    """Lleva la imagen a lado x lado SIN deformarla (rellena con negro), igual que
    tf.image.resize_with_pad en el entrenamiento."""
    h, w = img.shape[:2]
    esc = min(lado / w, lado / h)
    nw, nh = max(1, int(w * esc)), max(1, int(h * esc))
    chica = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    out = np.zeros((lado, lado, 3), dtype=chica.dtype)
    y, x = (lado - nh) // 2, (lado - nw) // 2
    out[y:y + nh, x:x + nw] = chica
    return out


def requeridos_por_defecto():
    """EPP exigidos en esta practica. Se configura con la variable de entorno EPP_REQUERIDOS
    (p. ej. "cofia,bata" cuando la practica no exige guantes). Por defecto: todos."""
    txt = os.getenv("EPP_REQUERIDOS", "")
    req = [e.strip().lower() for e in txt.split(",") if e.strip()]
    return tuple(req) if req else ELEMENTOS_TODOS


def _faltan_en_clase(clase):
    """Elementos que faltan segun el nombre de una clase de estado."""
    if clase == "sin_epp":
        return set(ELEMENTOS_TODOS)
    if clase.startswith("falta_"):
        return set(clase[6:].split("_"))
    return set()


class PredictorClasico:
    """Mejor modelo de ML clasico (models/clasico.joblib, lo guarda benchmark_fase1.py)."""

    tipo = "estado"

    def __init__(self, ruta=None):
        import joblib

        from . import features  # mismas caracteristicas con las que se entreno

        self.features = features
        self.pipe = joblib.load(ruta or MODELS / "clasico.joblib")
        self.clases = [str(c) for c in self.pipe.classes_]

    def predecir(self, frame_bgr):
        x = np.asarray(self.features.extract(frame_bgr), dtype=float).reshape(1, -1)
        p = self.pipe.predict_proba(x)[0]
        return dict(zip(self.clases, (float(v) for v in p)))


class PredictorCNN:
    """MobileNetV2 con Transfer Learning (models/epp_mobilenet.keras + classes.json).
    El reescalado esta dentro del modelo: aqui solo se envian pixeles 0-255 en RGB."""

    tipo = "estado"

    def __init__(self, ruta=None):
        import tensorflow as tf

        self.modelo = tf.keras.models.load_model(ruta or MODELS / "epp_mobilenet.keras")
        self.clases = json.loads((MODELS / "classes.json").read_text(encoding="utf-8"))

    def predecir(self, frame_bgr):
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        x = cv2.resize(rgb, (IMG, IMG), interpolation=cv2.INTER_LINEAR).astype("float32")
        p = self.modelo(x[None], training=False).numpy()[0]
        return dict(zip(self.clases, (float(v) for v in p)))


class PredictorMulti:
    """Modelo MULTI-ETIQUETA (models/epp_multi.keras + elementos.json): evalua cada elemento por
    separado y devuelve {elemento: probabilidad de que este PRESENTE}."""

    tipo = "elementos"

    def __init__(self, ruta=None):
        import tensorflow as tf

        # compile=False: la perdida enmascarada solo se necesita para entrenar
        self.modelo = tf.keras.models.load_model(ruta or MODELS / "epp_multi.keras",
                                                 compile=False)
        meta = json.loads((MODELS / "elementos.json").read_text(encoding="utf-8"))
        self.elementos, self.img = meta["elementos"], meta.get("img", IMG)
        # Modelos nuevos guardan la zona de recorte en elementos.json; los antiguos no
        # (se entrenaron con el cuadro completo deformado a cuadrado) y se usan como antes.
        self.zona = tuple(meta["zona"]) if meta.get("zona") else None

    def predecir(self, frame_bgr):
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        if self.zona:
            x = letterbox(recortar_zona(rgb, self.zona), self.img)
        else:
            x = cv2.resize(rgb, (self.img, self.img), interpolation=cv2.INTER_LINEAR)
        p = self.modelo(x.astype("float32")[None], training=False).numpy()[0]
        return dict(zip(self.elementos, (float(v) for v in p)))


class PredictorAPI:
    """Envia el fotograma a la API desplegada (POST multipart, campo 'archivo').
    El tipo (estado | elementos) lo decide el modelo que corre en la API."""

    tipo = "estado"

    def __init__(self, url, timeout=5.0):
        import requests

        self.requests, self.url, self.timeout = requests, url, timeout

    def predecir(self, frame_bgr):
        ok, buf = cv2.imencode(".jpg", frame_bgr, [cv2.IMWRITE_JPEG_QUALITY, 90])
        if not ok:
            raise RuntimeError("no se pudo codificar el fotograma")
        r = self.requests.post(self.url, files={"archivo": ("frame.jpg", buf.tobytes(),
                                                            "image/jpeg")}, timeout=self.timeout)
        r.raise_for_status()
        j = r.json()
        self.tipo = j.get("tipo", "estado")
        return {k: float(v) for k, v in j["probabilidades"].items()}


def crear_predictor(backend, url=None, modelo=None):
    if backend == "clasico":
        return PredictorClasico(modelo)
    if backend == "cnn":
        return PredictorCNN(modelo)
    if backend == "multi":
        return PredictorMulti(modelo)
    if backend == "api":
        if not url:
            raise ValueError("el backend 'api' requiere --url")
        return PredictorAPI(url)
    raise ValueError(f"backend desconocido: {backend}")


class Suavizador:
    """Promedio temporal de las probabilidades de los ultimos N fotogramas: evita que un
    solo fotograma ruidoso cambie el resultado en pantalla."""

    def __init__(self, ventana=10, minimo=3):
        self.buf, self.minimo = deque(maxlen=ventana), minimo

    def agregar(self, probs):
        self.buf.append(probs)

    def limpiar(self):
        self.buf.clear()

    def __len__(self):
        return len(self.buf)

    def promedio(self):
        """Probabilidades promedio de la ventana, o None si aun hay pocas muestras."""
        if len(self.buf) < self.minimo:
            return None
        return {c: float(np.mean([p.get(c, 0.0) for p in self.buf])) for c in self.buf[0]}

    def resultado(self):
        """(clase, confianza, probabilidades_promedio) para modelos de estado."""
        prom = self.promedio()
        if prom is None:
            return None
        clase = max(prom, key=prom.get)
        return clase, prom[clase], prom


def decidir_estado(probs, umbral=UMBRAL_DEFECTO, umbral_aprobar=UMBRAL_APROBAR, requeridos=None):
    """Decision para modelos de ESTADO teniendo en cuenta solo los EPP exigidos.

    Una clase "cumple" si no le falta ningun elemento requerido (p. ej. con requeridos
    cofia,bata la clase falta_guantes cumple). Se suma la probabilidad de todas las clases que
    cumplen: aprobar exige esa suma >= umbral_aprobar. Devuelve (categoria, texto, clase, conf)."""
    req = set(requeridos or ELEMENTOS_TODOS)
    todos = req >= set(ELEMENTOS_TODOS)
    p_ok = sum(p for c, p in probs.items() if not (_faltan_en_clase(c) & req))
    clase = max(probs, key=probs.get)
    if p_ok >= max(umbral, umbral_aprobar):
        return "aprobado", "INGRESO APROBADO", clase, p_ok
    faltan = _faltan_en_clase(clase) & req
    if not faltan or probs[clase] < umbral:
        return "incierto", "INCIERTO - REVISION MANUAL", clase, max(p_ok, 1 - p_ok)
    if todos and clase == "sin_epp":
        return "denegado", "INGRESO DENEGADO - FALTAN TODOS LOS EPP", clase, probs[clase]
    orden = [e for e in ELEMENTOS_TODOS if e in faltan] + sorted(faltan - set(ELEMENTOS_TODOS))
    return "denegado", "INGRESO DENEGADO - FALTA: " + ", ".join(orden).upper(), clase, probs[clase]


def interpretar(clase, confianza, umbral=UMBRAL_DEFECTO, umbral_aprobar=UMBRAL_APROBAR):
    """(categoria, texto) segun la clase y la confianza.
    Categorias: aprobado | denegado | incierto (este ultimo pide revision manual).

    Aprobar exige mas confianza que denegar: el error grave es dejar entrar a quien no cumple.
    Una clase combinada como `falta_cofia_guantes` se muestra como FALTA: COFIA, GUANTES."""
    if clase == "completo":
        if confianza >= max(umbral, umbral_aprobar):
            return "aprobado", "INGRESO APROBADO"
        return "incierto", "INCIERTO - REVISION MANUAL"
    if confianza < umbral:
        return "incierto", "INCIERTO - REVISION MANUAL"
    if clase == "sin_epp":
        return "denegado", "INGRESO DENEGADO - FALTAN TODOS LOS EPP"
    if clase.startswith("falta_"):
        return "denegado", "INGRESO DENEGADO - FALTA: " + ", ".join(clase[6:].split("_")).upper()
    return "denegado", "INGRESO DENEGADO - " + clase.upper()


def evaluar_elementos(probs, umbral=UMBRAL_DEFECTO, umbral_aprobar=UMBRAL_APROBAR,
                      requeridos=None):
    """Decision a partir de {elemento: P(presente)}. Devuelve (categoria, texto, estados).

    Solo cuentan los elementos REQUERIDOS (por defecto EPP_REQUERIDOS o todos); los demas no
    aparecen en `estados` ni afectan la decision.
    Por elemento: ok si P >= umbral_aprobar | falta si P <= 1 - umbral | incierto en medio.
    Ingreso: aprobado solo si TODOS son ok; denegado si alguno falta (se nombran los que faltan);
    incierto si ninguno falta pero alguno no es seguro (revision manual)."""
    req = set(requeridos or requeridos_por_defecto())
    estados = {}
    for e, p in probs.items():
        if e not in req:
            continue
        estados[e] = "ok" if p >= umbral_aprobar else ("falta" if p <= 1 - umbral else "incierto")
    faltan = [e.upper() for e, s in estados.items() if s == "falta"]
    dudan = [e.upper() for e, s in estados.items() if s == "incierto"]
    if faltan:
        return "denegado", "INGRESO DENEGADO - FALTA: " + ", ".join(faltan), estados
    if dudan:
        return "incierto", "INCIERTO - REVISAR: " + ", ".join(dudan), estados
    return "aprobado", "INGRESO APROBADO", estados


def decidir(tipo, probs, umbral=UMBRAL_DEFECTO, umbral_aprobar=UMBRAL_APROBAR, requeridos=None):
    """Unifica los dos tipos de modelo. Devuelve (categoria, texto, estados, clase, confianza);
    `estados` es {elemento: ok|falta|incierto} (None en modelos de estado) y `clase` usa la
    convencion del proyecto (completo | falta_<e>[_<e2>] | incierto_<e>).
    `requeridos`: EPP exigidos; si es None se usa EPP_REQUERIDOS (o todos)."""
    req = tuple(requeridos) if requeridos else requeridos_por_defecto()
    if tipo == "elementos":
        cat, txt, estados = evaluar_elementos(probs, umbral, umbral_aprobar, req)
        faltan = [e for e, s in estados.items() if s == "falta"]
        dudan = [e for e, s in estados.items() if s == "incierto"]
        clase = ("falta_" + "_".join(faltan)) if faltan else (
            ("incierto_" + "_".join(dudan)) if dudan else "completo")
        usados = [p for e, p in probs.items() if e in estados] or list(probs.values())
        return cat, txt, estados, clase, min(max(p, 1 - p) for p in usados)
    if set(req) >= set(ELEMENTOS_TODOS):  # comportamiento original
        clase = max(probs, key=probs.get)
        cat, txt = interpretar(clase, probs[clase], umbral, umbral_aprobar)
        return cat, txt, None, clase, probs[clase]
    cat, txt, clase, conf = decidir_estado(probs, umbral, umbral_aprobar, req)
    return cat, txt, None, clase, conf
