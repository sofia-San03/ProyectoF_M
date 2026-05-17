import os
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "credenciales.json"
)

import json
import re
from io import StringIO

import google.generativeai as genai
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

from google.cloud import storage, bigquery
import requests
import time
import google.auth
import google.auth.transport.requests
import google.oauth2.id_token

# --- CONFIGURACIÓN GCP ---
GCS_BUCKET = "archivos_back"
PROJECT    = "project-6d52cafa-4432-4186-aeb"
DATASET    = "Cubo"
CF_URL     = "https://armar-cubo-697875837946.northamerica-south1.run.app"

def subir_a_gcs(archivo, carpeta):
    client = get_storage_client()  # no storage.Client() directo
    bucket = client.bucket(GCS_BUCKET)
    blob = bucket.blob(f"{carpeta}/{archivo.name}")
    blob.upload_from_file(archivo, rewind=True)

def tabla_existe_en_bq(tabla_id):
    client = get_bq_client()  # no bigquery.Client() directo
    try:
        client.get_table(f"{PROJECT}.{DATASET}.{tabla_id}")
        return True
    except Exception:
        return False

def leer_cubo_de_bq():
    client = get_bq_client()
    query = f"SELECT * FROM `{PROJECT}.{DATASET}.cubo_analitico`"
    job_config = bigquery.QueryJobConfig()
    return client.query(
        query,
        job_config=job_config,
        location="northamerica-south1"
    ).to_dataframe()

def esperar_tablas_bq(nombres_tablas, timeout=120, intervalo=5):
    inicio = time.time()
    while time.time() - inicio < timeout:
        if all(tabla_existe_en_bq(t) for t in nombres_tablas):
            return True
        time.sleep(intervalo)
    return False

def llamar_build_cubo():
    try:
        import google.auth
        import google.auth.transport.requests

        credentials, project = google.auth.default(
            scopes=["https://www.googleapis.com/auth/cloud-platform"]
        )
        auth_req = google.auth.transport.requests.Request()
        credentials.refresh(auth_req)
        
        response = requests.post(
            CF_URL,
            headers={
                "Authorization": f"Bearer {credentials.token}",
                "Content-Type": "application/json"
            }
        )
        return response.status_code == 200, response.json()
    except Exception as e:
        return False, str(e)

def get_bq_client():
    return bigquery.Client(
        project=PROJECT,
        location="northamerica-south1"
    )

def get_storage_client():
    return storage.Client(project=PROJECT)

from CargarDatos import AnalizarDatos
from CleanData import Transformar_Df
from MODELS import (
    Arbol_decision,
    Clustering_optimizacion,
    Credit_scoring,
    KNN_model,
    Redes_neuronales,
    Regresion_lineal,
    Regresion_logistica,
)


MODELOS_DISPONIBLES = {
    "Regresion_lineal": {
        "funcion": Regresion_lineal,
        "implementado": True,
        "tipo_problema": "regresion",
        "metricas": ["R2", "MSE", "RMSE", "MAE", "MAPE"],
    },
    "Arbol_decision": {
        "funcion": Arbol_decision,
        "implementado": True,
        "tipo_problema": "supervisado",
        "metricas": ["Accuracy/R2", "Precision/RMSE", "Recall/MAE", "F1", "ROC-AUC"],
    },
    "Regresion_logistica": {
        "funcion": Regresion_logistica,
        "implementado": True,
        "tipo_problema": "clasificacion_binaria",
        "metricas": ["Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC"],
    },
    "Clustering_optimizacion": {
        "funcion": Clustering_optimizacion,
        "implementado": True,
        "tipo_problema": "clustering",
        "metricas": [
            "Silhouette Score",
            "Davies-Bouldin Index",
            "Inercia",
            "Distancia Euclidiana Media",
            "Distancia Manhattan Media",
        ],
    },
    "Redes_neuronales": {
        "funcion": Redes_neuronales,
        "implementado": True,
        "tipo_problema": "supervisado",
        "metricas": ["Accuracy/R2", "Precision/RMSE", "Recall/MAE", "F1", "ROC-AUC", "Loss"],
    },
    "KNN": {
        "funcion": KNN_model,
        "implementado": True,
        "tipo_problema": "supervisado",
        "metricas": ["Accuracy/R2", "Precision/RMSE", "Recall/MAE", "F1", "Error vs K"],
    },
    "Credit_scoring": {
        "funcion": Credit_scoring,
        "implementado": True,
        "tipo_problema": "clasificacion_binaria",
        "metricas": ["ROC-AUC", "KS", "Gini", "Precision", "Recall"],
    },
}

ALIAS_MODELOS = {
    "regresion_lineal": "Regresion_lineal",
    "regresión_lineal": "Regresion_lineal",
    "lineal": "Regresion_lineal",
    "linear": "Regresion_lineal",
    "arbol_decision": "Arbol_decision",
    "árbol_decisión": "Arbol_decision",
    "arbol": "Arbol_decision",
    "árbol": "Arbol_decision",
    "decision_tree": "Arbol_decision",
    "regresion_logistica": "Regresion_logistica",
    "regresión_logística": "Regresion_logistica",
    "logistica": "Regresion_logistica",
    "logística": "Regresion_logistica",
    "clustering": "Clustering_optimizacion",
    "clustering_optimizacion": "Clustering_optimizacion",
    "cluster": "Clustering_optimizacion",
    "redes_neuronales": "Redes_neuronales",
    "red_neuronal": "Redes_neuronales",
    "neural_network": "Redes_neuronales",
    "knn": "KNN",
    "k_vecinos_mas_cercanos": "KNN",
    "k_vecinos_más_cercanos": "KNN",
    "k vecinos mas cercanos": "KNN",
    "k vecinos más cercanos": "KNN",
    "credit_scoring": "Credit_scoring",
    "credit scoring": "Credit_scoring",
    "scoring": "Credit_scoring",
}


st.set_page_config(page_title="Autopilot", page_icon="⚡", layout="wide")

st.markdown(
    """
<style>
    .stApp {
        background: radial-gradient(circle at top right, #1e293b, #0f172a);
        color: #e2e8f0;
    }

    [data-testid="stVerticalBlock"] > div:has(.stMarkdown):not(:has(style)) {
        background: rgba(30, 41, 59, 0.5);
        backdrop-filter: blur(10px);
        border-radius: 20px;
        padding: 25px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
        margin-bottom: 20px;
    }

    h1 {
        background: linear-gradient(90deg, #10b981, #3b82f6);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800 !important;
        letter-spacing: -1px;
    }

    h3 { color: #10b981 !important; font-weight: 600 !important; }

    .stButton>button {
        background: linear-gradient(135deg, #059669 0%, #10b981 100%) !important;
        color: white !important;
        border: none !important;
        border-radius: 12px !important;
        font-weight: 700 !important;
        padding: 0.75rem 1.5rem !important;
        transition: all 0.3s ease !important;
        box-shadow: 0 4px 15px rgba(16, 185, 129, 0.3) !important;
        text-transform: uppercase;
        letter-spacing: 1px;
    }

    .stButton>button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 6px 20px rgba(16, 185, 129, 0.5) !important;
        background: linear-gradient(135deg, #10b981 0%, #34d399 100%) !important;
    }

    .stTable {
        border-radius: 12px;
        overflow: hidden;
        border: 1px solid rgba(255, 255, 255, 0.1);
    }

    thead tr th {
        background-color: #1e293b !important;
        color: #10b981 !important;
    }

    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
        background-color: transparent;
    }

    .stTabs [data-baseweb="tab"] {
        background-color: #1e293b;
        border-radius: 10px 10px 0 0;
        padding: 10px 20px;
        color: #94a3b8;
    }

    .stTabs [aria-selected="true"] {
        background-color: #10b981 !important;
        color: white !important;
    }

    .stMarkdown p, .stMarkdown li, .stTable, .stTextArea textarea, .stMarkdown span, .stMarkdown div {
        font-size: 1.25rem !important;
        line-height: 1.6 !important;
    }

    h1 { font-size: 2.8rem !important; }
    h2 { font-size: 2.2rem !important; }
    h3 { font-size: 1.8rem !important; }

    .equal-height-box {
        height: 600px;
        overflow-y: auto;
        padding: 25px;
        background: rgba(30, 41, 59, 0.3);
        border-radius: 20px;
        border: 1px solid rgba(255, 255, 255, 0.05);
        scrollbar-width: thin;
        scrollbar-color: #10b981 #1e293b;
    }

    [data-testid="stText"] {
        font-size: 1.5rem !important;
        line-height: 1.4 !important;
        color: #94a3b8;
    }
</style>
""",
    unsafe_allow_html=True,
)


api_key = None
try:
    if "GEMINI_API_KEY" in st.secrets:
        api_key = st.secrets["GEMINI_API_KEY"]
except Exception:
    pass

if not api_key and os.path.exists("GEMINI_KEY.txt"):
    try:
        with open("GEMINI_KEY.txt", "r", encoding="utf-8") as f:
            api_key = f.read().strip()
    except Exception:
        pass

if api_key:
    genai.configure(api_key=api_key)
else:
    st.error("No se encontro una API key valida. Configura GEMINI_API_KEY o GEMINI_KEY.txt.")
    st.stop()

guia_tecnica = ""
if os.path.exists("GUIA_TECNICA_IA.txt"):
    with open("GUIA_TECNICA_IA.txt", "r", encoding="utf-8") as f:
        guia_tecnica = f.read()

model_ia = genai.GenerativeModel("gemini-2.5-flash")


def normalizar_modelo(tipo_modelo):
    if not tipo_modelo:
        raise ValueError("No se recibio un modelo en la configuracion.")

    modelo_limpio = str(tipo_modelo).strip()
    if modelo_limpio in MODELOS_DISPONIBLES:
        return modelo_limpio

    clave = modelo_limpio.lower().replace("-", "_").replace(" ", "_")
    if clave in ALIAS_MODELOS:
        return ALIAS_MODELOS[clave]

    raise ValueError(
        f"Modelo no reconocido: '{tipo_modelo}'. Modelos disponibles: {', '.join(MODELOS_DISPONIBLES.keys())}"
    )


def extraer_configuracion_pipeline(conf):
    target = conf.get("col_target", conf.get("target"))
    reglas = conf.get("metodos_imputacion", conf.get("reglas_dict", {}))
    modelo_t = conf.get("tipo_modelo", conf.get("modelo", "Regresion_lineal"))
    modelo_normalizado = normalizar_modelo(modelo_t)
    es_pca = conf.get("EsPCA", conf.get("es_pca", False))
    if MODELOS_DISPONIBLES[modelo_normalizado]["tipo_problema"] == "clustering":
        target = None
    return target, reglas, modelo_normalizado, es_pca


def obtener_metricas_esperadas(modelo_nombre):
    return MODELOS_DISPONIBLES[modelo_nombre]["metricas"]


def ocultar_woe_interfaz(valor):
    if isinstance(valor, dict):
        return {
            clave: ocultar_woe_interfaz(contenido)
            for clave, contenido in valor.items()
            if "woe" not in str(clave).lower()
        }
    if isinstance(valor, list):
        return [ocultar_woe_interfaz(item) for item in valor]
    return valor


def validar_tipo_problema(df, target, modelo_nombre):
    info_modelo = MODELOS_DISPONIBLES[modelo_nombre]
    tipo = info_modelo["tipo_problema"]

    if tipo == "clustering":
        return True, "No supervisado: no requiere variable objetivo."

    if not target:
        return False, "Este modelo requiere una variable objetivo."

    if target not in df.columns:
        return False, f"La variable objetivo '{target}' no existe en el dataset."

    y = df[target].dropna()
    clases = y.nunique()

    if tipo == "regresion":
        if pd.api.types.is_numeric_dtype(y):
            return True, "Regresion supervisada con target numerico."
        return False, "Regresion lineal requiere una variable objetivo numerica."

    if tipo == "clasificacion_binaria":
        if clases == 2:
            return True, "Clasificacion binaria valida."
        return False, f"Este modelo requiere exactamente 2 clases en el target; se detectaron {clases}."

    if tipo == "supervisado":
        return True, f"Modelo supervisado con {clases} valores unicos en el target."

    return True, "Tipo de problema validado."


def aplicar_limpieza_interna(df, col_target, reglas_dict=None, es_pca=False):
    transformador = Transformar_Df(df, col_target=col_target)
    transformador.Clean_All_Rows(reglas_dict=reglas_dict, EsPCA=es_pca)
    return transformador, transformador.df, transformador.y


def orquestador_modelos_interno(X, y, tipo_modelo):
    modelo_nombre = normalizar_modelo(tipo_modelo)
    info_modelo = MODELOS_DISPONIBLES[modelo_nombre]

    if not info_modelo["implementado"]:
        raise NotImplementedError(
            f"El modelo '{modelo_nombre}' esta reconocido por la arquitectura, pero no esta implementado."
        )

    if y is None and info_modelo["tipo_problema"] != "clustering":
        raise ValueError(f"El modelo '{modelo_nombre}' requiere variable objetivo.")

    json_res, modelo, cols = info_modelo["funcion"](X, y)
    return modelo, json.loads(json_res), cols


def es_modelo_clustering(tipo_modelo):
    return MODELOS_DISPONIBLES[normalizar_modelo(tipo_modelo)]["tipo_problema"] == "clustering"


def es_modelo_redes_neuronales(tipo_modelo):
    return normalizar_modelo(tipo_modelo) == "Redes_neuronales"


def es_modelo_knn(tipo_modelo):
    return normalizar_modelo(tipo_modelo) == "KNN"


def es_modelo_arbol(tipo_modelo):
    return normalizar_modelo(tipo_modelo) == "Arbol_decision"


def es_modelo_regresion_logistica(tipo_modelo):
    return normalizar_modelo(tipo_modelo) == "Regresion_logistica"


def es_modelo_credit_scoring(tipo_modelo):
    return normalizar_modelo(tipo_modelo) == "Credit_scoring"


if "phase" not in st.session_state:
    st.session_state.phase = "CARGA"
if "df" not in st.session_state:
    st.session_state.df = None
if "proposal" not in st.session_state:
    st.session_state.proposal = None
if "config_pipeline" not in st.session_state:
    st.session_state.config_pipeline = None
if "results" not in st.session_state:
    st.session_state.results = None
if "cleaner" not in st.session_state:
    st.session_state.cleaner = None
if "report_html" not in st.session_state:
    st.session_state.report_html = None


def get_ia_proposal(df, feedback=""):
    dtypes = df.dtypes.apply(lambda x: str(x)).to_dict()
    nulls = df.isnull().sum().to_dict()

    narrativa_solicitada = (
        "EMPIEZA TU RESPUESTA DICIENDO: 'Entendido, he procesado tus ajustes. Este es el nuevo plan estrategico...'"
        if feedback
        else "Presenta un plan inicial de ciencia de datos."
    )

    prompt = f"""
    Eres un Consultor de Negocio y Estratega de Datos.
    Guia tecnica interna: {guia_tecnica}
    Metadatos: {json.dumps(dtypes)}
    Valores nulos: {json.dumps(nulls)}

    INSTRUCCIONES DEL USUARIO: {feedback if feedback else 'Analisis inicial sin instrucciones previas.'}

    TAREA:
    1. {narrativa_solicitada} Explica la estrategia de negocio y por que elegiste el modelo.
    2. Incluye recomendaciones sobre nulos, outliers, inconsistencias, distribuciones y variables relevantes.
    3. NUNCA menciones nombres de funciones tecnicas internas de Python.
    4. Al final incluye un bloque JSON valido con: col_target, tipo_modelo, metodos_imputacion y EsPCA.
    """
    response = model_ia.generate_content(prompt)
    return response.text


st.title("Data Mining Autopilot")
st.text("Automatizacion del preprocesamiento de datos y entrenamiento de modelos de machine learning")

if st.session_state.phase == "CARGA":
    st.markdown("### Sube tus datos")
    col1, col2 = st.columns(2)

    with col1:
        hechos = st.file_uploader(
            "Tabla de hechos", type=["csv", "xlsx"]
        )
    with col2:
        dimensiones = st.file_uploader(
            "Dimensiones", type=["csv", "xlsx"],
            accept_multiple_files=True
        )

    if hechos and dimensiones:
        if st.button("✅ Cargar y construir cubo"):
            try:
                # Paso 1: subir a GCS
                with st.spinner("Subiendo archivos a Cloud Storage..."):
                    subir_a_gcs(hechos, "Tabla_hechos")
                    for dim in dimensiones:
                        subir_a_gcs(dim, "Dimensiones")
                st.success("Archivos subidos")

                # Paso 2: esperar que el trigger cargue a BigQuery
                nombres_esperados = ["hechos_raw"] + [
                    f"dim_{dim.name.split('.')[0].upper()}_raw"
                    for dim in dimensiones
                ]
                with st.spinner("Esperando carga en BigQuery..."):
                    ok = esperar_tablas_bq(nombres_esperados)

                if not ok:
                    st.error("Timeout: las tablas no aparecieron en BigQuery. Revisa los logs.")
                    st.stop()
                st.success("Tablas cargadas en BigQuery")

                # Paso 3: construir la vista
                with st.spinner("Construyendo cubo analítico..."):
                    ok, resultado = llamar_build_cubo()

                if not ok:
                    st.error(f"Error al construir el cubo: {resultado}")
                    st.stop()
                st.success("Cubo construido")

                # Paso 4: leer el cubo a dataframe para el resto del flujo
                with st.spinner("Cargando datos para análisis..."):
                    st.session_state.df = leer_cubo_de_bq()

                st.session_state.phase = "PROPUESTA"
                st.rerun()

            except Exception as e:
                st.error(f"Error: {str(e)}")

elif st.session_state.phase == "PROPUESTA":
    tab1, tab2 = st.tabs(["Estrategia de IA", "Reporte de Datos"])

    with tab1:
        if not st.session_state.proposal:
            with st.spinner("El agente esta disenando la estrategia inicial..."):
                st.session_state.proposal = get_ia_proposal(st.session_state.df)

        json_match = re.search(r"```json\s*(\{.*?\})\s*```", st.session_state.proposal, re.DOTALL)
        json_str = json_match.group(1) if json_match else "{}"
        explicacion = re.sub(r"```json.*?```", "", st.session_state.proposal, flags=re.DOTALL).strip()

        col1, col2 = st.columns([1.6, 1.4], gap="large")

        with col1:
            st.markdown("### Propuesta Estrategica")
            st.markdown(f'<div class="equal-height-box">{explicacion}</div>', unsafe_allow_html=True)

        with col2:
            st.markdown("### Configuracion Tecnica")
            try:
                conf_data = json.loads(json_str)
                target_detectado, reglas_detectadas, modelo_detectado, es_pca_detectado = extraer_configuracion_pipeline(conf_data)
                opciones_modelos = list(MODELOS_DISPONIBLES.keys())
                modelo_seleccionado = st.selectbox(
                    "Modelo detectado",
                    opciones_modelos,
                    index=opciones_modelos.index(modelo_detectado),
                )
                conf_data["modelo"] = modelo_seleccionado
                conf_data["tipo_modelo"] = modelo_seleccionado

                target_validacion = (
                    None
                    if MODELOS_DISPONIBLES[modelo_seleccionado]["tipo_problema"] == "clustering"
                    else target_detectado
                )
                if target_validacion is None:
                    conf_data["target"] = None
                    conf_data["col_target"] = None

                target_mostrado = target_validacion if target_validacion else "No requerido"
                st.markdown(f"**Target:** `{target_mostrado}`")
                st.markdown(f"**Modelo:** `{modelo_seleccionado}`")
                st.markdown(f"**Metricas esperadas:** `{', '.join(obtener_metricas_esperadas(modelo_seleccionado))}`")

                es_valido, mensaje_validacion = validar_tipo_problema(
                    st.session_state.df, target_validacion, modelo_seleccionado
                )
                if es_valido:
                    st.success(f"Validacion del problema: {mensaje_validacion}")
                else:
                    st.error(f"Validacion del problema: {mensaje_validacion}")

                puede_ejecutar = es_valido and MODELOS_DISPONIBLES[modelo_seleccionado]["implementado"]

                st.markdown("### Tratamiento de nulos y columnas")
                if reglas_detectadas:
                    table_data = []
                    for col, params in reglas_detectadas.items():
                        params_validos = isinstance(params, dict)
                        table_data.append(
                            {
                                "Columna": col,
                                "Tratamiento": "AUTO",
                                "Estado": "✅" if params_validos else "❌",
                                "Dummies": "Si" if params_validos and params.get("Dummies") else "No",
                            }
                        )
                    st.table(pd.DataFrame(table_data))
            except Exception as e:
                conf_data = {}
                puede_ejecutar = False
                st.error(f"Error en configuracion JSON: {e}")

            if st.button("Ejecutar Pipeline", use_container_width=True, disabled=not puede_ejecutar):
                st.session_state.config_pipeline = conf_data
                st.session_state.phase = "EJECUCION"
                st.rerun()

        st.markdown("### Refinar Plan y Recomendaciones")
        feedback_val = st.text_area(
            "Anade tus ajustes o contexto adicional:",
            placeholder="Ej: No utilices la columna X, cambia el modelo a Y, elimina la columna Z, etc...",
            label_visibility="collapsed",
        )

        if st.button("Actualizar Propuesta Estrategica", use_container_width=True):
            instruction = f"Usa este json como base: {json_str}. Cambia solo lo que pida el feedback: {feedback_val}"
            with st.spinner("Ajustando estrategia..."):
                st.session_state.proposal = get_ia_proposal(st.session_state.df, instruction)
                st.rerun()

    with tab2:
        st.markdown("### Reporte Exploratorio Detallado")
        if not st.session_state.report_html:
            with st.spinner("Generando reporte interactivo de calidad de datos..."):
                st.session_state.report_html = AnalizarDatos(st.session_state.df)
        components.html(st.session_state.report_html, height=1000, scrolling=True)

elif st.session_state.phase == "EJECUCION":
    conf = st.session_state.config_pipeline
    try:
        target, reglas, modelo_t, es_pca = extraer_configuracion_pipeline(conf)

        with st.spinner("Iniciando limpieza automatizada..."):
            cleaner, X, y = aplicar_limpieza_interna(
                st.session_state.df,
                col_target=target,
                reglas_dict=reglas,
                es_pca=es_pca,
            )
            st.session_state.cleaner = cleaner

            try:
                if y is not None:
                    df_export = pd.concat([X, y], axis=1)
                else:
                    df_export = X.copy()
                df_export.to_excel("dataset_limpio.xlsx", index=False)
                st.success("Dataset limpio guardado como 'dataset_limpio.xlsx'")
            except Exception as e:
                st.warning(f"Error al guardar excel: {e}")

        with st.spinner(f"Optimizando y entrenando {modelo_t}..."):
            X_numeric = X.select_dtypes(include=["number"])
            cols_eliminadas = set(X.columns) - set(X_numeric.columns)
            if cols_eliminadas:
                st.warning(f"Columnas eliminadas por seguridad: {list(cols_eliminadas)}")

            X_numeric = X_numeric.apply(pd.to_numeric, errors="coerce").fillna(0).astype(float)
            if X_numeric.empty:
                raise ValueError("No quedaron columnas numericas disponibles para entrenar el modelo.")

            modelo_obj, metricas, cols = orquestador_modelos_interno(X_numeric, y, tipo_modelo=modelo_t)
            st.session_state.results = {
                "modelo": modelo_obj,
                "metricas": metricas,
                "cols": cols,
                "tipo_modelo": modelo_t,
            }

        st.session_state.phase = "RESULTADOS"
        st.rerun()
    except Exception as e:
        st.error(f"Error en el Pipeline: {e}")
        if st.button("Reintentar Propuesta"):
            st.session_state.phase = "PROPUESTA"
            st.rerun()

elif st.session_state.phase == "RESULTADOS":
    res = st.session_state.results
    st.balloons()

    tipo_modelo = res.get("tipo_modelo", "Regresion_lineal")
    es_clustering_resultado = es_modelo_clustering(tipo_modelo)
    es_redes_resultado = es_modelo_redes_neuronales(tipo_modelo)
    es_knn_resultado = es_modelo_knn(tipo_modelo)
    es_arbol_resultado = es_modelo_arbol(tipo_modelo)
    es_logistica_resultado = es_modelo_regresion_logistica(tipo_modelo)
    es_credit_resultado = es_modelo_credit_scoring(tipo_modelo)
    metricas_interfaz = ocultar_woe_interfaz(res["metricas"])

    st.markdown("### Interpretacion Estrategica del Autopilot")
    with st.spinner("Analizando resultados del modelo..."):
        if es_clustering_resultado:
            tarea = "Analiza la calidad de separacion, numero optimo de clusters y oportunidades de segmentacion."
        elif es_credit_resultado:
            tarea = "Analiza KS, Gini, ROC-AUC, segmentos de riesgo y scorecard."
        elif es_logistica_resultado:
            tarea = "Analiza metricas, class_weight, curva ROC, matriz de confusion y coeficientes."
        elif es_arbol_resultado:
            tarea = "Analiza si se eligio arbol o RandomForest, metricas e importancia de variables."
        elif es_knn_resultado:
            tarea = "Analiza K, weights, distancia, metricas y curva error vs K."
        elif es_redes_resultado:
            tarea = "Analiza si resolvio clasificacion o regresion, metricas y curva de perdida."
        else:
            tarea = "Analiza relevancia, fiabilidad, metricas e impacto de negocio."

        interp_prompt = f"""
        Actua como un Consultor de Data Science Senior.
        Resultados: {json.dumps(metricas_interfaz)}
        Variables usadas: {res['cols']}
        Tarea: {tarea}
        Concluye con una recomendacion estrategica.
        """
        explicacion = model_ia.generate_content(interp_prompt).text
        st.markdown(explicacion)

    st.write(f"**Variables procesadas:** {', '.join(res['cols'])}")

    if es_clustering_resultado:
        metricas_cluster = metricas_interfaz
        st.markdown("### Resultados de Clustering")
        st.write(f"**Algoritmo seleccionado:** {metricas_cluster.get('modelo_seleccionado')}")
        st.write(f"**Mejor numero de clusters:** {metricas_cluster.get('mejor_numero_clusters')}")
        st.json(metricas_cluster.get("metricas_precision", {}))

        visualizaciones = metricas_cluster.get("visualizaciones", {})
        col_img1, col_img2 = st.columns(2)
        with col_img1:
            if visualizaciones.get("scatter_clusters"):
                st.image(visualizaciones["scatter_clusters"], caption="Scatter de clusters")
            if visualizaciones.get("dendrograma"):
                st.image(visualizaciones["dendrograma"], caption="Dendrograma")
        with col_img2:
            if visualizaciones.get("elbow_chart"):
                st.image(visualizaciones["elbow_chart"], caption="Metodo del codo")

        with st.expander("Comparacion de algoritmos"):
            st.json(metricas_cluster.get("metricas_por_algoritmo", []))
    else:
        if es_redes_resultado:
            metricas_red = metricas_interfaz
            st.markdown("### Resultados de Redes Neuronales")
            st.write(f"**Tipo de problema:** {metricas_red.get('tipo_problema')}")
            st.json(metricas_red.get("metricas_precision", {}))
            visualizaciones = metricas_red.get("visualizaciones", {})
            col_nn1, col_nn2 = st.columns(2)
            with col_nn1:
                if visualizaciones.get("perdida"):
                    st.image(visualizaciones["perdida"], caption="Curva de perdida")
                if visualizaciones.get("curva_roc"):
                    st.image(visualizaciones["curva_roc"], caption="Curva ROC")
            with col_nn2:
                if visualizaciones.get("matriz_confusion"):
                    st.image(visualizaciones["matriz_confusion"], caption="Matriz de confusion")
                if visualizaciones.get("real_vs_prediccion"):
                    st.image(visualizaciones["real_vs_prediccion"], caption="Real vs prediccion")

        if es_knn_resultado:
            metricas_knn = metricas_interfaz
            st.markdown("### Resultados de KNN")
            st.write(f"**Tipo de problema:** {metricas_knn.get('tipo_problema')}")
            st.json(metricas_knn.get("metricas_precision", {}))
            visualizaciones = metricas_knn.get("visualizaciones", {})
            col_knn1, col_knn2 = st.columns(2)
            with col_knn1:
                if visualizaciones.get("error_vs_k"):
                    st.image(visualizaciones["error_vs_k"], caption="Error vs K")
                if visualizaciones.get("matriz_confusion"):
                    st.image(visualizaciones["matriz_confusion"], caption="Matriz de confusion")
            with col_knn2:
                if visualizaciones.get("real_vs_prediccion"):
                    st.image(visualizaciones["real_vs_prediccion"], caption="Prediccion vs valores reales")

        if es_arbol_resultado:
            metricas_arbol = metricas_interfaz
            st.markdown("### Resultados de Arboles")
            st.write(f"**Modelo seleccionado:** {metricas_arbol.get('modelo_seleccionado')}")
            st.write(f"**Tipo de problema:** {metricas_arbol.get('tipo_problema')}")
            st.json(metricas_arbol.get("metricas_precision", {}))
            visualizaciones = metricas_arbol.get("visualizaciones", {})
            col_tree1, col_tree2 = st.columns(2)
            with col_tree1:
                if visualizaciones.get("arbol"):
                    st.image(visualizaciones["arbol"], caption="Arbol")
                if visualizaciones.get("matriz_confusion"):
                    st.image(visualizaciones["matriz_confusion"], caption="Matriz de confusion")
            with col_tree2:
                if visualizaciones.get("feature_importance"):
                    st.image(visualizaciones["feature_importance"], caption="Feature importance")

        if es_logistica_resultado:
            metricas_log = metricas_interfaz
            st.markdown("### Resultados de Regresion Logistica")
            st.write(f"**Tipo de problema:** {metricas_log.get('tipo_problema')}")
            st.json(metricas_log.get("metricas_precision", {}))
            visualizaciones = metricas_log.get("visualizaciones", {})
            col_log1, col_log2 = st.columns(2)
            with col_log1:
                if visualizaciones.get("curva_roc"):
                    st.image(visualizaciones["curva_roc"], caption="Curva ROC")
                if visualizaciones.get("matriz_confusion"):
                    st.image(visualizaciones["matriz_confusion"], caption="Matriz de confusion")
            with col_log2:
                if visualizaciones.get("coeficientes"):
                    st.image(visualizaciones["coeficientes"], caption="Coeficientes")

        if es_credit_resultado:
            metricas_credit = metricas_interfaz
            st.markdown("### Resultados de Credit Scoring")
            st.write(f"**Tipo de problema:** {metricas_credit.get('tipo_problema')}")
            st.json(metricas_credit.get("metricas_precision", {}))
            visualizaciones = metricas_credit.get("visualizaciones", {})
            col_credit1, col_credit2 = st.columns(2)
            with col_credit1:
                if visualizaciones.get("score_distribution"):
                    st.image(visualizaciones["score_distribution"], caption="Distribucion de score")
                if visualizaciones.get("curva_roc"):
                    st.image(visualizaciones["curva_roc"], caption="Curva ROC")
                if visualizaciones.get("matriz_confusion"):
                    st.image(visualizaciones["matriz_confusion"], caption="Matriz de confusion")
            with col_credit2:
                if visualizaciones.get("segmentos_riesgo"):
                    st.image(visualizaciones["segmentos_riesgo"], caption="Segmentos de riesgo")
                if visualizaciones.get("coeficientes"):
                    st.image(visualizaciones["coeficientes"], caption="Coeficientes")

        with st.expander("Validacion, hiperparametros y detalles"):
            st.json(
                {
                    "validacion": metricas_interfaz.get("validacion", {}),
                    "mejores_hiperparametros": metricas_interfaz.get("mejores_hiperparametros", {}),
                    "comparacion_distancias": metricas_interfaz.get("comparacion_distancias", []),
                    "scorecard": metricas_interfaz.get("scorecard", {}),
                    "segmentos_riesgo": metricas_interfaz.get("segmentos_riesgo", {}),
                }
            )

        entrada = st.text_input("Realizar prediccion (formato CSV):")
        if st.button("Predecir"):
            try:
                df_n = pd.read_csv(StringIO(entrada))
                df_p = st.session_state.cleaner.transformar_nueva_tupla(df_n)
                p = res["modelo"].predict(df_p[res["cols"]])
                st.success(f"Resultado predicho: {p[0]}")
            except Exception as e:
                st.error(f"Error: {e}")

    if st.button("Iniciar Nuevo Proyecto"):
        st.session_state.clear()
        st.rerun()
