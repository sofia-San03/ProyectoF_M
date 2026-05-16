import json
import os
import google.generativeai as genai
from google.api_core import exceptions
import pandas as pd
import streamlit as st

from CODIGO.CleanData import Transformar_Df
from CODIGO.MODELS import (
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

def load_css(file_name):
    try:
        with open(file_name, "r") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    except FileNotFoundError:
        st.warning(f"No se encontró el archivo de estilos: {file_name}")

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

def iniciar_chat():
    return model_ia.start_chat(history=[])

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
    reglas1 = conf.get("reglas_dict", {})
    reglas2 = conf.get("metodos_imputacion", {})
    reglas = reglas1 if reglas1 else reglas2
    modelo_t = conf.get("tipo_modelo", conf.get("modelo", "Regresion_lineal"))
    modelo_normalizado = normalizar_modelo(modelo_t)
    es_pca = conf.get("EsPCA", conf.get("es_pca", False))
    n_clusters = conf.get("n_clusters", None)
    
    if MODELOS_DISPONIBLES[modelo_normalizado]["tipo_problema"] == "clustering":
        target = None
        
    return target, reglas, modelo_normalizado, es_pca, n_clusters

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

def orquestador_modelos_interno(X, y, tipo_modelo, n_clusters_fix=None):
    modelo_nombre = normalizar_modelo(tipo_modelo)
    info_modelo = MODELOS_DISPONIBLES[modelo_nombre]

    if not info_modelo["implementado"]:
        raise NotImplementedError(
            f"El modelo '{modelo_nombre}' esta reconocido por la arquitectura, pero no esta implementado."
        )

    if y is None and info_modelo["tipo_problema"] != "clustering":
        raise ValueError(f"El modelo '{modelo_nombre}' requiere variable objetivo.")

    if info_modelo["tipo_problema"] == "clustering":
        json_res, modelo, cols = info_modelo["funcion"](X, y, n_clusters_fix=n_clusters_fix)
    else:
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

def get_ia_proposal(chat_session, df, feedback="", is_initial=False):
    dtypes = df.dtypes.apply(lambda x: str(x)).to_dict()
    nulls = df.isnull().sum().to_dict()

    if is_initial:
        prompt = f"""
        Eres un Consultor de Negocio y Estratega de Datos.
        Guia tecnica interna: {guia_tecnica}
        Metadatos: {json.dumps(dtypes)}
        Valores nulos: {json.dumps(nulls)}

        TAREA INICIAL:
        1. Presenta un plan inicial de ciencia de datos. Explica la estrategia de negocio y por que elegiste el modelo.
        2. Incluye recomendaciones sobre nulos, outliers, inconsistencias, distribuciones y variables relevantes.
        3. NUNCA menciones nombres de funciones tecnicas internas de Python.
        4. Al final incluye un bloque JSON valido con: col_target, tipo_modelo, reglas_dict y EsPCA.
        """
    else:
        prompt = f"""
        Eres un Consultor de Negocio y Estratega de Datos.
        INSTRUCCIONES DEL USUARIO: "{feedback}"
        
        EVALUACIÓN DE INTENCIÓN:
        Analiza si el usuario te está pidiendo MODIFICAR el plan de tratamiento/modelo o si solo está haciendo una PREGUNTA/DUDA.
        
        SI PIDE MODIFICAR EL PLAN:
        1. Empieza diciendo: 'Entendido, he procesado tus ajustes...'
        2. Explica los cambios estratégicos.
        3. OBLIGATORIO: Al final incluye un nuevo bloque JSON válido con la configuración técnica actualizada (col_target, tipo_modelo, reglas_dict y EsPCA).
        
        SI SOLO HACE UNA PREGUNTA (o pide sugerencias/aclaraciones sin pedir cambios al plan):
        1. Responde a su duda detalladamente enfocándote en negocio, datos y algoritmos.
        2. NO hables de código.
        3. MUY IMPORTANTE: NO incluyas ningún bloque JSON al final. De esta forma el sistema sabrá que no hay cambios técnicos.
        """

    print("\n" + "="*60)
    print("PROMPT ENVIADO A LA IA (Propuesta Estratégica):")
    print("="*60)
    print(prompt)
    print("="*60 + "\n")
    
    try:
        response = chat_session.send_message(prompt)
        
        print("\n" + "="*60)
        print("RESPUESTA DE LA IA (Propuesta Estratégica):")
        print("="*60)
        print(response.text)
        print("="*60 + "\n")
        
        return response.text
    except exceptions.ResourceExhausted:
        return "⚠️ **Error de Cuota Superado (429):** El Agente IA ha recibido demasiadas solicitudes en poco tiempo. Por favor, espera unos 30 segundos antes de volver a preguntar o intentar otro ajuste. Esto se debe a los límites de la capa gratuita de Gemini."
    except Exception as e:
        return f"⚠️ **Error al conectar con la IA:** {str(e)}"

def chat_resultados_ia_stream(chat_session, mensaje_usuario, model_context_prompt):
    prompt = f"""
    [INSTRUCCIONES DEL SISTEMA]
    {model_context_prompt}
    
    [NUEVA PREGUNTA DEL USUARIO]
    "{mensaje_usuario}"
    """
    try:
        response = chat_session.send_message(prompt, stream=True)
        return response
    except exceptions.ResourceExhausted:
        st.error("⚠️ **Error de Cuota (429):** Límite alcanzado. Espera 30 segundos.")
        return None
    except Exception as e:
        st.error(f"⚠️ **Error:** {str(e)}")
        return None

def build_model_context_prompt(model_info: dict) -> str:
    tipo        = model_info.get("tipo_modelo", "No especificado")
    target      = model_info.get("variable_obj", "No especificada")
    metricas    = model_info.get("metricas", {})
    n_reg       = model_info.get("n_registros", "—")
    n_feat      = model_info.get("n_features", "—")
    clases      = model_info.get("clases", [])
    encodings   = model_info.get("encodings", [])
    pca         = model_info.get("pca_aplicado", False)
    grid        = model_info.get("grid_search", False)

    metricas_str = "\n".join(f"    - {k}: {v}" for k, v in metricas.items()) if metricas else "    - No disponibles"
    clases_str   = ", ".join(str(c) for c in clases) if clases else "—"
    enc_str      = ", ".join(encodings) if encodings else "No especificados"

    return f"""Eres un asistente experto en Machine Learning integrado en "Data Mining Autopilot".
    Tu misión es ayudar al usuario a entender e interpretar el modelo recién entrenado.

    ════════════════════════════════════════
    CONTEXTO DEL MODELO ENTRENADO
    ════════════════════════════════════════
      Tipo de modelo       : {tipo}
      Variable objetivo    : {target}
      Clases (si aplica)   : {clases_str}
      Registros            : {n_reg}
      Features usadas      : {n_feat}
      Encodings aplicados  : {enc_str}
      PCA aplicado         : {"Sí" if pca else "No"}
      GridSearchCV usado   : {"Sí" if grid else "No"}

      Métricas obtenidas:
    {metricas_str}
    ════════════════════════════════════════

    INSTRUCCIONES:
      1. Responde siempre en español, de forma clara y útil.
      2. Traduce métricas técnicas a lenguaje de negocio.
      3. Si detectas problemas (overfitting, desbalanceo, etc.) menciónalos.
      4. Sé conciso pero completo.
    """

def interpretar_resultados(chat_session, metricas_interfaz, cols, tarea):
    interp_prompt = f"""
    Actua como un Consultor de Data Science Senior
    Resultados: {json.dumps(metricas_interfaz)}
    Variables usadas: {cols}
    Tarea: {tarea}
    Concluye con una recomendacion estrategica.
    """
    print("\n" + "="*60)
    print("PROMPT ENVIADO A LA IA (Interpretación de Resultados):")
    print("="*60)
    print(interp_prompt)
    print("="*60 + "\n")
    
    response_interp = chat_session.send_message(interp_prompt)
    explicacion = response_interp.text
    
    print("\n" + "="*60)
    print("RESPUESTA DE LA IA (Interpretación de Resultados):")
    print("="*60)
    print(explicacion)
    print("="*60 + "\n")
    
    return explicacion

def texto_a_dataframe(chat_session, texto_usuario, dtypes_dict):
    prompt = f"""
    Eres un experto en extracción de datos.
    Se requiere convertir la descripción en lenguaje natural de un usuario en un registro de datos estructurado.
    Las columnas originales del dataset y sus tipos de datos (dtypes) son:
    {json.dumps(dtypes_dict)}
    
    Descripción del usuario: "{texto_usuario}"
    
    Tu tarea:
    1. Extrae la información del texto y mapeala a las columnas dadas. 
    2. Si un dato no se menciona en absoluto y no se puede inferir, usa null para rellenarlo (el pipeline se encargará de imputarlo).
    3. Devuelve ÚNICAMENTE un bloque de código JSON válido donde las claves son los nombres de las columnas.
    """
    print("\n" + "="*60)
    print("PROMPT ENVIADO A LA IA (Texto a DataFrame):")
    print("="*60)
    print(prompt)
    print("="*60 + "\n")
    
    response = chat_session.send_message(prompt)
    
    print("\n" + "="*60)
    print("RESPUESTA DE LA IA (Texto a DataFrame):")
    print("="*60)
    print(response.text)
    print("="*60 + "\n")
    
    import re
    json_match = re.search(r"```json\s*(\{.*?\})\s*```", response.text, re.DOTALL)
    if json_match:
        json_str = json_match.group(1)
    else:
        json_str = response.text.replace('```', '').strip()
        # Intentar extraer solo lo que parece un diccionario
        dict_match = re.search(r"(\{.*?\})", json_str, re.DOTALL)
        if dict_match:
            json_str = dict_match.group(1)
            
    datos_json = json.loads(json_str)
    return pd.DataFrame([datos_json])
