import json
import os
import google.generativeai as genai
from google.api_core import exceptions
import pandas as pd
import streamlit as st
import requests
import time
import numpy as np
from google.cloud import storage, bigquery
import google.auth
import google.auth.transport.requests
import google.oauth2.id_token

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

gemini_key_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "credenciales", "GEMINI_KEY.txt"))

if not api_key:
    if os.path.exists(gemini_key_path):
        try:
            with open(gemini_key_path, "r", encoding="utf-8") as f:
                api_key = f.read().strip()
        except Exception:
            pass
    elif os.path.exists("GEMINI_KEY.txt"):
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

def get_ia_proposal(chat_session, df, feedback="", is_initial=False, diccionario_datos=None):
    dtypes = df.dtypes.apply(lambda x: str(x)).to_dict()
    nulls = df.isnull().sum().to_dict()

    # Obtener muestra de datos categóricos para contexto real
    cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    sample_data = ""
    if cat_cols:
        sample_data = "\nMUESTRA DE DATOS CATEGÓRICOS (Primeras 3 filas):\n"
        sample_data += df[cat_cols].head(3).to_string()

    dict_context = ""
    dict_instruction = ""
    if diccionario_datos:
        dict_context = f"\n========================================\nDICCIONARIO DE DATOS PROPORCIONADO POR EL USUARIO (Información de negocio extra):\n{diccionario_datos}\n========================================\n"
        dict_instruction = '\nOBLIGATORIO: Como has recibido un diccionario de datos del usuario, es mandatorio que Finalices tu respuesta de análisis (o de confirmación de ajustes) con la frase exacta: "He recibido tu diccionario de datos, donde..." y continúes resumiendo brevemente lo que comprendes de él y cómo influye de manera provechosa en tu propuesta estratégica.\n'

    if is_initial:
        prompt = f"""
        Eres un Consultor de Negocio y Estratega de Datos llamado Autopilot.
        Guia tecnica interna: {guia_tecnica}
        Metadatos: {json.dumps(dtypes)}
        Valores nulos: {json.dumps(nulls)}
        {sample_data}
        {dict_context}

        TAREA INICIAL:
        1. Presenta un plan inicial de ciencia de datos con un enfoque 100% estratégico. Utiliza el diccionario de datos como información extra del negocio para mayor conocimiento de las variables.
        2. Usa títulos (##) que hablen de NEGOCIO (ej. 'Impacto en la Rentabilidad', 'Visión General del Proyecto') en lugar de términos técnicos.
        3. Explica la estrategia de datos y por qué elegiste el modelo sin usar jerga compleja.
        4. Incluye recomendaciones sobre la calidad de la información y variables relevantes.
        5. NUNCA menciones nombres de funciones técnicas internas de Python ni palabras como 'Pipeline', 'JSON' o 'Preprocesamiento' en tus títulos o explicaciones.
        {dict_instruction}
        
        RESTRICCIONES CRÍTICAS PARA EL JSON (OBLIGATORIO):
        - Columnas de Nombres, correos, ids, telefono, direccion (name, nombre, apellido, email, phone, address, etc.): DEBES usar "metodo": "drop-column". Está PROHIBIDO usar TargetEncoding o Dummies en ellas, a no ser que el usuario lo pida explícitamente.
        - Columnas con separadores (genres, tags, keywords): Si ves "|" o muchos espacios en la muestra, DEBES usar "Lematizar": true. Prohibido usar Dummies aquí, a no ser que el usuario lo pida explícitamente.
        - Todas las columnas del dataset original deben aparecer en reglas_dict.
        
        REGLA DE IDENTIDAD Y FIRMA (OBLIGATORIO):
        - Si decides firmar tu respuesta al inicio, final, o mencionarte hazlo única y exclusivamente como "Autopilot". Ejemplo: "Atentamente,\nAutopilot".
        
        6. Al final, incluye un bloque JSON válido con: col_target, tipo_modelo, reglas_dict y EsPCA (este bloque será ocultado automáticamente).
        """
    else:
        prompt = f"""
        Eres un Consultor de Negocio y Estratega de Datos llamado Autopilot.
        INSTRUCCIONES DEL USUARIO: "{feedback}"
        {dict_context}
        
        EVALUACIÓN DE INTENCIÓN:
        Analiza si el usuario te está pidiendo MODIFICAR el plan de tratamiento/modelo o si solo está haciendo una PREGUNTA/DUDA.
        
        SI PIDE MODIFICAR EL PLAN:
        1. Empieza diciendo: 'Entendido, he procesado tus ajustes...'
        2. Explica los cambios estratégicos.
        {dict_instruction}
        3. OBLIGATORIO: Al final incluye un nuevo bloque JSON válido con la configuración técnica actualizada (col_target, tipo_modelo, reglas_dict y EsPCA).
        
        SI SOLO HACE UNA PREGUNTA (o pide sugerencias/aclaraciones sin pedir cambios al plan):
        1. Responde a su duda detalladamente enfocándote en negocio, datos and algoritmos.
        2. NO hables de código.
        {dict_instruction}
        3. MUY IMPORTANTE: NO incluyas ningún bloque JSON al final. De esta forma el sistema sabrá que no hay cambios técnicos.
        
        REGLA DE IDENTIDAD Y FIRMA (OBLIGATORIO):
        - Si decides firmar tu respuesta al inicio, final, o mencionarte hazlo única y exclusivamente como "Autopilot". Ejemplo: "Atentamente,\nAutopilot".
        """

    
    try:
        response = chat_session.send_message(prompt)
        
        
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
    algoritmo   = model_info.get("algoritmo_seleccionado", "")
    target      = model_info.get("variable_obj", "No especificada")
    metricas    = model_info.get("metricas", {})
    n_reg       = model_info.get("n_registros", "—")
    n_feat      = model_info.get("n_features", "—")
    clases      = model_info.get("clases", [])
    encodings   = model_info.get("encodings", [])
    pca         = model_info.get("pca_aplicado", False)
    grid        = model_info.get("grid_search", False)
    coeficientes = model_info.get("coeficientes", {})

    metricas_str = "\n".join(f"    - {k}: {v}" for k, v in metricas.items()) if metricas else "    - No disponibles"
    clases_str   = ", ".join(str(c) for c in clases) if clases else "—"
    enc_str      = ", ".join(encodings) if encodings else "No especificados"
    algoritmo_str = f"{tipo} → {algoritmo}" if algoritmo else tipo

    coef_str = ""
    if coeficientes:
        coef_str = "\n    Coeficientes por variable (caja blanca):\n"
        coef_str += "\n".join(f"      · {k}: {v}" for k, v in coeficientes.items())
        coef_str += "\n    (Un coeficiente positivo significa que al aumentar esa variable, el resultado sube; negativo = baja)"

    advertencia_r2 = ""
    r2_prueba = metricas.get("R2_prueba (conjunto de test)")
    r2_train  = metricas.get("R2_entrenamiento (solo conjunto de entrenamiento)")
    if r2_prueba is not None and r2_train is not None:
        advertencia_r2 = f"""
    ⚠️ DIFERENCIA IMPORTANTE ENTRE LAS MÉTRICAS DE R²:
    - R²_prueba = {r2_prueba} → Este es el R² REAL. Mide qué tan bien predice el modelo en datos NUNCA VISTOS. Usa ESTE valor para tu titular.
    - R²_entrenamiento = {r2_train} → Solo refleja qué tan bien memorizó el conjunto de entrenamiento. NO uses este para tu headline.
    - Si R²_entrenamiento > R²_prueba hay un riesgo de sobreajuste (overfitting) que debes mencionar."""

    return f"""Eres 'Autopilot', un Consultor Élite en Estrategia de IA y Negocios, experto en Machine Learning integrado en "Data Mining Autopilot".
    Tu misión es ayudar al usuario a entender e interpretar el modelo recién entrenado.

    INSTRUCCIONES PARA EL "EFECTO WOW" (TRADUCCIÓN A NEGOCIO):
¡Prohibido sonar como un libro de texto de estadística! Traduce cada métrica técnica a su equivalente financiero u operativo usando esta guía de pensamiento:

    ════════════════════════════════════════
    CONTEXTO DEL MODELO ENTRENADO
    ════════════════════════════════════════
      Tipo / Algoritmo usado : {algoritmo_str}
      Variable objetivo      : {target}
      Clases (si aplica)     : {clases_str}
      Registros              : {n_reg}
      Features usadas        : {n_feat}
      Encodings aplicados    : {enc_str}
      PCA aplicado           : {"Sí" if pca else "No"}
      GridSearchCV usado     : {"Sí" if grid else "No"}

      Métricas obtenidas:
    {metricas_str}
    {advertencia_r2}
    {coef_str}
    ════════════════════════════════════════

    INSTRUCCIONES:
      1. Responde siempre en español, de forma clara y útil.
      2. Menciona explícitamente el nombre del algoritmo/modelo usado.
      3. Usa ÚNICAMENTE R²_prueba (conjunto de test) para tu titular y narrativa. NUNCA uses R²_entrenamiento como medida de rendimiento real.
      4. Si hay coeficientes, úsalos para explicar qué variables mueven más el resultado y en qué dirección.
      5. Traduce métricas técnicas a lenguaje de negocio.
      6. Si detectas problemas (overfitting, desbalanceo, etc.) menciónalos.
      7. Sé conciso pero completo.
      8. Si decides firmar tu respuesta al inicio, final, o mencionarte hazlo única y exclusivamente como "Autopilot". Ejemplo: "Atentamente,\nAutopilot".
    
    ESTRUCTURA OBLIGATORIA DE LA RESPUESTA:
    1. El Titular WOW: Una sola frase de alto impacto que resuma el mayor beneficio del modelo para el negocio (Ej: "Nuestra nueva herramienta predictiva nos permite capturar el 85% de las fugas de clientes antes de que ocurran").
    2. La Historia del Desempeño: Explica los resultados basándote en la guía de traducción anterior. Conecta los números con escenarios de la vida real (ventas, ahorros, mitigación de riesgos).
    3. Recomendación Estratégica: ¿Qué decisión se debe tomar MAÑANA con esta herramienta? ¿Cómo sugerimos implementarla en la operación diaria? 
    
    """

def interpretar_resultados(chat_session, metricas_interfaz, cols, tarea):
    algoritmo = metricas_interfaz.get("modelo_seleccionado", "")
    tipo_modelo = metricas_interfaz.get("tipo_modelo", "")
    algoritmo_str = f"{tipo_modelo} → {algoritmo}" if algoritmo else tipo_modelo

    r2_prueba = metricas_interfaz.get("metricas_precision", {}).get("R2_prueba (conjunto de test)")
    r2_train  = metricas_interfaz.get("metricas_precision", {}).get("R2_entrenamiento (solo conjunto de entrenamiento)")
    advertencia_r2 = ""
    if r2_prueba is not None and r2_train is not None:
        advertencia_r2 = f"IMPORTANTE: R²_prueba={r2_prueba} es el valor REAL (datos nunca vistos). R²_entrenamiento={r2_train} solo refleja la memorización. Tu titular debe basarse en R²_prueba."

    coeficientes = metricas_interfaz.get("coeficientes_por_variable", {})
    coef_str = ""
    if coeficientes:
        coef_str = "Coeficientes del modelo (caja blanca): " + ", ".join(f"{k}={v}" for k, v in coeficientes.items() if k != "_intercepto")

    interp_prompt = f"""
    Actúa como 'Autopilot', un Consultor Élite en Estrategia de IA y Negocios. Tu objetivo es generar un "Efecto WOW", traduciendo métricas de evaluación de modelos predictivos en impacto real y tangible para directivos que no tienen perfil técnico.

    MODELO USADO: {algoritmo_str if algoritmo_str else "Ver tipo_modelo en JSON"}
    CONTEXTO DEL MODELO:
    Resultados de las métricas (JSON): {json.dumps(metricas_interfaz)}
    Variables que impulsan el modelo: {cols}
    Objetivo de Negocio / Tarea: {tarea}
    {advertencia_r2}
    {coef_str}

    INSTRUCCIONES PARA EL "EFECTO WOW" (TRADUCCIÓN A NEGOCIO):
    ¡Prohibido sonar como un libro de texto de estadística! Traduce cada métrica técnica a su equivalente financiero u operativo usando esta guía de pensamiento:
    - Accuracy (Exactitud): Preséntalo como la "Certeza Operativa". ¿Qué tanta confianza puede tener el negocio en esta herramienta?
    - Precision (Precisión): Tradúcelo como "Eficiencia de Recursos / ROI". De cada 100 veces que el modelo sugiere invertir tiempo o dinero, ¿cuántas damos en el blanco sin desperdiciar recursos (Falsos Positivos)?
    - Recall (Exhaustividad): Tradúcelo como "Captura de Oportunidades o Prevención de Riesgos". Del total del pastel que hay en el mercado (o el total de fraudes/fallos), ¿qué porcentaje estamos logrando "atrapar" para que no se escape (Falsos Negativos)?
    - RMSE / MAE (Errores): Tradúcelos como el "Margen de Desviación Financiera/Operativa".

    ESTRUCTURA OBLIGATORIA DE LA RESPUESTA:
    1. El Titular WOW: Una sola frase de alto impacto que resuma el mayor beneficio del modelo para el negocio (Ej: "Nuestra nueva herramienta predictiva nos permite capturar el 85% de las fugas de clientes antes de que ocurran").
    2. La Historia del Desempeño: Explica los resultados basándote en la guía de traducción anterior. Conecta los números con escenarios de la vida real (ventas, ahorros, mitigación de riesgos).
    3. El Motor del Modelo: Menciona brevemente, en lenguaje sencillo, cuáles son las 2 o 3 variables principales ({cols}) que están moviendo la aguja. Si hay coeficientes disponibles, explica la dirección del efecto ("a mayor X, el resultado sube/baja").
    4. Recomendación Estratégica: ¿Qué decisión se debe tomar MAÑANA con esta herramienta? ¿Cómo sugerimos implementarla en la operación diaria?

    REGLA DE IDENTIDAD Y FIRMA (ESTRICTAMENTE OBLIGATORIO):
    - Tienes totalmente prohibido presentarte como una IA. 
    - Si decides firmar tu respuesta al inicio, al final, o hacer referencia a ti mismo, hazlo única y exclusivamente como "Autopilot". 
    - Ejemplo de firma permitida: "Atentamente, Autopilot".
    """
    return _enviar_mensaje_ia(chat_session, interp_prompt, "Interpretación de Resultados")

def interpretar_resultados_perfilarDatos(chat_session, metricas_interfaz, cols, perfiles, tarea):
    """
    Función especializada para perfilar clases/clusters usando estadísticas descriptivas.
    """
    perfiles_acotados = {k: v for k, v in perfiles.items()} # Evitar saturar contexto si es muy grande
    
    interp_prompt = f"""
    Actúa como 'Autopilot', un Estratega de Negocios de Alto Nivel y Experto en Data Storytelling. Tu objetivo es traducir resultados de modelos de datos en narrativas de negocio accionables, empáticas y completamente libres de jerga técnica, dirigidas a tomadores de decisiones (CEOs, Marketing, Ventas).
    
    CONTEXTO TÉCNICO:
    Resultados del modelo: {json.dumps(metricas_interfaz)}
    Variables procesadas: {cols}
    
    PERFILAMIENTO DE LOS GRUPOS (ESTADÍSTICAS):
    {json.dumps(perfiles_acotados)}
    
    TAREA ESPECÍFICA:
    {tarea}
    Convierte estos grupos estadísticos en "Arquetipos de Cliente/Negocio". Debes contar una historia con los datos, dándole a cada grupo una identidad clara y humana que resuene con directivos no técnicos.
    
    REGLAS PARA EL DATA STORYTELLING (ESTRATEGIA Y PERFILAMIENTO):
    1. Tono: Ejecutivo, persuasivo y comercial. CERO jerga técnica (prohibido usar palabras como "clusters", "dispersión", "p-values" o "variables" en la narrativa).
    2. Para CADA grupo encontrado, estructura tu respuesta exactamente así:
   - Nombre del Arquetipo: Un título corto, creativo y memorable (ej. "Los Exploradores Digitales", "El Motor de Rentabilidad").
   - La Historia: Un párrafo narrativo que describa quiénes son en el mundo real, cómo se comportan y qué los motiva, basado estrictamente en sus datos estadísticos.
   - El Respaldo: 2 o 3 viñetas con los datos clave que los definen, pero traducidos a lenguaje de negocio (ej. "Tienen el ticket de compra más alto" en lugar de "mean_spending = 85.4").
   - Plan de Acción: Una recomendación estratégica clara. ¿Cómo los monetizamos, fidelizamos o qué riesgo mitigamos en este segmento?
    
    REGLA DE IDENTIDAD Y FIRMA (OBLIGATORIO):
    - Si decides firmar tu respuesta al inicio, final, o mencionarte hazlo única y exclusivamente como "Autopilot". Ejemplo: "Atentamente,\nAutopilot".
    """
    return _enviar_mensaje_ia(chat_session, interp_prompt, "Perfilamiento de Datos")

def _enviar_mensaje_ia(chat_session, prompt, titulo_log):
    
    response = chat_session.send_message(prompt)
    
    
    return response.text

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

    response = chat_session.send_message(prompt)

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

# --- CONFIGURACIÓN GCP ---
GCS_BUCKET = "archivos_back"
PROJECT    = "project-6d52cafa-4432-4186-aeb"
DATASET    = "Cubo"
CF_URL     = "https://armar-cubo-697875837946.northamerica-south1.run.app"

def limpiar_dataset_anterior():
    client = get_bq_client()
    gcs    = get_storage_client()
    tablas = client.list_tables(f"{PROJECT}.{DATASET}")
    for tabla in tablas:
        if tabla.table_id.endswith("_raw") or tabla.table_id == "cubo_analitico":
            client.delete_table(f"{PROJECT}.{DATASET}.{tabla.table_id}", not_found_ok=True)
    bucket = gcs.bucket(GCS_BUCKET)
    for carpeta in ["Tabla_hechos", "Dimensiones"]:
        for blob in list(bucket.list_blobs(prefix=f"{carpeta}/")):
            blob.delete()

def subir_a_gcs(archivo, carpeta):
    client = get_storage_client()
    bucket = client.bucket(GCS_BUCKET)
    blob = bucket.blob(f"{carpeta}/{archivo.name}")
    blob.upload_from_file(archivo, rewind=True)

def tabla_existe_en_bq(tabla_id):
    client = get_bq_client()
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

def llamar_build_cubo(nombres_dims=None):
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
            },
            json={"dimensiones": nombres_dims or []}
        )
        return response.status_code == 200, response.json()
    except Exception as e:
        return False, str(e)

def get_bq_client():
    import google.auth
    credentials, project = google.auth.default()
    return bigquery.Client(
        credentials=credentials,
        project=PROJECT,
        location="northamerica-south1"
    )

def get_storage_client():
    import google.auth
    credentials, project = google.auth.default()
    return storage.Client(credentials=credentials, project=PROJECT)

def _normalizar_clave_metrica(clave):
    return str(clave).lower().replace("_", "").replace("-", "").replace(" ", "")

def _obtener_metrica(metricas, nombres):
    if not isinstance(metricas, dict):
        return None
    objetivo = {_normalizar_clave_metrica(n) for n in nombres}
    for clave, valor in metricas.items():
        if _normalizar_clave_metrica(clave) in objetivo:
            return valor
    return None

def _formatear_metrica(valor):
    if valor is None:
        return "N/D"
    if isinstance(valor, (int, np.integer)):
        return f"{int(valor):,}"
    if isinstance(valor, (float, np.floating)):
        return f"{float(valor):,.4f}"
    return str(valor)

def _tipo_resultado_dashboard(metricas, es_clustering):
    tipo = str(metricas.get("tipo_problema", "")).lower() if isinstance(metricas, dict) else ""
    if es_clustering:
        return "clustering"
    if "regresion" in tipo or "regression" in tipo:
        return "regresion"
    if "forecast" in tipo or "sarima" in tipo:
        return "forecasting"
    return "clasificacion"

def _metricas_dashboard(metricas, es_clustering):
    metricas_precision = metricas.get("metricas_precision", {}) if isinstance(metricas, dict) else {}
    tipo = _tipo_resultado_dashboard(metricas, es_clustering)

    if tipo == "clustering":
        return [
            ("Silhouette Score", _obtener_metrica(metricas_precision, ["Silhouette Score", "silhouette"])),
            ("Davies Bouldin", _obtener_metrica(metricas_precision, ["Davies-Bouldin Index", "Davies Bouldin"])),
            ("Numero de clusters", metricas.get("mejor_numero_clusters") or metricas.get("n_clusters")),
        ]
    if tipo == "regresion":
        return [
            ("RMSE", _obtener_metrica(metricas_precision, ["RMSE"])),
            ("MAE", _obtener_metrica(metricas_precision, ["MAE"])),
            ("MAPE", _obtener_metrica(metricas_precision, ["MAPE"])),
            ("R²", _obtener_metrica(metricas_precision, ["R2", "R²"])),
        ]
    if tipo == "forecasting":
        return [
            ("MAE", _obtener_metrica(metricas_precision, ["MAE"])),
            ("RMSE", _obtener_metrica(metricas_precision, ["RMSE"])),
            ("MSE", _obtener_metrica(metricas_precision, ["MSE"])),
            ("MAPE", _obtener_metrica(metricas_precision, ["MAPE"])),
        ]
    return [
        ("Accuracy", _obtener_metrica(metricas_precision, ["Accuracy"])),
        ("Precision", _obtener_metrica(metricas_precision, ["Precision"])),
        ("Recall", _obtener_metrica(metricas_precision, ["Recall"])),
        ("F1 Score", _obtener_metrica(metricas_precision, ["F1-Score", "F1", "F1 Score"])),
        ("ROC-AUC", _obtener_metrica(metricas_precision, ["ROC-AUC", "ROC AUC"])),
        ("PR-AUC", _obtener_metrica(metricas_precision, ["PR-AUC", "PR AUC"])),
    ]

def _contar_logs(logs, patrones):
    total = 0
    for log in logs:
        accion = str(log.get("accion", "")).lower()
        if any(p in accion for p in patrones):
            total += 1
    return total

def _columnas_eliminadas_desde_logs(logs, columnas_originales, columnas_finales):
    eliminadas = set(columnas_originales) - set(columnas_finales)
    for log in logs:
        accion = str(log.get("accion", "")).lower()
        col = log.get("columna")
        if col and col != "__dataset__" and ("eliminacion_columna" in accion or "borrada" in accion):
            eliminadas.add(col)
    return sorted(eliminadas)

def _resumen_tecnico_pipeline(res, cleaner, df_original):
    logs = getattr(cleaner, "logs_limpieza", []) if cleaner is not None else []
    columnas_originales = list(df_original.columns) if df_original is not None else []
    columnas_finales = list(getattr(cleaner, "df", pd.DataFrame()).columns) if cleaner is not None else []
    columnas_usadas = res.get("cols", [])
    eliminadas = _columnas_eliminadas_desde_logs(logs, columnas_originales, columnas_finales + [res.get("target")])

    return {
        "columnas_originales": len(columnas_originales),
        "columnas_usadas": len(columnas_usadas),
        "columnas_eliminadas": eliminadas,
        "nulos_tratados": _contar_logs(logs, ["imputacion", "eliminacion_filas_nulas"]),
        "outliers_corregidos": _contar_logs(logs, ["outliers_recortados"]),
        "categoricas_codificadas": _contar_logs(logs, ["dummies", "target_encoding", "ordinal_encoding", "woe"]),
        "tiempo_total": res.get("tiempo_total_ejecucion", "N/D"),
        "logs": logs,
    }

def _render_metricas_clave(metricas_filtradas):
    visibles = [(nombre, valor) for nombre, valor in metricas_filtradas if valor is not None]
    if not visibles:
        return
    for inicio in range(0, len(visibles), 4):
        cols_metricas = st.columns(min(4, len(visibles) - inicio))
        for col_ui, (nombre, valor) in zip(cols_metricas, visibles[inicio:inicio + 4]):
            col_ui.metric(nombre, _formatear_metrica(valor))
