#STREAMLIT

from google.generativeai.types import file_types
from google.ai import generativelanguage
import streamlit as st
import pandas as pd
import json
import os
import re
import time
import asyncio
import google.generativeai as genai
from CleanData import Transformar_Df
from MODELS import (
    Regresion_lineal, Regresion_logistica, Arbol_decision,
    Random_Forest_Regressor, Random_Forest_Clasificador,
    Red_Neuronal_Regressor, Red_Neuronal_Clasificador,
    KNN_Regressor, KNN_Clasificador,
    KMeans_Clustering, KMedoids_Clustering, KMedianas_Clustering,
)
from CargarDatos import AnalizarDatos
import streamlit.components.v1 as components

# NUEVO - ETL Warehouse (Orquestador)
from etl_warehouse import registrar_resultado_en_warehouse, ETLWarehouse

st.set_page_config(page_title="Autopilot", page_icon="⚡", layout="wide")

st.markdown("""
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
    .stTable { border-radius: 12px; overflow: hidden; border: 1px solid rgba(255,255,255,0.1); }
    thead tr th { background-color: #1e293b !important; color: #10b981 !important; }
    .stTabs [data-baseweb="tab-list"] { gap: 10px; background-color: transparent; }
    .stTabs [data-baseweb="tab"] {
        background-color: #1e293b; border-radius: 10px 10px 0 0;
        padding: 10px 20px; color: #94a3b8;
    }
    .stTabs [aria-selected="true"] { background-color: #10b981 !important; color: white !important; }
    .stMarkdown p, .stMarkdown li, .stTable, .stTextArea textarea,
    .stMarkdown span, .stMarkdown div { font-size: 1.25rem !important; line-height: 1.6 !important; }
    h1 { font-size: 2.8rem !important; }
    h2 { font-size: 2.2rem !important; }
    h3 { font-size: 1.8rem !important; }
    .equal-height-box {
        height: 600px; overflow-y: auto; padding: 25px;
        background: rgba(30, 41, 59, 0.3); border-radius: 20px;
        border: 1px solid rgba(255, 255, 255, 0.05);
        scrollbar-width: thin; scrollbar-color: #10b981 #1e293b;
    }
    .bottom-feedback-area {
        margin-top: 30px; padding: 30px;
        background: rgba(15, 23, 42, 0.6); border-radius: 25px;
        border: 1px solid rgba(16, 185, 129, 0.2);
    }
    [data-testid="stText"] { font-size: 1.5rem !important; line-height: 1.4 !important; color: #94a3b8; }
    div:has(> .stMarkdown:empty) { display: none; }
</style>
""", unsafe_allow_html=True)

# IA
api_key = None
try:
    if "GEMINI_API_KEY" in st.secrets:
        api_key = st.secrets["GEMINI_API_KEY"]
except Exception:
    pass

if not api_key and os.path.exists("GEMINI_KEY.txt"):
    try:
        with open("GEMINI_KEY.txt", "r") as f:
            api_key = f.read().strip()
    except Exception:
        pass

if api_key:
    genai.configure(api_key=api_key)
else:
    st.error("🔑 No se encontró la API Key válida. Configura 'GEMINI_API_KEY' en tu carpeta.")
    st.stop()

guia_tecnica = ""
if os.path.exists("GUIA_TECNICA_IA.txt"):
    with open("GUIA_TECNICA_IA.txt", "r", encoding="utf-8") as f:
        guia_tecnica = f.read()

model_ia = genai.GenerativeModel("gemini-2.5-flash")

def aplicar_limpieza_interna(df, col_target, reglas_dict=None):
    transformador = Transformar_Df(df, col_target=col_target)
    reporte = transformador.Clean_All_Rows(reglas_dict=reglas_dict)
    return transformador, transformador.df, transformador.y

# Orquestador_modelos_interno
def orquestador_modelos_interno(X, y, tipo_modelo):

    m = tipo_modelo.lower()

    # Regresión
    if 'random_forest' in m and ('regres' in m or 'regressor' in m):
        json_res, modelo, cols = Random_Forest_Regressor(X, y)
    elif 'xgboost' in m and ('regres' in m or 'regressor' in m):
        json_res, modelo, cols = XGBoost_Regressor(X, y)
    elif 'lineal' in m or ('regres' in m and 'logis' not in m and 'random' not in m and 'xgb' not in m):
        json_res, modelo, cols = Regresion_lineal(X, y)

    # Clasificación
    elif 'random_forest' in m or ('random' in m and 'forest' in m):
        json_res, modelo, cols = Random_Forest_Clasificador(X, y)
    elif 'xgboost' in m or 'xgb' in m:
        json_res, modelo, cols = XGBoost_Clasificador(X, y)
    elif 'arbol' in m or 'decision' in m or 'tree' in m:
        json_res, modelo, cols = Arbol_decision(X, y)
    elif 'logis' in m:
        json_res, modelo, cols = Regresion_logistica(X, y)

    # Clustering
    elif 'kmeans' in m or 'cluster' in m or 'agrupamiento' in m:
        json_res, modelo, cols = KMeans_Clustering(X)

    else:
        raise ValueError(
            f"ERROR: El modelo solicitado '{tipo_modelo}' no coincide con "
            f"ninguna de las arquitecturas implementadas en el orquestador."
        )

    return modelo, json.loads(json_res), cols

defaults = {
    "phase": "CARGA", "df": None, "proposal": None,
    "config_pipeline": None, "results": None,
    "cleaner": None, "report_html": None,
    "wh_analysis_id": None,
    "execution_time": None,
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v


# prompt optimizado
def get_ia_proposal(df, feedback=""):
    dtypes   = df.dtypes.apply(str).to_dict()
    nulls    = df.isnull().sum().to_dict()
    n_rows, n_cols = df.shape
    sample_vals = df.head(3).to_dict(orient="list")

    narrativa = (
        "EMPIEZA DICIENDO: 'Entendido, he procesado tus ajustes. "
        "Este es el nuevo plan estratégico...'"
        if feedback else
        "Presenta un plan inicial de ciencia de datos con visión de negocio."
    )

    modelos_disponibles = (
        "regresion_lineal | random_forest_regressor | xgboost_regressor | "
        "regresion_logistica | arbol_decision | random_forest_clasificador | "
        "xgboost_clasificador | kmeans_clustering"
    )

    prompt = f"""
Eres un Consultor Senior de Ciencia de Datos y Estrategia de Negocio.
Hablas en español. Eres directo, preciso y orientado a resultados de negocio.

═══ CONTEXTO DEL DATASET ═══
- Filas: {n_rows} | Columnas: {n_cols}
- Tipos de datos: {json.dumps(dtypes, ensure_ascii=False)}
- Valores Nulos por columna: {json.dumps(nulls, ensure_ascii=False)}
- Muestra (3 filas): {json.dumps(sample_vals, ensure_ascii=False, default=str)}

═══ GUÍA TÉCNICA INTERNA (OBLIGATORIA) ═══
{guia_tecnica}

═══ INSTRUCCIÓN DEL USUARIO ═══
{feedback if feedback else 'Análisis inicial. Sin instrucciones previas del usuario.'}

═══ TU MISIÓN ═══
{narrativa}

PASO 1 — DIAGNÓSTICO DE NEGOCIO (3-5 oraciones):
  - ¿Qué problema de negocio resuelve este dataset?
  - ¿Por qué el modelo elegido es el más adecuado?
  - ¿Qué variable objetivo tiene más sentido predecir?

PASO 2 — ESTRATEGIA DE LIMPIEZA:
  - Explica brevemente el tratamiento de nulos y variables propuesto.
  - Justifica las columnas que se eliminarán o transformarán.

PASO 3 — EXPECTATIVAS DE RENDIMIENTO:
  - ¿Qué métricas esperamos y por qué?
  - ¿Cuáles son los riesgos del modelo elegido con estos datos?

PASO 4 — JSON DE CONFIGURACIÓN (OBLIGATORIO AL FINAL):
  Modelos disponibles: {modelos_disponibles}

```json
{{
  "col_target": "NOMBRE_COLUMNA_OBJETIVO",
  "tipo_modelo": "UNO_DE_LOS_MODELOS_DISPONIBLES",
  "metodos_imputacion": {{
    "NOMBRE_COLUMNA": {{
      "metodo": "mean | median | mode | drop-column | drop-values",
      "Dummies": true | false,
      "TargetEncoding": false,
      "WOE": false,
      "Ordinal": false,
      "orden": []
    }}
  }}
}}
```

RESTRICCIONES CRÍTICAS:
- col_target DEBE ser una columna real del dataset.
- tipo_modelo DEBE ser exactamente uno de los valores listados.
- NUNCA incluyas col_target dentro de metodos_imputacion.
- NUNCA menciones nombres de funciones Python internas.
- El JSON debe ser válido y estar AL FINAL de tu respuesta.
"""
    response = model_ia.generate_content(prompt)
    return response.text

st.title("⚡ Data Mining Autopilot")
st.text("Automatización del preprocesamiento de datos y entrenamiento de modelos de Machine Learning")

# Fase: Carga
if st.session_state.phase == "CARGA":
    uploaded_file = st.file_uploader("Sube tu archivo", type=["csv", "xlsx"])
    if uploaded_file:
        with st.spinner("🚀 Cargando y procesando datos..."):
            if uploaded_file.name.endswith(".csv"):
                try:
                    st.session_state.df = pd.read_csv(uploaded_file, encoding='utf-8')
                except UnicodeDecodeError:
                    uploaded_file.seek(0)
                    st.session_state.df = pd.read_csv(uploaded_file, encoding='latin1')
            else:
                st.session_state.df = pd.read_excel(uploaded_file)
            st.session_state.phase = "PROPUESTA"
            st.rerun()

# Fase: Propuesta
elif st.session_state.phase == "PROPUESTA":
    tab1, tab2 = st.tabs(["📝 Estrategia de IA", "📊 Reporte de Datos"])

    with tab1:
        if not st.session_state.proposal:
            with st.spinner("🧠 El agente está diseñando la estrategia inicial..."):
                st.session_state.proposal = get_ia_proposal(st.session_state.df)

        json_match = re.search(r"```json\s*(\{.*?\})\s*```",
                               st.session_state.proposal, re.DOTALL)
        json_str   = json_match.group(1) if json_match else "{}"
        explicacion = re.sub(r"```json.*?```", "", st.session_state.proposal, flags=re.DOTALL).strip()

        col1, col2 = st.columns([1.6, 1.4], gap="large")

        with col1:
            st.markdown("### 📝 Propuesta Estratégica")
            st.markdown(f'<div class="equal-height-box">{explicacion}</div>',
                        unsafe_allow_html=True)

        with col2:
            st.markdown("### 🛠️ Configuración Técnica")
            try:
                conf_data = json.loads(json_str)
                st.markdown(f"**🎯 Target:** `{conf_data.get('col_target')}`")
                st.markdown(f"**🧠 Modelo:** `{conf_data.get('tipo_modelo')}`")
                st.markdown("### 📊 Tratamiento de variables")
                rules = conf_data.get('metodos_imputacion',
                                      conf_data.get('reglas_dict', {}))
                if rules:
                    table_data = [
                        {
                            "Columna": col,
                            "Tratamiento": params.get("metodo", "—"),
                            "Dummies": "✅" if params.get("Dummies") else "❌",
                            "Target Enc.": "✅" if params.get("TargetEncoding") else "❌",
                        }
                        for col, params in rules.items()
                    ]
                    st.table(pd.DataFrame(table_data))
            except Exception:
                st.error("Error al parsear la configuración JSON")

            if st.button("🚀 Ejecutar Pipeline", use_container_width=True):
                st.session_state.config_pipeline = json.loads(json_str)
                st.session_state.phase = "EJECUCION"
                st.rerun()

        st.markdown("### 🎯 Refinar Plan")
        feedback_val = st.text_area(
            "Añade tus ajustes o contexto adicional:",
            placeholder="Ej: No utilices la columna X, cambia el modelo...",
            label_visibility="collapsed"
        )
        if st.button("🔄 Actualizar Propuesta Estratégica", use_container_width=True):
            instruction = (
                f"Usa este JSON como base: {json_str}. "
                f"Aplica SOLO los cambios del feedback: {feedback_val}"
            )
            with st.spinner("🔄 Ajustando estrategia..."):
                st.session_state.proposal = get_ia_proposal(st.session_state.df, instruction)
                st.rerun()

    with tab2:
        st.markdown("### 📊 Reporte Exploratorio Detallado")
        if not st.session_state.report_html:
            with st.spinner("Generando reporte interactivo de calidad de datos..."):
                st.session_state.report_html = AnalizarDatos(st.session_state.df)
        components.html(st.session_state.report_html, height=1000, scrolling=True)

# Fase: Ejecución
elif st.session_state.phase == "EJECUCION":
    conf = st.session_state.config_pipeline
    try:
        target   = conf.get('col_target', conf.get('target'))
        reglas   = conf.get('metodos_imputacion', conf.get('reglas_dict', {}))
        modelo_t = conf.get('tipo_modelo', conf.get('modelo', 'regresion_logistica'))

        t_inicio = time.time()

        with st.spinner("🧹 Iniciando Limpieza Automatizada..."):
            cleaner, X, y = aplicar_limpieza_interna(
                st.session_state.df, col_target=target, reglas_dict=reglas)
            st.session_state.cleaner = cleaner
            try:
                df_export = pd.concat([X, y], axis=1)
                df_export.to_excel("dataset_limpio.xlsx", index=False)
                st.success("✅ Dataset limpio guardado como 'dataset_limpio.xlsx'")
            except Exception as e:
                st.warning(f"Error al guardar Excel: {e}")

        with st.spinner(f"🧠 Optimizando y Entrenando {modelo_t}..."):
            X_numeric = X.select_dtypes(include=['number'])
            cols_eliminadas = set(X.columns) - set(X_numeric.columns)
            if cols_eliminadas:
                st.warning(f"⚠️ Columnas no numéricas omitidas: {list(cols_eliminadas)}")

            modelo_obj, metricas, cols = orquestador_modelos_interno(
                X_numeric, y, tipo_modelo=modelo_t)
            st.session_state.results = {
                "modelo": modelo_obj, "metricas": metricas, "cols": cols
            }

        t_fin = time.time()
        st.session_state.execution_time = round(t_fin - t_inicio, 2)

        # NUEVO - Registrar en ETL Warehouse
        try:
            task_map = {
                "regresion": "regresion", "lineal": "regresion",
                "logis": "clasificacion", "arbol": "clasificacion",
                "forest": "clasificacion", "xgb": "clasificacion",
                "kmeans": "clustering", "cluster": "clustering",
            }
            task_type = "clasificacion"
            for kw, ttype in task_map.items():
                if kw in modelo_t.lower():
                    task_type = ttype
                    break

            an_id = registrar_resultado_en_warehouse(
                df=st.session_state.df,
                dataset_name=f"dataset_{target}",
                model_name=modelo_t,
                model_type=task_type,
                target_col=target or "",
                task_type=task_type,
                metrics_json=metricas,
                cols_used=cols,
                ai_conclusion="",
                execution_secs=st.session_state.execution_time,
                hyperparams=metricas.get("mejores_hiperparametros", {}),
            )
            st.session_state.wh_analysis_id = an_id
        except Exception as e:
            st.warning(f"⚠️ ETL Warehouse: {e}")

        st.session_state.phase = "RESULTADOS"
        st.rerun()

    except Exception as e:
        st.error(f"Error en el Pipeline: {e}")
        if st.button("Reintentar Propuesta"):
            st.session_state.phase = "PROPUESTA"
            st.rerun()

# Fase: Resultados
elif st.session_state.phase == "RESULTADOS":
    res = st.session_state.results
    st.balloons()

    tab_res, tab_pred, tab_dwh = st.tabs([
        "🏆 Resultados del Modelo",
        "🎯 Predicción",
        "📦 Data Warehouse"
    ])

    with tab_res:
        st.markdown("### 🧠 Interpretación Estratégica del Autopilot")

        # Prompt de interpretación optimizado
        conf = st.session_state.config_pipeline or {}
        target_col = conf.get('col_target', conf.get('target', 'Desconocida'))
        tipo_modelo = conf.get('tipo_modelo', 'modelo')

        metricas_fmt = json.dumps(res['metricas'], indent=2, ensure_ascii=False)

        interp_prompt = f"""
Eres un Consultor Senior de Ciencia de Datos. Tu cliente NO es técnico.
Debes traducir resultados estadísticos a lenguaje de negocio claro y accionable.
Hablas en español. Sé conciso, directo y usa bullet points donde ayude la claridad.

═══ CONTEXTO DEL ANÁLISIS ═══
- Variable Objetivo: {target_col}
- Modelo Entrenado: {tipo_modelo}
- Variables Utilizadas: {', '.join(res['cols'])}
- Tiempo de Ejecución: {st.session_state.execution_time or '?'} segundos

═══ MÉTRICAS DEL MODELO ═══
{metricas_fmt}

═══ TU ANÁLISIS DEBE INCLUIR ═══

**1. 📊 Diagnóstico del Rendimiento**
- Califica el modelo: ¿excelente, bueno, aceptable o insuficiente? ¿Por qué?
- Traduce cada métrica a impacto concreto de negocio.
  Ejemplo: "Un F1-Score de 0.87 significa que de cada 100 casos críticos,
  el modelo detecta correctamente 87 y falla en 13."

**2. 🔍 Variables Más Influyentes**
- ¿Cuáles son los factores que más determinan el resultado?
- ¿Hay alguna variable que sorprenda o que merezca investigación adicional?

**3. ⚠️ Riesgos y Limitaciones**
- ¿Hay señales de overfitting o underfitting?
- ¿Qué sesgos podría tener el modelo con estos datos?

**4. 🚀 Recomendaciones Estratégicas (Top 3)**
- Acciones concretas que el negocio puede tomar con estos resultados.
- ¿Qué datos adicionales mejorarían significativamente el modelo?
- ¿Cuál es el siguiente paso recomendado?

**5. 🎯 Veredicto Final**
- En 2-3 oraciones: ¿Está el modelo listo para producción? ¿Con qué condiciones?
"""
        with st.spinner("Analizando resultados con IA..."):
            ai_interpretation = model_ia.generate_content(interp_prompt).text
            st.markdown(ai_interpretation)

            if st.session_state.wh_analysis_id:
                try:
                    wh = ETLWarehouse()
                    with wh._conn() as conn:
                        conn.execute(
                            "UPDATE fact_analysis SET ai_conclusion=? WHERE analysis_id=?",
                            (ai_interpretation[:2000], st.session_state.wh_analysis_id)
                        )
                except Exception:
                    pass

        st.markdown("---")
        st.markdown(f"**⏱️ Tiempo de entrenamiento:** `{st.session_state.execution_time}s`")
        st.write(f"**Variables procesadas:** {', '.join(res['cols'])}")

        if st.button("🔄 Iniciar Nuevo Proyecto"):
            st.session_state.clear()
            st.rerun()

    with tab_pred:
        st.markdown("### 🎯 Realizar Predicciones con el Modelo Entrenado")

        pred_mode = st.radio(
            "Modo de entrada:",
            ["📝 Lenguaje Natural", "📄 CSV Manual", "📁 Subir CSV"],
            horizontal=True
        )
        st.markdown("---")

        # Opción 1: Lenguaje Natural
        if pred_mode == "📝 Lenguaje Natural":
            st.markdown(
                "Describe el registro en lenguaje natural y la IA lo convertirá "
                "al formato correcto automáticamente."
            )
            ejemplo_cols = ", ".join([f"`{c}`" for c in res['cols'][:5]])
            nl_input = st.text_area(
                f"Describe el caso (variables relevantes: {ejemplo_cols}...):",
                placeholder=(
                    "Ej: 'El cliente tiene 35 años, ingresos de 45000, "
                    "antigüedad de 2 años y saldo de 12000'"
                ),
                height=120
            )
            if st.button("🤖 Interpretar y Predecir", use_container_width=True):
                if nl_input.strip():
                    nl_prompt = f"""
Eres un extractor de datos estructurados. Tu tarea es convertir texto en lenguaje
natural a un JSON con exactamente estas claves (y solo estas):
{json.dumps(res['cols'])}

Texto del usuario: "{nl_input}"

Reglas:
- Si un valor no está mencionado, usa null.
- Los valores numéricos deben ser números (sin comillas).
- Devuelve SOLO el JSON, sin explicaciones, sin markdown, sin texto adicional.
Ejemplo de salida válida: {{"edad": 35, "ingresos": 45000, "saldo": 12000}}
"""
                    with st.spinner("🤖 Interpretando lenguaje natural..."):
                        try:
                            raw = model_ia.generate_content(nl_prompt).text
                            raw_clean = re.sub(r"```json|```", "", raw).strip()
                            datos_dict = json.loads(raw_clean)

                            st.markdown("**🔍 Datos extraídos:**")
                            st.json(datos_dict)

                            df_nl = pd.DataFrame([datos_dict])
                            df_nl = df_nl.reindex(columns=res['cols'])
                            df_nl = df_nl.apply(pd.to_numeric, errors='coerce')
                            df_nl.fillna(df_nl.mean(), inplace=True)

                            pred = res['modelo'].predict(df_nl[res['cols']])
                            prob_txt = ""
                            if hasattr(res['modelo'], 'predict_proba'):
                                prob = res['modelo'].predict_proba(df_nl[res['cols']])[0]
                                prob_txt = f"  |  Confianza: `{max(prob):.1%}`"

                            st.success(f"🎯 **Resultado Predicho:** `{pred[0]}`{prob_txt}")

                            # Guardar predicción en DWH
                            if st.session_state.wh_analysis_id:
                                try:
                                    ETLWarehouse().save_predictions(
                                        st.session_state.wh_analysis_id,
                                        [datos_dict], [pred[0]]
                                    )
                                except Exception:
                                    pass

                        except json.JSONDecodeError:
                            st.error("❌ No se pudo extraer un JSON válido del texto. Intenta ser más específico.")
                        except Exception as e:
                            st.error(f"❌ Error en predicción: {e}")
                else:
                    st.warning("Escribe una descripción para continuar.")

        # Opción 2: CSV Manual
        elif pred_mode == "📄 CSV Manual":
            cols_str = ",".join(res['cols'])
            st.markdown(f"**Columnas esperadas:** `{cols_str}`")
            st.caption("Pega los valores separados por comas (una fila por línea).")
            csv_input = st.text_area(
                "Entrada CSV:",
                value=cols_str + "\n",
                height=150,
                placeholder=f"{cols_str}\nvalor1,valor2,valor3"
            )
            if st.button("⚡ Predecir desde CSV", use_container_width=True):
                try:
                    from io import StringIO
                    df_csv = pd.read_csv(StringIO(csv_input))
                    df_csv = df_csv.reindex(columns=res['cols'])
                    df_csv = df_csv.apply(pd.to_numeric, errors='coerce')
                    df_csv.fillna(df_csv.mean(), inplace=True)

                    preds = res['modelo'].predict(df_csv[res['cols']])
                    df_csv['🎯 PREDICCIÓN'] = preds

                    if hasattr(res['modelo'], 'predict_proba'):
                        probs = res['modelo'].predict_proba(df_csv[res['cols']])
                        df_csv['📊 CONFIANZA'] = [f"{max(p):.1%}" for p in probs]

                    st.success(f"✅ {len(preds)} predicciones generadas:")
                    st.dataframe(df_csv, use_container_width=True)

                    if st.session_state.wh_analysis_id:
                        try:
                            records = df_csv.drop(
                                columns=['🎯 PREDICCIÓN', '📊 CONFIANZA'],
                                errors='ignore'
                            ).to_dict(orient='records')
                            ETLWarehouse().save_predictions(
                                st.session_state.wh_analysis_id, records, preds.tolist()
                            )
                        except Exception:
                            pass

                except Exception as e:
                    st.error(f"❌ Error: {e}")

        # Opción 3: Subir CSV
        elif pred_mode == "📁 Subir CSV":
            pred_file = st.file_uploader(
                "Sube un CSV con los datos a predecir",
                type=["csv"], key="pred_uploader"
            )
            if pred_file:
                try:
                    df_up = pd.read_csv(pred_file)
                    st.markdown(f"**{len(df_up)} registros cargados.** Preview:")
                    st.dataframe(df_up.head(5), use_container_width=True)

                    if st.button("⚡ Predecir Archivo Completo", use_container_width=True):
                        df_up_sel = df_up.reindex(columns=res['cols'])
                        df_up_sel = df_up_sel.apply(pd.to_numeric, errors='coerce')
                        df_up_sel.fillna(df_up_sel.mean(), inplace=True)

                        preds = res['modelo'].predict(df_up_sel[res['cols']])
                        df_up['🎯 PREDICCIÓN'] = preds

                        if hasattr(res['modelo'], 'predict_proba'):
                            probs = res['modelo'].predict_proba(df_up_sel[res['cols']])
                            df_up['📊 CONFIANZA'] = [f"{max(p):.1%}" for p in probs]

                        st.success(f"✅ {len(preds)} predicciones generadas:")
                        st.dataframe(df_up, use_container_width=True)

                        csv_out = df_up.to_csv(index=False).encode('utf-8')
                        st.download_button(
                            "⬇️ Descargar Predicciones CSV",
                            csv_out,
                            "predicciones_autopilot.csv",
                            "text/csv",
                            use_container_width=True
                        )

                        if st.session_state.wh_analysis_id:
                            try:
                                records = df_up_sel.to_dict(orient='records')
                                ETLWarehouse().save_predictions(
                                    st.session_state.wh_analysis_id,
                                    records, preds.tolist()
                                )
                            except Exception:
                                pass

                except Exception as e:
                    st.error(f"❌ Error al procesar archivo: {e}")

    with tab_dwh:
        st.markdown("### 📦 Data Warehouse — Historial de Análisis")
        st.caption(
            "Star Schema SQLite en `dwh/warehouse.db` — "
            "Persiste entre sesiones de la app."
        )

        try:
            wh = ETLWarehouse()

            col_a, col_b = st.columns(2)
            with col_a:
                st.markdown("#### 📋 Últimos Análisis")
                df_hist = wh.get_analysis_history(limit=20)
                if not df_hist.empty:
                    st.dataframe(df_hist, use_container_width=True)
                else:
                    st.info("No hay análisis registrados aún.")

            with col_b:
                st.markdown("#### 🗄️ Datasets Registrados")
                df_ds = wh.get_dataset_stats()
                if not df_ds.empty:
                    st.dataframe(df_ds, use_container_width=True)
                else:
                    st.info("No hay datasets registrados.")

            st.markdown("#### 🏆 Mejores Modelos")
            task_filter = st.selectbox(
                "Tipo de tarea:", ["clasificacion", "regresion", "clustering"])
            df_best = wh.get_best_models(task_type=task_filter, top_n=5)
            if not df_best.empty:
                st.dataframe(df_best, use_container_width=True)
            else:
                st.info(f"No hay modelos de '{task_filter}' registrados.")

            st.markdown("---")
            if st.button("📊 Exportar Star Schema a Excel", use_container_width=True):
                with st.spinner("Exportando todas las tablas..."):
                    path = wh.export_star_schema_excel()
                    with open(path, "rb") as f:
                        st.download_button(
                            "⬇️ Descargar Excel del Data Warehouse",
                            f.read(),
                            "star_schema_autopilot.xlsx",
                            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                            use_container_width=True
                        )

        except Exception as e:
            st.error(f"Error al conectar con el Data Warehouse: {e}")

        if st.button("🔄 Iniciar Nuevo Proyecto", key="nuevo_dwh"):
            st.session_state.clear()
            st.rerun()
