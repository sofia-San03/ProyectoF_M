import json
import os
import re
from io import StringIO

import pandas as pd
import numpy as np
import streamlit as st
import streamlit.components.v1 as components

from CODIGO.CargarDatos import AnalizarDatos
from CODIGO.Funcionalidades import (
    MODELOS_DISPONIBLES,
    extraer_configuracion_pipeline,
    obtener_metricas_esperadas,
    ocultar_woe_interfaz,
    validar_tipo_problema,
    aplicar_limpieza_interna,
    orquestador_modelos_interno,
    es_modelo_clustering,
    es_modelo_redes_neuronales,
    es_modelo_knn,
    es_modelo_arbol,
    es_modelo_regresion_logistica,
    es_modelo_credit_scoring,
    get_ia_proposal,
    load_css,
    iniciar_chat,
    interpretar_resultados,
    interpretar_resultados_perfilarDatos,
    texto_a_dataframe
)

st.set_page_config(page_title="Autopilot", page_icon="⚡", layout="wide")
load_css("styles.css")

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
if "chat_session" not in st.session_state:
    st.session_state.chat_session = iniciar_chat()
if "messages_propuesta" not in st.session_state:
    st.session_state.messages_propuesta = []
if "chat_resultados_session" not in st.session_state:
    st.session_state.chat_resultados_session = iniciar_chat()
if "messages_resultados" not in st.session_state:
    st.session_state.messages_resultados = []
if "data_dict_content" not in st.session_state:
    st.session_state.data_dict_content = None
if "data_dict_name" not in st.session_state:
    st.session_state.data_dict_name = None
if "last_uploaded_file" not in st.session_state:
    st.session_state.last_uploaded_file = None
if "last_uploaded_dict" not in st.session_state:
    st.session_state.last_uploaded_dict = None


st.title("  Data mining Autopilot ")
st.text("Automatización del preprocesamiento de datos y entrenamiento de modelos de machine learning")

# Navegación Lateral (Sidebar)
opciones_nav = ["Carga de Datos"]
if st.session_state.df is not None:
    opciones_nav.append("Propuesta")
if st.session_state.results is not None:
    opciones_nav.append("Resultados y Predicción")

map_phase_to_nav = {
    "CARGA": "Carga de Datos",
    "PROPUESTA": "Propuesta",
    "EJECUCION": "Propuesta",
    "RESULTADOS": "Resultados y Predicción"
}
map_nav_to_phase = {
    "Carga de Datos": "CARGA",
    "Propuesta": "PROPUESTA",
    "Resultados y Predicción": "RESULTADOS"
}

with st.sidebar:
    st.markdown("""
    <div style="display:flex; align-items:center; gap: 10px; margin-bottom: 20px;">
        <div style="width: 40px; height: 40px; border-radius: 8px; background: rgba(78, 222, 163, 0.1); display: flex; align-items: center; justify-content: center; border: 1px solid rgba(78, 222, 163, 0.2);">
            <span style="font-size: 20px;">📑</span>
        </div>
        <div>
            <h2 style="margin:0; font-size: 14px; color: #4edea3 !important; padding:0; background:none; border:none; box-shadow:none;">Data Mining Autopilot</h2>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("<p style='margin-bottom: 5px; color: #bbcabf; font-weight: 600; font-family: Sora, sans-serif; font-size: 0.9rem; text-transform: uppercase; letter-spacing: 1px;'>Navegación</p>", unsafe_allow_html=True)
    
    current_active_nav = map_phase_to_nav.get(st.session_state.phase, "Carga de Datos")
    
    for opcion in opciones_nav:
        is_active = (opcion == current_active_nav)
        if st.button(opcion, key=f"nav_btn_{opcion}", type="primary" if is_active else "secondary", use_container_width=True):
            if st.session_state.phase != "EJECUCION":
                st.session_state.phase = map_nav_to_phase[opcion]
                st.rerun()
        
    st.markdown("<br><br>", unsafe_allow_html=True)
    if st.button("🔴 Reiniciar Sistema", use_container_width=True):
        st.session_state.clear()
        st.rerun()

if st.session_state.phase == "CARGA":
    st.markdown("<h2 style='text-align:center; border:none; background:none; box-shadow:none;'>Inicia el Futuro de tus Datos</h2>", unsafe_allow_html=True)
    st.markdown("<p style='text-align:center; color:#bbcabf'>Sube tus datasets para comenzar el procesamiento neuronal.</p><br>", unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        uploaded_file = st.file_uploader("Sube tu archivo", type=["csv", "xlsx", "json"])
        if uploaded_file:
            file_key = f"df_loaded_{uploaded_file.name}_{uploaded_file.size}"
            if st.session_state.get("last_uploaded_file") != file_key:
                st.session_state.proposal = None
                st.session_state.config_pipeline = None
                st.session_state.results = None
                st.session_state.cleaner = None
                st.session_state.messages_propuesta = []
                st.session_state.messages_resultados = []
                st.session_state.chat_session = iniciar_chat()
                st.session_state.chat_resultados_session = iniciar_chat()
                
                with st.spinner("Cargando y procesando datos..."):
                    try:
                        if uploaded_file.name.endswith(".csv"):
                            try:
                                df_temp = pd.read_csv(uploaded_file, encoding="utf-8")
                            except UnicodeDecodeError:
                                uploaded_file.seek(0)
                                df_temp = pd.read_csv(uploaded_file, encoding="latin1")
                        elif uploaded_file.name.endswith(".json"):
                            try:
                                df_temp = pd.read_json(uploaded_file)
                            except ValueError:
                                uploaded_file.seek(0)
                                df_temp = pd.read_json(uploaded_file, lines=True)
                        else:
                            df_temp = pd.read_excel(uploaded_file)
                        st.session_state.df = df_temp
                        st.session_state.last_uploaded_file = file_key
                        st.success(f"✅ Dataset cargado con éxito: **{uploaded_file.name}**")
                    except Exception as e:
                        st.error(f"Error al procesar el archivo: {e}")
            else:
                st.success(f"✅ Dataset activo: **{uploaded_file.name}** ({st.session_state.df.shape[0]} filas, {st.session_state.df.shape[1]} columnas)")
        else:
            st.session_state.df = None
            st.session_state.last_uploaded_file = None

    with col2:
        uploaded_dict = st.file_uploader("Sube tus diccionarios", type=["csv", "xlsx", "json", "txt"], key="dict_uploader")
        if uploaded_dict:
            dict_key = f"dict_loaded_{uploaded_dict.name}_{uploaded_dict.size}"
            if st.session_state.get("last_uploaded_dict") != dict_key:
                with st.spinner("Procesando diccionario de datos..."):
                    try:
                        if uploaded_dict.name.endswith(".csv"):
                            try:
                                df_dict = pd.read_csv(uploaded_dict, encoding="utf-8")
                            except UnicodeDecodeError:
                                uploaded_dict.seek(0)
                                df_dict = pd.read_csv(uploaded_dict, encoding="latin1")
                            st.session_state.data_dict_content = df_dict.to_markdown(index=False)
                        elif uploaded_dict.name.endswith(".xlsx"):
                            df_dict = pd.read_excel(uploaded_dict)
                            st.session_state.data_dict_content = df_dict.to_markdown(index=False)
                        elif uploaded_dict.name.endswith(".json"):
                            try:
                                df_dict = pd.read_json(uploaded_dict)
                            except ValueError:
                                uploaded_dict.seek(0)
                                df_dict = pd.read_json(uploaded_dict, lines=True)
                            st.session_state.data_dict_content = df_dict.to_markdown(index=False)
                        elif uploaded_dict.name.endswith(".txt"):
                            st.session_state.data_dict_content = uploaded_dict.read().decode("utf-8", errors="ignore")
                        
                        st.session_state.data_dict_name = uploaded_dict.name
                        st.session_state.last_uploaded_dict = dict_key
                        st.success(f"✅ Diccionario de datos cargado con éxito: **{uploaded_dict.name}**")
                    except Exception as e:
                        st.error(f"Error al procesar el diccionario de datos: {e}")
            else:
                st.success(f"✅ Diccionario de datos activo: **{st.session_state.data_dict_name}**")
            
            if st.button("Eliminar Diccionario", use_container_width=True):
                st.session_state.data_dict_content = None
                st.session_state.data_dict_name = None
                st.session_state.last_uploaded_dict = None
                st.rerun()
        else:
            st.session_state.data_dict_content = None
            st.session_state.data_dict_name = None
            st.session_state.last_uploaded_dict = None

    if st.session_state.df is not None:
        st.markdown("<br>", unsafe_allow_html=True)
        col_btn_left, col_btn_center, col_btn_right = st.columns([1, 2, 1])
        with col_btn_center:
            if st.button("🚀 Confirmar Datos", use_container_width=True, type="primary"):
                st.session_state.phase = "PROPUESTA"
                st.rerun()

elif st.session_state.phase == "PROPUESTA":
    tab1, tab2 = st.tabs(["Estrategia de IA", "Reporte de Datos"])

    with tab1:
        if not st.session_state.messages_propuesta:
            with st.spinner("El agente esta disenando la estrategia inicial..."):
                initial_proposal = get_ia_proposal(
                    st.session_state.chat_session,
                    st.session_state.df,
                    is_initial=True,
                    diccionario_datos=st.session_state.data_dict_content
                )
                st.session_state.messages_propuesta.append({"role": "assistant", "content": initial_proposal})

        json_str = "{}"
        for msg in reversed(st.session_state.messages_propuesta):
            if msg["role"] == "assistant":
                json_match = re.search(r"```json\s*(\{.*?\})\s*```", msg["content"], re.DOTALL)
                if json_match:
                    json_str = json_match.group(1)
                    break

        st.markdown("### Chat Estratégico")
        
        chat_container = st.container(height=450)
        with chat_container:
            for msg in st.session_state.messages_propuesta:
                with st.chat_message(msg["role"]):
                    content_to_show = re.sub(r"```json.*?```", "", msg["content"], flags=re.DOTALL)
                    content_to_show = re.sub(r"(?i)-{3,}", "", content_to_show)
                    content_to_show = re.sub(r"(?i)#+.*?Configuraci[oó]n.*?(?:JSON|Preprocesamiento|T[eé]cnica).*?\n?", "", content_to_show)
                    content_to_show = re.sub(r"(?i)#+.*?Pipeline.*?\n?", "", content_to_show)
                    
                    content_to_show = content_to_show.strip()
                    if content_to_show:
                        st.markdown(content_to_show)
        
        if user_input := st.chat_input("Escribe tus dudas o pide ajustes al plan..."):
            st.session_state.messages_propuesta.append({"role": "user", "content": user_input})
            with st.spinner("Procesando tu solicitud..."):
                instruction = f"Usa este json como base para modificaciones si aplica: {json_str}. El usuario dice: {user_input}"
                response = get_ia_proposal(
                    st.session_state.chat_session,
                    st.session_state.df,
                    feedback=instruction,
                    diccionario_datos=st.session_state.data_dict_content
                )
                st.session_state.messages_propuesta.append({"role": "assistant", "content": response})
            st.rerun()

        st.markdown("---")
        st.markdown("### Configuracion Tecnica")
        try:
            conf_data = json.loads(json_str) if json_str != "{}" else {}
            target_detectado, reglas_detectadas, modelo_detectado, es_pca_detectado, n_clusters_detectado = extraer_configuracion_pipeline(conf_data)
            opciones_modelos = list(MODELOS_DISPONIBLES.keys())
            modelo_seleccionado = st.selectbox(
                "Modelo detectado",
                opciones_modelos,
                index=opciones_modelos.index(modelo_detectado) if modelo_detectado in opciones_modelos else 0,
            )
            conf_data["modelo"] = modelo_seleccionado
            conf_data["tipo_modelo"] = modelo_seleccionado

            if MODELOS_DISPONIBLES[modelo_seleccionado]["tipo_problema"] == "clustering":
                n_clusters_mostrado = str(n_clusters_detectado) if n_clusters_detectado else "AUTO"
                st.markdown(f"**Número de Clusters:** `{n_clusters_mostrado}`")
                target_validacion = None
            else:
                target_mostrado = target_detectado if target_detectado else "⚠️ Pendiente de definir"
                st.markdown(f"**Variable Objetivo:** `{target_mostrado}`")
                target_validacion = target_detectado
            st.markdown(f"**Modelo sugerido:** `{modelo_seleccionado}`")
            st.markdown(f"**Metricas esperadas:** `{', '.join(obtener_metricas_esperadas(modelo_seleccionado))}`")

            es_valido, mensaje_validacion = validar_tipo_problema(
                st.session_state.df, target_validacion, modelo_seleccionado
            )
            if es_valido:
                st.success(f"Validacion del problema: {mensaje_validacion}")
            else:
                st.error(f"Validacion del problema: {mensaje_validacion}")

            # --- LIMPIEZA DE RESULTADOS PREVIOS AL CAMBIAR MODELO O REGLAS ---
            actual_reglas = conf_data.get("reglas_dict", {})
            if st.session_state.results is not None:
                modelo_cambio = st.session_state.results.get("tipo_modelo") != modelo_seleccionado
                reglas_cambio = st.session_state.results.get("reglas") != actual_reglas
                
                if modelo_cambio or reglas_cambio:
                    st.session_state.results = None
                    st.session_state.messages_resultados = []
                    st.session_state.chat_resultados_session = iniciar_chat()

            puede_ejecutar = es_valido and MODELOS_DISPONIBLES[modelo_seleccionado]["implementado"]

            st.markdown("#### Tratamiento de nulos y columnas")
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

        col_empty, col_exec = st.columns([3, 1])
        with col_exec:
            if st.button(" Ejecutar Pipeline", use_container_width=True, disabled=not puede_ejecutar, key="btn_ejecutar_pipeline"):
                st.session_state.config_pipeline = conf_data
                st.session_state.phase = "EJECUCION"
                st.rerun()

    with tab2:
        st.markdown("### Reporte Exploratorio Detallado")
        if st.session_state.proposal is None:
            st.markdown("""
                <div style="background-color: #1a2c3d; border-left: 5px solid #00f2fe; padding: 20px; border-radius: 10px; margin: 20px 0;">
                    <span style="font-size: 20px;">⏳</span> 
                    <span style="color: #00f2fe; font-weight: 500; margin-left: 10px;">
                        Por favor espera a que la IA termine de generar la propuesta estratégica antes de ver el reporte de datos.
                    </span>
                </div>
            """, unsafe_allow_html=True)
        else:
            if not st.session_state.report_html:
                with st.spinner("Generando reporte interactivo de calidad de datos..."):
                    st.session_state.report_html = AnalizarDatos(st.session_state.df)
            components.html(st.session_state.report_html, height=1000, scrolling=True)

elif st.session_state.phase == "EJECUCION":
    conf = st.session_state.config_pipeline
    try:
        import os, shutil
        if os.path.exists("Resultados"):
            shutil.rmtree("Resultados")
        if os.path.exists("MODELOS"):
            shutil.rmtree("MODELOS")
        os.makedirs("Resultados", exist_ok=True)
        os.makedirs("MODELOS", exist_ok=True)

        target, reglas, modelo_t, es_pca, n_clusters_fix = extraer_configuracion_pipeline(conf)

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
                df_export.to_excel("Resultados/dataset_limpio.xlsx", index=False)
                st.success("Dataset limpio guardado como 'Resultados/dataset_limpio.xlsx'")
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

            modelo_obj, metricas, cols = orquestador_modelos_interno(X_numeric, y, tipo_modelo=modelo_t, n_clusters_fix=n_clusters_fix)
            
            # --- PERFILAMIENTO ESTRATÉGICO (SOLO PARA CLUSTERING) ---
            perfiles = {}
            if es_modelo_clustering(modelo_t):
                try:
                    df_temp = st.session_state.df.loc[X.index].copy()
                    if hasattr(modelo_obj, 'labels') and modelo_obj.labels is not None:
                        labels = modelo_obj.labels
                    else:
                        labels = modelo_obj.estimator.labels_ if hasattr(modelo_obj, 'estimator') and modelo_obj.estimator else None
                    
                    if labels is None:
                        raise ValueError("No se pudieron obtener las etiquetas del modelo para el perfilamiento.")
                    
                    df_temp['__GRUPO__'] = labels
                    grupos = df_temp['__GRUPO__'].unique()
                    
                    for g in grupos:
                        df_g = df_temp[df_temp['__GRUPO__'] == g]
                        desc_num = df_g.describe().to_dict()
                        cat_freqs = {}
                        for col in df_g.select_dtypes(include=['object', 'category']).columns:
                            freq = (df_g[col].value_counts(normalize=True) * 100).round(1).to_dict()
                            cat_freqs[col] = freq
                        
                        perfiles[str(g)] = {
                            "estadisticas_numericas": desc_num,
                            "distribucion_categorias": cat_freqs,
                            "tamaño_grupo": len(df_g)
                        }
                except Exception as e:
                    perfiles = {"error": f"No se pudo generar perfilamiento: {e}"}

            st.session_state.results = {
                "modelo": modelo_obj,
                "metricas": metricas,
                "cols": cols,
                "tipo_modelo": modelo_t,
                "target": target,
                "reglas": reglas,
                "es_pca": es_pca,
                "perfiles": perfiles
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

    tipo_modelo = res.get("tipo_modelo", "Regresion_lineal")
    es_clustering_resultado = es_modelo_clustering(tipo_modelo)
    es_redes_resultado = es_modelo_redes_neuronales(tipo_modelo)
    es_knn_resultado = es_modelo_knn(tipo_modelo)
    es_arbol_resultado = es_modelo_arbol(tipo_modelo)
    es_logistica_resultado = es_modelo_regresion_logistica(tipo_modelo)
    es_credit_resultado = es_modelo_credit_scoring(tipo_modelo)
    metricas_interfaz = ocultar_woe_interfaz(res["metricas"])

    tab_res, tab_pred = st.tabs(["📊 Resultados del Modelo", "🔮 Nueva Predicción"])

    with tab_pred:
        st.markdown("### Nueva Predicción")
        if not st.session_state.messages_resultados:
            st.info("⏳ Por favor espera a que la IA termine de analizar los resultados del modelo antes de realizar nuevas predicciones.")
        else:
            tab_text, tab_file = st.tabs(["Ingreso por Chat", "Subir Archivo"])
            
            with tab_text:
                texto_pred = st.text_area("Describe el nuevo registro (ej. 'soy un pasajero de clase 2 hombre...'):")
                if st.button("Predecir desde texto", use_container_width=True):
                    if texto_pred:
                        with st.spinner("Interpretando texto y prediciendo..."):
                            try:
                                dtypes_dict = st.session_state.df.dtypes.apply(lambda x: str(x)).to_dict()
                                df_n = texto_a_dataframe(st.session_state.chat_session, texto_pred, dtypes_dict)
                                
                                st.write("Datos interpretados por el agente:")
                                st.dataframe(df_n)
                                
                                df_p = st.session_state.cleaner.transformar_nueva_tupla(df_n)
                                p = res["modelo"].predict(df_p[res["cols"]])
                                
                                # Redondear si es numérico
                                val_pred = p[0]
                                if isinstance(val_pred, (float, np.float64, np.float32, np.number)):
                                    val_pred = round(float(val_pred), 4)
                                    
                                st.success(f"Resultado predicho: **{val_pred}**")
                            except Exception as e:
                                st.error(f"Error en la prediccion por texto: {e}")
                    else:
                        st.warning("Por favor ingresa una descripcion.")
                        
            with tab_file:
                uploaded_pred = st.file_uploader("Sube tus nuevos datos para predecir", type=["csv", "xlsx", "json"])
                if st.button("Predecir desde archivo", use_container_width=True):
                    if uploaded_pred:
                        with st.spinner("Procesando y prediciendo..."):
                            try:
                                if uploaded_pred.name.endswith(".csv"):
                                    try:
                                        df_n = pd.read_csv(uploaded_pred, encoding="utf-8")
                                    except UnicodeDecodeError:
                                        uploaded_pred.seek(0)
                                        df_n = pd.read_csv(uploaded_pred, encoding="latin1")
                                elif uploaded_pred.name.endswith(".json"):
                                    try:
                                        df_n = pd.read_json(uploaded_pred)
                                    except ValueError:
                                        uploaded_pred.seek(0)
                                        df_n = pd.read_json(uploaded_pred, lines=True)
                                else:
                                    df_n = pd.read_excel(uploaded_pred)
                                    
                                st.write(f"Datos cargados: {df_n.shape[0]} filas")
                                
                                df_p = st.session_state.cleaner.transformar_nueva_tupla(df_n)
                                p = res["modelo"].predict(df_p[res["cols"]])
                                
                                df_res = df_n.copy()
                                # Redondear predicciones si son numéricas
                                if p.dtype.kind in 'fc': # float o complex
                                    df_res['Prediccion'] = np.round(p.astype(float), 4)
                                else:
                                    df_res['Prediccion'] = p
                                st.success("Predicciones completadas.")
                                
                                st.write("Mostrando los primeros 100 registros:")
                                st.dataframe(df_res.head(100))
                                
                                csv = df_res.to_csv(index=False).encode('utf-8')
                                st.download_button(
                                    label="📥 Descargar Resultados",
                                    data=csv,
                                    file_name="predicciones.csv",
                                    mime="text/csv",
                                    use_container_width=True
                                )
                            except Exception as e:
                                st.error(f"Error en la prediccion por archivo: {e}")
                    else:
                        st.warning("Por favor sube un archivo primero.")

    with tab_res:
        st.markdown("### Interpretación y Análisis Estratégico")
        
        if not st.session_state.messages_resultados:
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

                if es_clustering_resultado:
                    explicacion = interpretar_resultados_perfilarDatos(
                        st.session_state.chat_resultados_session,
                        metricas_interfaz,
                        res['cols'],
                        res.get('perfiles', {}),
                        tarea
                    )
                else:
                    explicacion = interpretar_resultados(
                        st.session_state.chat_resultados_session,
                        metricas_interfaz,
                        res['cols'],
                        tarea
                    )
                st.session_state.messages_resultados.append({"role": "assistant", "content": explicacion})
                st.rerun()

        st.write(f"**Variables procesadas:** {', '.join(res['cols'])}")

        if es_clustering_resultado:
            metricas_cluster = metricas_interfaz
            st.markdown("#### Resultados de Clustering")
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
                st.markdown("#### Resultados de Redes Neuronales")
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
                st.markdown("#### Resultados de KNN")
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
                st.markdown("#### Resultados de Arboles")
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
                st.markdown("#### Resultados de Regresion Logistica")
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
                st.markdown("#### Resultados de Credit Scoring")
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

        st.markdown("### Chat de Resultados")
        chat_container_res = st.container(height=400)
        with chat_container_res:
            for msg in st.session_state.messages_resultados:
                with st.chat_message(msg["role"]):
                    st.markdown(msg["content"])

        from CODIGO.Funcionalidades import chat_resultados_ia_stream, build_model_context_prompt
        
        # Preparar contexto para la IA
        model_info = {
            "tipo_modelo":   tipo_modelo,
            "variable_obj":  res.get("target") if res.get("target") else "No requerida",
            "metricas":      res["metricas"].get("metricas_precision", {}),
            "n_registros":   len(st.session_state.df),
            "n_features":    len(res["cols"]),
            "clases":        res["metricas"].get("clases_detectadas", []),
            "encodings":     [k for k, v in res.get("reglas", {}).items() if v.get("Dummies") or v.get("TargetEncoding") or v.get("Ordinal") or v.get("WOE")],
            "pca_aplicado":  res.get("es_pca", False),
            "grid_search":   True,
            "perfilamiento_grupos": res.get("perfiles", {})
        }
        context_prompt = build_model_context_prompt(model_info)

        # Chips de preguntas rápidas
        CHIPS = [
            "¿Cómo interpreto estas métricas?",
            "¿Hay riesgo de overfitting?",
            "¿Qué features son más importantes?",
            "¿Cómo puedo mejorar el modelo?",
            "Explica el preprocesamiento aplicado",
            "Traduce los resultados a lenguaje de negocio",
        ]
        
        st.markdown("**Preguntas rápidas:**")
        cols_chips = st.columns(len(CHIPS))
        pregunta_chip = None
        for i, chip in enumerate(CHIPS):
            if cols_chips[i].button(chip, key=f"chip_{i}", use_container_width=True):
                pregunta_chip = chip

        if user_input_res := st.chat_input("Pregunta dudas sobre el modelo, métricas o negocio...") or pregunta_chip:
            pregunta_final = user_input_res if user_input_res else pregunta_chip
            st.session_state.messages_resultados.append({"role": "user", "content": pregunta_final})
            
            # Mostrar el mensaje del usuario inmediatamente antes del stream
            with chat_container_res:
                with st.chat_message("user"):
                    st.markdown(pregunta_final)
            
            with chat_container_res:
                with st.chat_message("assistant"):
                    response_stream = chat_resultados_ia_stream(
                        st.session_state.chat_resultados_session, 
                        pregunta_final, 
                        context_prompt
                    )
                    if response_stream:
                        respuesta_completa = st.write_stream(chunk.text for chunk in response_stream if hasattr(chunk, "text"))
                        st.session_state.messages_resultados.append({"role": "assistant", "content": respuesta_completa})
            st.rerun()
