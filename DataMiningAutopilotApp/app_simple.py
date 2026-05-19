import json
import os
import re
import time
import numpy as np
from io import StringIO

# --- CONFIGURACIÓN AUTOMÁTICA DE CREDENCIALES GCP ---
path_credenciales = os.path.abspath(os.path.join(os.path.dirname(__file__), "credenciales", "BigQuery_credentials.json"))

if os.path.exists(path_credenciales):
    try:
        import json as _json_check
        with open(path_credenciales, "r", encoding="utf-8") as _fc:
            _cred_check = _json_check.load(_fc)
        if _cred_check.get("type") == "service_account":
            os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = path_credenciales
    except Exception:
        pass  # archivo vacío o inválido — dejamos que ADC del sistema tome el control

import pandas as pd
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
    texto_a_dataframe,
    subir_a_gcs,
    tabla_existe_en_bq,
    leer_cubo_de_bq,
    esperar_tablas_bq,
    llamar_build_cubo,
    limpiar_dataset_anterior,
    get_bq_client,
    PROJECT,
    DATASET,
    _resumen_tecnico_pipeline,
    _metricas_dashboard,
    _render_metricas_clave,
    _formatear_metrica,
    chat_resultados_ia_stream,
    build_model_context_prompt,
)


# ╔══════════════════════════════════════════════════════════════╗
# ║  UTILITY FUNCTIONS (UNCHANGED)                               ║
# ╚══════════════════════════════════════════════════════════════╝

def verificar_modo_operacion() -> dict:
    """
    Detecta si el entorno tiene credenciales GCP válidas.

    Retorna "cloud" si existe un service account JSON válido en
    credenciales/BigQuery_credentials.json (campo "type": "service_account")
    o si las Application Default Credentials (ADC) están disponibles en el
    entorno (gcloud auth application-default login, Workload Identity, etc.).
    Retorna "local" en cualquier otro caso.

    NOTA: Cuando BigQuery_credentials.json existe pero es inválido, el arranque
    de la app habrá puesto GOOGLE_APPLICATION_CREDENTIALS apuntando a ese archivo.
    google.auth.default() usaría ese path y fallaría antes de revisar las ADC del
    sistema. Por eso el bloque ADC hace un pop temporal de esa variable.

    Resultado:
        {
            "modo": "cloud" | "local",
            "razon": str,
            "credenciales_ok": bool,        # True si hay service account JSON válido
            "adc_disponible": bool,          # True si hay ADC en el entorno
            "tipo_credenciales": str,        # "service_account" | "user_adc" | "none"
            "proyecto_detectado": str | None
        }
    """
    if os.environ.get("FORCE_LOCAL_MODE", "").lower() in ("1", "true", "yes"):
        return {
            "modo": "local",
            "razon": "Modo local forzado via FORCE_LOCAL_MODE",
            "credenciales_ok": False,
            "adc_disponible": False,
            "tipo_credenciales": "forced_local",
            "proyecto_detectado": None,
        }

    import json as _json

    credenciales_ok = False
    adc_disponible = False
    tipo_credenciales = "none"
    proyecto_detectado = None

    # --- Intento 1: service account JSON local ---
    if os.path.exists(path_credenciales):
        try:
            with open(path_credenciales, "r", encoding="utf-8") as _f:
                _data = _json.load(_f)
            if _data.get("type") == "service_account":
                credenciales_ok = True
                tipo_credenciales = "service_account"
                proyecto_detectado = _data.get("project_id")
        except Exception:
            pass

    # --- Intento 2: ADC del sistema (gcloud, Workload Identity, etc.) ---
    if not credenciales_ok:
        try:
            import google.auth as _gauth
            # Si GOOGLE_APPLICATION_CREDENTIALS apunta a un archivo inválido
            # (vacío, malformado, no es service_account), google.auth.default()
            # falla al intentar cargarlo y nunca llega a revisar las ADC del
            # sistema. Lo descartamos temporalmente para que pueda descubrir
            # las credenciales reales (ej: ~/.config/gcloud/application_default_credentials.json).
            _env_key = "GOOGLE_APPLICATION_CREDENTIALS"
            _saved_gac = os.environ.pop(_env_key, None)
            try:
                _creds, _project = _gauth.default()
                adc_disponible = True
                proyecto_detectado = _project
                _mod = type(_creds).__module__
                tipo_credenciales = "service_account" if "service_account" in _mod else "user_adc"
            finally:
                if _saved_gac is not None:
                    os.environ[_env_key] = _saved_gac
        except Exception:
            pass

    # --- Construir resultado ---
    _proj_str = proyecto_detectado or "desconocido"
    if credenciales_ok:
        return {
            "modo": "cloud",
            "razon": f"Credenciales de servicio GCP encontradas en credenciales/BigQuery_credentials.json (proyecto: {_proj_str}).",
            "credenciales_ok": True,
            "adc_disponible": False,
            "tipo_credenciales": tipo_credenciales,
            "proyecto_detectado": proyecto_detectado,
        }
    if adc_disponible:
        return {
            "modo": "cloud",
            "razon": f"Application Default Credentials (ADC) disponibles en el entorno (proyecto: {_proj_str}).",
            "credenciales_ok": False,
            "adc_disponible": True,
            "tipo_credenciales": tipo_credenciales,
            "proyecto_detectado": proyecto_detectado,
        }
    return {
        "modo": "local",
        "razon": "No se encontraron credenciales GCP ni ADC. Operando en modo local.",
        "credenciales_ok": False,
        "adc_disponible": False,
        "tipo_credenciales": "none",
        "proyecto_detectado": None,
    }


def _leer_archivo_local(archivo) -> pd.DataFrame:
    """
    Lee un UploadedFile de Streamlit como DataFrame.
    Soporta CSV (UTF-8 con fallback a latin1) y Excel (.xlsx).
    Usado exclusivamente en modo local.
    """
    nombre = archivo.name.lower()
    if nombre.endswith(".xlsx"):
        return pd.read_excel(archivo)
    try:
        return pd.read_csv(archivo, encoding="utf-8")
    except UnicodeDecodeError:
        archivo.seek(0)
        return pd.read_csv(archivo, encoding="latin1")


def _construir_cubo_local(df_hechos: pd.DataFrame, dfs_dim: list) -> pd.DataFrame:
    """
    Construye el cubo analítico localmente aplicando LEFT JOIN de df_hechos
    con cada DataFrame de dimensión sobre las columnas con nombre en común.

    Equivale funcionalmente a la Cloud Function 'armar-cubo' en modo cloud,
    aunque sin la lógica de negocio personalizada que pueda tener esa función.
    Usado exclusivamente en modo local.
    """
    df = df_hechos.copy()
    for df_dim in dfs_dim:
        cols_comunes = list(set(df.columns) & set(df_dim.columns))
        if cols_comunes:
            df = df.merge(df_dim, on=cols_comunes, how="left")
    return df


# ╔══════════════════════════════════════════════════════════════╗
# ║  DIALOGS / MODALS                                            ║
# ╚══════════════════════════════════════════════════════════════╝

@st.dialog("Bienvenido a DataMining Autopilot")
def _onboarding_modal():
    st.markdown("##### ¿Cómo te llamamos?")
    nombre = st.text_input(
        "nombre",
        placeholder="Tu nombre",
        label_visibility="collapsed",
        key="input_nombre_onboarding",
    )
    btn_disabled = not nombre.strip()
    if st.button("Continuar", type="primary", use_container_width=True, disabled=btn_disabled):
        st.session_state.user_name = nombre.strip()
        st.session_state.user_initial = nombre.strip()[0].upper()
        st.rerun()


@st.dialog("Nueva Predicción", width="large")
def _prediction_modal():
    res = st.session_state.get("results")
    if res is None:
        st.warning("No hay resultados de modelo disponibles.")
        return
    if not st.session_state.get("messages_resultados"):
        st.info("⏳ Espera a que el agente termine de analizar los resultados antes de predecir.")
        return

    tab_text, tab_file = st.tabs(["💬 Ingreso por Chat", "📂 Subir Archivo"])

    with tab_text:
        texto_pred = st.text_area("Describe la situación para predecir:", height=100, key="pred_modal_text")
        if st.button("Predecir desde texto", use_container_width=True, type="primary", key="pred_modal_btn_text"):
            if texto_pred:
                with st.spinner("Interpretando y prediciendo..."):
                    try:
                        dtypes_dict = st.session_state.df.dtypes.apply(lambda x: str(x)).to_dict()
                        df_n = texto_a_dataframe(st.session_state.chat_session, texto_pred, dtypes_dict)
                        st.write("Datos interpretados por el agente:")
                        st.dataframe(df_n)
                        df_p = st.session_state.cleaner.transformar_nueva_tupla(df_n)
                        print("\n" + "="*60)
                        print("DATOS TRANSFORMADOS ENVIADOS AL MODELO (DEBUG):")
                        print(df_p[res["cols"]].to_string())
                        print("="*60 + "\n")
                        p = res["modelo"].predict(df_p[res["cols"]])
                        val_pred = p[0]
                        if isinstance(val_pred, (float, np.float64, np.float32, np.number)):
                            val_pred = round(float(val_pred), 4)
                        st.success(f"**Resultado predicho: {val_pred}**")
                    except Exception as e:
                        st.error(f"Error en la predicción por texto: {e}")
            else:
                st.warning("Por favor ingresa una descripción.")

    with tab_file:
        uploaded_pred = st.file_uploader(
            "Sube tus nuevos datos para predecir",
            type=["csv", "xlsx", "json"],
            key="pred_modal_uploader",
        )
        if st.button("Predecir desde archivo", use_container_width=True, key="pred_modal_btn_file"):
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
                        if p.dtype.kind in "fc":
                            df_res["Prediccion"] = np.round(p.astype(float), 4)
                        else:
                            df_res["Prediccion"] = p
                        st.success("Predicciones completadas.")
                        st.write("Mostrando los primeros 100 registros:")
                        st.dataframe(df_res.head(100))
                        csv = df_res.to_csv(index=False).encode("utf-8")
                        st.download_button(
                            label="📥 Descargar Resultados",
                            data=csv,
                            file_name="predicciones.csv",
                            mime="text/csv",
                            use_container_width=True,
                            key="pred_modal_download",
                        )
                    except Exception as e:
                        st.error(f"Error en la predicción por archivo: {e}")
            else:
                st.warning("Por favor sube un archivo primero.")


# ╔══════════════════════════════════════════════════════════════╗
# ║  CHAT HANDLER (assistant panel backend)                      ║
# ╚══════════════════════════════════════════════════════════════╝

def _handle_chat_input(user_input: str, phase: str, msg_key: str) -> None:
    """Send a user message to Gemini, store response, and rerun."""
    msgs = st.session_state.get(msg_key, [])
    msgs.append({"role": "user", "content": user_input})
    st.session_state[msg_key] = msgs

    with st.spinner("Procesando…"):
        try:
            if phase in ("PROPUESTA", "EJECUCION"):
                json_str = "{}"
                for m in reversed(msgs):
                    if m["role"] == "assistant":
                        match = re.search(r"```json\s*(\{.*?\})\s*```", m["content"], re.DOTALL)
                        if match:
                            json_str = match.group(1)
                            break
                instruction = (
                    f"Usa este json como base para modificaciones si aplica: {json_str}. "
                    f"El usuario dice: {user_input}"
                )
                response = get_ia_proposal(
                    st.session_state.chat_session,
                    st.session_state.df,
                    feedback=instruction,
                    diccionario_datos=st.session_state.data_dict_content,
                    nombre_usuario=st.session_state.get("user_name"),
                )
                msgs.append({"role": "assistant", "content": response})
                st.session_state.proposal = response
                if re.search(r"```json\s*(\{.*?\})\s*```", response, re.DOTALL):
                    st.session_state.proposal_update_notice = True

            elif phase == "RESULTADOS":
                res = st.session_state.get("results")
                if res:
                    model_info = {
                        "tipo_modelo":          res.get("tipo_modelo", ""),
                        "variable_obj":         res.get("target") or "No requerida",
                        "metricas":             res["metricas"].get("metricas_precision", {}),
                        "n_registros":          len(st.session_state.df) if st.session_state.df is not None else 0,
                        "n_features":           len(res.get("cols", [])),
                        "clases":               res["metricas"].get("clases_detectadas", []),
                        "encodings":            [
                            k for k, v in res.get("reglas", {}).items()
                            if isinstance(v, dict) and (
                                v.get("Dummies") or v.get("TargetEncoding")
                                or v.get("Ordinal") or v.get("WOE")
                            )
                        ],
                        "pca_aplicado":         res.get("es_pca", False),
                        "grid_search":          True,
                        "perfilamiento_grupos": res.get("perfiles", {}),
                    }
                    context_prompt = build_model_context_prompt(model_info, nombre_usuario=st.session_state.get("user_name"))
                    response_stream = chat_resultados_ia_stream(
                        st.session_state.chat_resultados_session,
                        user_input,
                        context_prompt,
                        nombre_usuario=st.session_state.get("user_name"),
                    )
                    if response_stream:
                        full_text = "".join(
                            chunk.text for chunk in response_stream if hasattr(chunk, "text")
                        )
                        msgs.append({"role": "assistant", "content": full_text})
        except Exception as e:
            msgs.append({"role": "assistant", "content": f"⚠️ Error al procesar: {e}"})

    st.session_state[msg_key] = msgs
    st.rerun()


# ╔══════════════════════════════════════════════════════════════╗
# ║  LAYOUT RENDER FUNCTIONS                                     ║
# ╚══════════════════════════════════════════════════════════════╝

def _build_chrome_html(phase: str, modo_app: dict, df, results, user_initial: str = "?") -> str:
    """
    Returns the HTML for the fixed-position chrome:
    TopBar (44 px) + Stepper (56 px, left 75%) + Context Bar (56 px, right 25%).
    All three are position:fixed so the page scrolls freely underneath.
    """
    # ── Topbar values ─────────────────────────────────────────
    is_cloud = modo_app.get("modo") == "cloud"
    mode_label = "Cloud" if is_cloud else "Local"
    mode_class = "cloud" if is_cloud else "local"
    mode_dot   = "🟢"   if is_cloud else "🔵"

    _phase_labels = {
        "CARGA":      "Carga de Datos",
        "PROPUESTA":  "Propuesta IA",
        "EJECUCION":  "Ejecutando pipeline…",
        "RESULTADOS": "Resultados",
    }
    phase_label  = _phase_labels.get(phase, "—")
    dataset_info = (
        f"{df.shape[0]:,} filas · {df.shape[1]} cols"
        if df is not None else "Sin dataset"
    )

    # ── Stepper values — 3 steps only ────────────────────────
    _phase_order = {"CARGA": 0, "PROPUESTA": 1, "EJECUCION": 1, "RESULTADOS": 2}
    current_idx  = _phase_order.get(phase, 0)
    _steps = [("1", "Carga"), ("2", "Propuesta"), ("3", "Resultados")]

    def _step(num, label, idx):
        if idx < current_idx:
            state, circle = "completed", "✓"
        elif idx == current_idx:
            state, circle = "current", num
        else:
            state, circle = "pending", num
        return (
            f'<div class="ae-step {state}">'
            f'  <div class="ae-step-circle">{circle}</div>'
            f'  <span class="ae-step-label">{label}</span>'
            f'</div>'
        )

    def _connector(idx):
        cls = "passed" if idx < current_idx else "pending"
        return f'<div class="ae-step-connector {cls}"></div>'

    stepper_items = "".join(
        _step(n, l, i) + (_connector(i) if i < len(_steps) - 1 else "")
        for i, (n, l) in enumerate(_steps)
    )

    # ── Context bar values ────────────────────────────────────
    if results:
        tipo_modelo = results.get("tipo_modelo", "—")
        target      = results.get("target") or "No requerida"
        ctx_icon    = "🤖"
        ctx_line1   = tipo_modelo
    else:
        ctx_icon  = "⚙️"
        ctx_line1 = "Sin modelo activo"
        target    = "—"
    n_reg      = f"{df.shape[0]:,}" if df is not None else "—"
    ctx_line2  = f"{target} · {n_reg} reg."

    # ── Assembly ──────────────────────────────────────────────
    return f"""
<div class="ae-topbar">
    <div class="ae-topbar-left">
        <div class="ae-topbar-logo">⚡</div>
        <span class="ae-topbar-brand">DM Autopilot</span>
    </div>
    <div class="ae-topbar-center">
        <div class="ae-breadcrumb">
            <span class="ae-breadcrumb-item">DataMining</span>
            <span class="ae-breadcrumb-sep">/</span>
            <span class="ae-breadcrumb-item">{dataset_info}</span>
            <span class="ae-breadcrumb-sep">/</span>
            <span class="ae-breadcrumb-active">{phase_label}</span>
        </div>
    </div>
    <div class="ae-topbar-right">
        <span class="ae-mode-badge {mode_class}">{mode_dot} {mode_label}</span>
        <div class="ae-avatar" title="{st.session_state.get('user_name', '')}">{user_initial}</div>
    </div>
</div>

<div class="ae-stepper">
    {stepper_items}
</div>

<div class="ae-context-bar">
    <div class="ae-context-icon">{ctx_icon}</div>
    <div class="ae-context-info">
        <div class="ae-context-line1">{ctx_line1}</div>
        <div class="ae-context-line2">{ctx_line2}</div>
    </div>
    <div class="ae-context-menu" title="Opciones">⋮</div>
</div>
"""


def _render_assistant_panel(collapsed: bool) -> None:
    """Persistent chat panel — right column, all phases."""
    phase = st.session_state.get("phase", "CARGA")

    _suggestions = {
        "CARGA":     ["¿Qué formato acepta?", "¿Cómo construye el cubo?", "¿Modos cloud vs local?"],
        "PROPUESTA": ["Justifica esta propuesta", "Sugiere modelos alternativos", "¿Y si elimino una columna?"],
        "EJECUCION": ["¿Qué hace el pipeline?", "¿Cuánto tarda esto?"],
        "RESULTADOS":["¿Cómo interpreto esto?", "¿Hay overfitting?", "Traduce a lenguaje de negocio", "¿Qué features importan?"],
    }
    chips = _suggestions.get(phase, [])

    # Collapsed strip
    if collapsed:
        st.markdown(
            '<div class="ae-assistant collapsed">'
            '<div class="ae-assistant-icon-v">A&nbsp;I</div>'
            '</div>',
            unsafe_allow_html=True,
        )
        if st.button("▶", key="btn_expand_assistant", help="Expandir asistente IA",
                     use_container_width=True):
            st.session_state.assistant_collapsed = False
            st.rerun()
        return

    # ── Header ────────────────────────────────────────────────
    st.markdown("""
<div class="ae-assistant-header">
  <div class="ae-assistant-title">
    <div class="ae-assistant-avatar">✦</div>
    <span class="ae-assistant-name">DM Autopilot</span>
  </div>
</div>""", unsafe_allow_html=True)

    # ── Determine message store ───────────────────────────────
    chat_active = phase in ("PROPUESTA", "RESULTADOS")
    if phase in ("PROPUESTA", "EJECUCION"):
        msg_key = "messages_propuesta"
    elif phase == "RESULTADOS":
        msg_key = "messages_resultados"
    else:
        msg_key = None

    msgs = st.session_state.get(msg_key, []) if msg_key else []

    # ── Messages area ─────────────────────────────────────────
    with st.container(height=480):
        if msgs:
            for m in msgs:
                with st.chat_message(m["role"]):
                    content = m["content"]
                    if msg_key == "messages_propuesta":
                        content = re.sub(r"```json.*?```", "", content, flags=re.DOTALL)
                        content = re.sub(r"(?i)-{3,}", "", content)
                        content = re.sub(
                            r"(?i)#+.*?Configuraci[oó]n.*?(?:JSON|Preprocesamiento|T[eé]cnica).*?\n?",
                            "", content,
                        )
                        content = re.sub(r"(?i)#+.*?Pipeline.*?\n?", "", content)
                        content = content.strip()
                    if content:
                        st.markdown(content)
        else:
            st.markdown(
                '<div style="color:var(--text-muted);text-align:center;'
                'padding:40px 12px;font-size:12px;line-height:1.7">'
                '✦<br>El agente responderá aquí.</div>',
                unsafe_allow_html=True,
            )

    # ── Suggestion chips ──────────────────────────────────────
    if chips:
        st.markdown('<div class="ae-suggestions-label">Sugerencias</div>', unsafe_allow_html=True)
        chip_disabled = not chat_active or not msg_key
        for chip in chips:
            if st.button(chip, key=f"chip_{phase[:3]}_{chip[:14]}", use_container_width=True,
                         disabled=chip_disabled):
                _handle_chat_input(chip, phase, msg_key)

    # ── Chat input ────────────────────────────────────────────
    if chat_active and msg_key:
        user_in = st.chat_input("Pregunta al agente…", key="asst_chat_input")
        if user_in:
            _handle_chat_input(user_in, phase, msg_key)

    # ── Collapse ──────────────────────────────────────────────
    if st.button("◀ Colapsar", key="btn_collapse_assistant",
                 help="Colapsar panel IA", use_container_width=True):
        st.session_state.assistant_collapsed = True
        st.rerun()


# ╔══════════════════════════════════════════════════════════════╗
# ║  WORKSPACE — all existing phase logic lives here             ║
# ╚══════════════════════════════════════════════════════════════╝

def _render_workspace():
    """Renders phase content inside the workspace column."""

    # ── FASE: CARGA ───────────────────────────────────────────
    if st.session_state.phase == "CARGA":

        _modo_actual = st.session_state.modo_app
        if _modo_actual["modo"] == "local":
            st.markdown(
                '<div class="ae-info-compact">'
                '🔵 <strong>Modo local activo</strong> — GCS y BigQuery no disponibles. '
                'Sube tu CSV/Excel directamente y se procesará en memoria.'
                '<span class="ae-info-tooltip" '
                'title="Sin credenciales GCP detectadas. Para activar Modo Cloud, '
                'coloca un service account válido en credenciales/BigQuery_credentials.json '
                'o ejecuta: gcloud auth application-default login">ⓘ</span>'
                '</div>',
                unsafe_allow_html=True,
            )

        col1, col2, col3 = st.columns(3)

        with col1:
            hechos = st.file_uploader("Dataset principal (Tabla de hechos)", type=["csv", "xlsx"])
            if hechos:
                st.success(f"✅ Dataset principal listo: **{hechos.name}**")

        with col2:
            dimensiones = st.file_uploader(
                "Dimensiones (Opcional)", type=["csv", "xlsx"], accept_multiple_files=True
            )
            if dimensiones:
                st.success(f"✅ {len(dimensiones)} archivos de dimensiones listos")

        with col3:
            uploaded_dict = st.file_uploader(
                "Diccionario de datos (Opcional)",
                type=["csv", "xlsx", "json", "txt"],
                key="dict_uploader",
            )
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
                                st.session_state.data_dict_content = uploaded_dict.read().decode(
                                    "utf-8", errors="ignore"
                                )
                            st.session_state.data_dict_name = uploaded_dict.name
                            st.session_state.last_uploaded_dict = dict_key
                            st.success(f"✅ Diccionario cargado: **{uploaded_dict.name}**")
                        except Exception as e:
                            st.error(f"Error al procesar el diccionario de datos: {e}")
                else:
                    st.success(f"✅ Diccionario activo: **{st.session_state.data_dict_name}**")

                if st.button("Eliminar Diccionario", use_container_width=True):
                    st.session_state.data_dict_content = None
                    st.session_state.data_dict_name = None
                    st.session_state.last_uploaded_dict = None
                    st.rerun()
            else:
                st.session_state.data_dict_content = None
                st.session_state.data_dict_name = None
                st.session_state.last_uploaded_dict = None

        puede_construir = hechos is not None

        col_btn_left, col_btn_center, col_btn_right = st.columns([1, 2, 1])
        with col_btn_center:
            if st.button(
                "🚀 Cargar y construir cubo",
                use_container_width=True,
                type="primary",
                disabled=not puede_construir,
            ):
                st.session_state.proposal = None
                st.session_state.config_pipeline = None
                st.session_state.results = None
                st.session_state.cleaner = None
                st.session_state.messages_propuesta = []
                st.session_state.messages_resultados = []
                st.session_state.proposal_update_notice = False
                st.session_state.chat_session = iniciar_chat()
                st.session_state.chat_resultados_session = iniciar_chat()

                if st.session_state.modo_app["modo"] == "local":
                    try:
                        with st.spinner("Procesando archivos localmente..."):
                            df_hechos = _leer_archivo_local(hechos)
                            if dimensiones:
                                dfs_dim = [_leer_archivo_local(d) for d in dimensiones]
                                st.session_state.df = _construir_cubo_local(df_hechos, dfs_dim)
                                st.success(
                                    f"Cubo local construido: {len(dimensiones)} dimensión(es) unidas "
                                    f"({st.session_state.df.shape[0]:,} filas × {st.session_state.df.shape[1]} columnas)."
                                )
                            else:
                                st.session_state.df = df_hechos
                                st.success(
                                    f"Dataset cargado: {df_hechos.shape[0]:,} filas × {df_hechos.shape[1]} columnas."
                                )
                        st.session_state.phase = "PROPUESTA"
                        st.rerun()
                    except Exception as e:
                        st.error(f"Error al procesar archivos localmente: {str(e)}")

                else:
                    try:
                        with st.spinner("Limpiando dataset e historial anterior en la nube..."):
                            limpiar_dataset_anterior()
                        
                        with st.spinner("Subiendo archivos a Cloud Storage..."):
                            subir_a_gcs(hechos, "Tabla_hechos")
                            if dimensiones:
                                for dim in dimensiones:
                                    subir_a_gcs(dim, "Dimensiones")
                        st.success("Archivos subidos a Cloud Storage")

                        nombres_esperados = ["hechos_raw"]
                        if dimensiones:
                            nombres_esperados += [
                                f"dim_{dim.name.split('.')[0].upper()}_raw"
                                for dim in dimensiones
                            ]

                        with st.spinner("Esperando carga en BigQuery..."):
                            ok = esperar_tablas_bq(nombres_esperados)

                        if not ok:
                            raise Exception("Timeout: las tablas no aparecieron en BigQuery. Revisa los logs.")
                        st.success("Tablas cargadas en BigQuery")

                        with st.spinner("Construyendo cubo analítico..."):
                            if dimensiones:
                                ok, resultado = llamar_build_cubo()
                                if not ok:
                                    raise Exception(f"Error al construir el cubo: {resultado}")
                            else:
                                client = get_bq_client()
                                client.query(f"""
                                    CREATE OR REPLACE VIEW `{PROJECT}.{DATASET}.cubo_analitico` AS
                                    SELECT * FROM `{PROJECT}.{DATASET}.hechos_raw`
                                """, location="northamerica-south1").result()
                        st.success("Cubo construido con éxito")

                        with st.spinner("Cargando datos para análisis..."):
                            st.session_state.df = leer_cubo_de_bq()

                        st.session_state.phase = "PROPUESTA"
                        st.rerun()

                    except Exception as e:
                        st.warning(f"⚠️ El procesamiento en la nube falló: {str(e)}")
                        st.info("Cambiando automáticamente a modo local para continuar...")
                        try:
                            with st.spinner("Procesando archivos localmente..."):
                                df_hechos = _leer_archivo_local(hechos)
                                if dimensiones:
                                    dfs_dim = [_leer_archivo_local(d) for d in dimensiones]
                                    st.session_state.df = _construir_cubo_local(df_hechos, dfs_dim)
                                    st.success(
                                        f"Cubo local construido como fallback: {len(dimensiones)} dimensión(es) unidas "
                                        f"({st.session_state.df.shape[0]:,} filas × {st.session_state.df.shape[1]} columnas)."
                                    )
                                else:
                                    st.session_state.df = df_hechos
                                    st.success(
                                        f"Dataset cargado localmente como fallback: {df_hechos.shape[0]:,} filas × {df_hechos.shape[1]} columnas."
                                    )
                            # Actualizar modo a local para reflejarlo en la UI
                            st.session_state.modo_app["modo"] = "local"
                            st.session_state.modo_app["razon"] = f"Fallo en la nube: {str(e)}. Fallback local activado."
                            
                            time.sleep(2)
                            st.session_state.phase = "PROPUESTA"
                            st.rerun()
                        except Exception as local_err:
                            st.error(f"Error crítico: También falló el procesamiento local: {str(local_err)}")

    # ── FASE: PROPUESTA ───────────────────────────────────────
    elif st.session_state.phase == "PROPUESTA":

        # Initial proposal generation (logic — no chat display here)
        if not st.session_state.messages_propuesta:
            with st.spinner("El agente está diseñando la estrategia inicial…"):
                initial_proposal = get_ia_proposal(
                    st.session_state.chat_session,
                    st.session_state.df,
                    is_initial=True,
                    diccionario_datos=st.session_state.data_dict_content,
                    nombre_usuario=st.session_state.get("user_name"),
                )
                st.session_state.messages_propuesta.append(
                    {"role": "assistant", "content": initial_proposal}
                )
                st.session_state.proposal = initial_proposal
        else:
            if st.session_state.proposal is None:
                for msg in reversed(st.session_state.messages_propuesta):
                    if msg["role"] == "assistant":
                        st.session_state.proposal = msg["content"]
                        break

        # Extract latest JSON from proposal messages
        json_str = "{}"
        for msg in reversed(st.session_state.messages_propuesta):
            if msg["role"] == "assistant":
                json_match = re.search(r"```json\s*(\{.*?\})\s*```", msg["content"], re.DOTALL)
                if json_match:
                    json_str = json_match.group(1)
                    break

        # Generate report before tabs so tab-switch reruns don't cancel the long computation
        if st.session_state.proposal is not None and not st.session_state.report_html:
            with st.spinner("Generando reporte de calidad de datos… (puede tardar 1-2 min)"):
                try:
                    st.session_state.report_html = AnalizarDatos(st.session_state.df)
                except Exception as _rep_err:
                    st.session_state.report_html = (
                        f"<div style='color:#ef4444;padding:16px'>"
                        f"Error al generar reporte: {_rep_err}</div>"
                    )

        tab1, tab2 = st.tabs(["⚙️ Configuración del Modelo", "📋 Reporte de Datos"])

        with tab1:
            if st.session_state.get("proposal_update_notice"):
                st.success("✅ Las propuestas de limpieza fueron actualizadas correctamente.")
                st.session_state.proposal_update_notice = False

            try:
                conf_data = json.loads(json_str) if json_str != "{}" else {}
                target_detectado, reglas_detectadas, modelo_detectado, es_pca_detectado, n_clusters_detectado = (
                    extraer_configuracion_pipeline(conf_data)
                )

                # Summary cards
                mc1, mc2, mc3 = st.columns(3)
                mc1.metric("Modelo detectado", modelo_detectado or "Pendiente")
                mc2.metric(
                    "Variable objetivo",
                    target_detectado if target_detectado else "⚠️ Pendiente",
                )
                mc3.metric(
                    "Métricas esperadas",
                    ", ".join(obtener_metricas_esperadas(modelo_detectado)) if modelo_detectado else "—",
                )

                st.markdown("---")

                opciones_modelos = list(MODELOS_DISPONIBLES.keys())
                modelo_seleccionado = st.selectbox(
                    "Modelo detectado",
                    opciones_modelos,
                    index=opciones_modelos.index(modelo_detectado)
                    if modelo_detectado in opciones_modelos
                    else 0,
                )
                conf_data["modelo"] = modelo_seleccionado
                conf_data["tipo_modelo"] = modelo_seleccionado

                if MODELOS_DISPONIBLES[modelo_seleccionado]["tipo_problema"] == "clustering":
                    n_clusters_mostrado = (
                        str(n_clusters_detectado) if n_clusters_detectado else "AUTO"
                    )
                    st.markdown(f"**Número de Clusters:** `{n_clusters_mostrado}`")
                    target_validacion = None
                else:
                    target_mostrado = target_detectado if target_detectado else "⚠️ Pendiente de definir"
                    st.markdown(f"**Variable Objetivo:** `{target_mostrado}`")
                    target_validacion = target_detectado
                st.markdown(f"**Modelo sugerido:** `{modelo_seleccionado}`")

                es_valido, mensaje_validacion = validar_tipo_problema(
                    st.session_state.df, target_validacion, modelo_seleccionado
                )
                if es_valido:
                    st.success(f"Validación del problema: {mensaje_validacion}")
                else:
                    st.error(f"Validación del problema: {mensaje_validacion}")

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
                        metodo_usado = params.get("metodo", "AUTO") if params_validos else "AUTO"
                        metodo_normalizado = str(metodo_usado).lower()

                        if metodo_normalizado in [
                            "mean", "median", "mode", "media", "mediana", "moda",
                            "auto", "imputar", "imputacion", "imputación",
                        ]:
                            tratamiento_str = "Imputacion"
                        elif metodo_normalizado == "drop-column":
                            tratamiento_str = "Eliminar Columna"
                        else:
                            tratamiento_str = "Imputacion"

                        if col in st.session_state.df.columns:
                            if metodo_normalizado == "drop-column":
                                conteo_datos = f"{len(st.session_state.df)} filas"
                            elif params_validos and (
                                params.get("Dummies")
                                or params.get("TargetEncoding")
                                or params.get("Ordinal")
                                or params.get("WOE")
                            ):
                                conteo_datos = f"{st.session_state.df[col].nunique(dropna=True)} valores únicos"
                            else:
                                conteo_datos = f"{int(st.session_state.df[col].isna().sum())} nulos"
                        else:
                            conteo_datos = "Columna no encontrada"

                        table_data.append(
                            {
                                "Columna": col,
                                "Tratamiento": tratamiento_str,
                                "Estado": "Listo" if params_validos else "Revisar",
                                "Conteo/Datos": conteo_datos,
                                "Dummies": "Si" if params_validos and params.get("Dummies") else "No",
                            }
                        )
                    st.table(pd.DataFrame(table_data))
                    with st.expander("¿Qué significa esta tabla?"):
                        st.markdown(
                            """
                            - **Tratamiento:** acción que se aplicará a la columna (imputar nulos, eliminar, transformar categorías).
                            - **Estado:** indica si la propuesta está lista o necesita revisión.
                            - **Conteo/Datos:** registros afectados (nulos a rellenar, filas, o valores únicos a transformar).
                            """
                        )

            except Exception as e:
                conf_data = {}
                puede_ejecutar = False
                st.error(f"Error en configuración JSON: {e}")

            col_empty, col_exec = st.columns([3, 1])
            with col_exec:
                if st.button(
                    "▶ Ejecutar Pipeline",
                    use_container_width=True,
                    disabled=not puede_ejecutar,
                    key="btn_ejecutar_pipeline",
                    type="primary",
                ):
                    st.session_state.config_pipeline = conf_data
                    st.session_state.phase = "EJECUCION"
                    st.rerun()

        with tab2:
            st.markdown("### Reporte Exploratorio Detallado")
            if st.session_state.proposal is None:
                st.markdown("""
                    <div style="background-color: #1a2c3d; border-left: 5px solid #00f2fe;
                         padding: 20px; border-radius: 10px; margin: 20px 0;">
                        <span style="font-size: 20px;">⏳</span>
                        <span style="color: #00f2fe; font-weight: 500; margin-left: 10px;">
                            Por favor espera a que la IA termine de generar la propuesta
                            estratégica antes de ver el reporte de datos.
                        </span>
                    </div>
                """, unsafe_allow_html=True)
            elif st.session_state.report_html:
                components.html(st.session_state.report_html, height=1000, scrolling=True)
            else:
                st.info("⏳ Generando reporte… por favor espera.")

    # ── FASE: EJECUCION ───────────────────────────────────────
    elif st.session_state.phase == "EJECUCION":
        conf = st.session_state.config_pipeline
        try:
            import os, shutil
            tiempo_inicio_pipeline = time.time()
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

                X_numeric = (
                    X_numeric.apply(pd.to_numeric, errors="coerce").fillna(0).astype(float)
                )
                if X_numeric.empty:
                    raise ValueError(
                        "No quedaron columnas numéricas disponibles para entrenar el modelo."
                    )

                modelo_obj, metricas, cols = orquestador_modelos_interno(
                    X_numeric, y, tipo_modelo=modelo_t, n_clusters_fix=n_clusters_fix
                )

                perfiles = {}
                if es_modelo_clustering(modelo_t):
                    try:
                        df_temp = st.session_state.df.loc[X.index].copy()
                        if hasattr(modelo_obj, "labels") and modelo_obj.labels is not None:
                            labels = modelo_obj.labels
                        else:
                            labels = (
                                modelo_obj.estimator.labels_
                                if hasattr(modelo_obj, "estimator") and modelo_obj.estimator
                                else None
                            )

                        if labels is None:
                            raise ValueError(
                                "No se pudieron obtener las etiquetas del modelo para el perfilamiento."
                            )

                        df_temp["__GRUPO__"] = labels
                        grupos = df_temp["__GRUPO__"].unique()

                        for g in grupos:
                            df_g = df_temp[df_temp["__GRUPO__"] == g]
                            desc_num = df_g.describe().to_dict()
                            cat_freqs = {}
                            for col in df_g.select_dtypes(include=["object", "category"]).columns:
                                freq = (
                                    df_g[col].value_counts(normalize=True) * 100
                                ).round(1).to_dict()
                                cat_freqs[col] = freq

                            perfiles[str(g)] = {
                                "estadisticas_numericas": desc_num,
                                "distribucion_categorias": cat_freqs,
                                "tamaño_grupo": len(df_g),
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
                    "perfiles": perfiles,
                    "tiempo_total_ejecucion": round(time.time() - tiempo_inicio_pipeline, 2),
                }

            st.session_state.phase = "RESULTADOS"
            st.rerun()
        except Exception as e:
            st.error(f"Error en el Pipeline: {e}")
            if st.button("Reintentar Propuesta"):
                st.session_state.phase = "PROPUESTA"
                st.rerun()

    # ── FASE: RESULTADOS ──────────────────────────────────────
    elif st.session_state.phase == "RESULTADOS":
        res = st.session_state.results

        tipo_modelo = res.get("tipo_modelo", "Regresion_lineal")
        es_clustering_resultado  = es_modelo_clustering(tipo_modelo)
        es_redes_resultado       = es_modelo_redes_neuronales(tipo_modelo)
        es_knn_resultado         = es_modelo_knn(tipo_modelo)
        es_arbol_resultado       = es_modelo_arbol(tipo_modelo)
        es_logistica_resultado   = es_modelo_regresion_logistica(tipo_modelo)
        es_credit_resultado      = es_modelo_credit_scoring(tipo_modelo)
        metricas_interfaz        = ocultar_woe_interfaz(res["metricas"])

        # Initial Gemini analysis — runs once
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
                        res["cols"],
                        res.get("perfiles", {}),
                        tarea,
                        nombre_usuario=st.session_state.get("user_name"),
                    )
                else:
                    # --- Extraer coeficientes / importancias para modelos de caja blanca ---
                    coeficientes_info = None
                    modelo_obj = res.get("modelo")
                    cols_modelo = res.get("cols", [])
                    try:
                        if modelo_obj is not None and cols_modelo:
                            if hasattr(modelo_obj, "feature_importances_"):
                                # Árboles, RandomForest
                                importancias = modelo_obj.feature_importances_
                                pares = sorted(zip(cols_modelo, importancias), key=lambda x: abs(x[1]), reverse=True)
                                coeficientes_info = "\n".join(
                                    f"  {i+1:2}. {col:40s}  importancia = {val:.4f}"
                                    for i, (col, val) in enumerate(pares[:15])
                                )
                            elif hasattr(modelo_obj, "coef_"):
                                # Regresión lineal / logística
                                coef = np.asarray(modelo_obj.coef_).flatten()
                                pares = sorted(zip(cols_modelo, coef), key=lambda x: abs(x[1]), reverse=True)
                                coeficientes_info = "\n".join(
                                    f"  {i+1:2}. {col:40s}  coeficiente = {val:+.4f}"
                                    for i, (col, val) in enumerate(pares[:15])
                                )
                            # Modelos envueltos (TransformedTargetRegressor)
                            elif hasattr(modelo_obj, "regressor_") and hasattr(modelo_obj.regressor_, "coef_"):
                                coef = np.asarray(modelo_obj.regressor_.coef_).flatten()
                                pares = sorted(zip(cols_modelo, coef), key=lambda x: abs(x[1]), reverse=True)
                                coeficientes_info = "\n".join(
                                    f"  {i+1:2}. {col:40s}  coeficiente = {val:+.4f}"
                                    for i, (col, val) in enumerate(pares[:15])
                                )
                    except Exception as _e_coef:
                        print(f"[INFO] No se pudieron extraer coeficientes del modelo: {_e_coef}")

                    explicacion = interpretar_resultados(
                        st.session_state.chat_resultados_session,
                        metricas_interfaz,
                        res["cols"],
                        tarea,
                        nombre_usuario=st.session_state.get("user_name"),
                        coeficientes_info=coeficientes_info,
                    )
                st.session_state.messages_resultados.append(
                    {"role": "assistant", "content": explicacion}
                )
                st.rerun()

        # Pipeline summary
        cleaner_actual = st.session_state.cleaner
        resumen_pipeline = _resumen_tecnico_pipeline(res, cleaner_actual, st.session_state.df)
        metricas_clave_dashboard = _metricas_dashboard(metricas_interfaz, es_clustering_resultado)
        columnas_eliminadas = resumen_pipeline["columnas_eliminadas"]

        # ── Compact results header ────────────────────────────
        col_info, col_btns = st.columns([3, 1])
        with col_info:
            target_display = res.get("target") or "No requerida"
            n_reg_display = len(st.session_state.df) if st.session_state.df is not None else 0
            st.markdown(
                f'<div class="ae-results-header">'
                f'<span class="ae-results-model">{tipo_modelo}</span>'
                f'<span class="ae-results-meta">'
                f'Target: <code>{target_display}</code> · {n_reg_display:,} registros'
                f'</span></div>',
                unsafe_allow_html=True,
            )
        with col_btns:
            bc1, bc2 = st.columns(2)
            with bc1:
                if st.button("🔮 Predecir", key="btn_pred_hdr", use_container_width=True, type="primary"):
                    _prediction_modal()
            with bc2:
                if st.session_state.df is not None:
                    csv_export = st.session_state.df.to_csv(index=False).encode("utf-8")
                    st.download_button(
                        "📥 Exportar",
                        data=csv_export,
                        file_name="dataset.csv",
                        mime="text/csv",
                        use_container_width=True,
                        key="btn_export_hdr",
                    )

        # ── KPI cards ─────────────────────────────────────────
        _render_metricas_clave(metricas_clave_dashboard)

        # Pipeline summary row
        resumen_cols = st.columns(5)
        resumen_cols[0].metric("Columnas usadas", _formatear_metrica(resumen_pipeline["columnas_usadas"]))
        resumen_cols[1].metric("Eliminadas", _formatear_metrica(len(columnas_eliminadas)))
        resumen_cols[2].metric("Outliers", _formatear_metrica(resumen_pipeline["outliers_corregidos"]))
        resumen_cols[3].metric("Nulos tratados", _formatear_metrica(resumen_pipeline["nulos_tratados"]))
        if resumen_pipeline["tiempo_total"] != "N/D":
            resumen_cols[4].metric("Tiempo", f"{resumen_pipeline['tiempo_total']} s")

        # ── Analysis tabs ─────────────────────────────────────
        tab_perf, tab_pipe, tab_ins = st.tabs(
            ["📊 Performance", "🔍 Pipeline", "💡 Insights IA"]
        )

        with tab_perf:
            if es_clustering_resultado:
                metricas_cluster = metricas_interfaz
                st.markdown("#### Resultados de Clustering")
                st.write(f"**Algoritmo:** {metricas_cluster.get('modelo_seleccionado')}")
                st.write(f"**Mejor número de clusters:** {metricas_cluster.get('mejor_numero_clusters')}")

                visualizaciones = metricas_cluster.get("visualizaciones", {})
                col_img1, col_img2 = st.columns(2)
                with col_img1:
                    if visualizaciones.get("scatter_clusters"):
                        st.image(visualizaciones["scatter_clusters"], caption="Scatter de clusters")
                    if visualizaciones.get("dendrograma"):
                        st.image(visualizaciones["dendrograma"], caption="Dendrograma")
                with col_img2:
                    if visualizaciones.get("elbow_chart"):
                        st.image(visualizaciones["elbow_chart"], caption="Método del codo")

                with st.expander("Comparación de algoritmos"):
                    st.json(metricas_cluster.get("metricas_por_algoritmo", []))

            else:
                if tipo_modelo == "Regresion_lineal":
                    metricas_lineal = metricas_interfaz
                    st.markdown("#### Resultados de Regresión Lineal")
                    st.write(f"**Tipo de problema:** {metricas_lineal.get('tipo_problema', 'regresion')}")
                    visualizaciones = metricas_lineal.get("visualizaciones", {})
                    if visualizaciones.get("real_vs_prediccion"):
                        st.image(visualizaciones["real_vs_prediccion"], caption="Real vs predicción",
                                 use_column_width=True)
                    else:
                        st.info("Visualización real vs predicción no disponible.")

                if es_redes_resultado:
                    metricas_red = metricas_interfaz
                    st.markdown("#### Resultados de Redes Neuronales")
                    st.write(f"**Tipo de problema:** {metricas_red.get('tipo_problema')}")
                    visualizaciones = metricas_red.get("visualizaciones", {})
                    col_nn1, col_nn2 = st.columns(2)
                    with col_nn1:
                        if visualizaciones.get("perdida"):
                            st.image(visualizaciones["perdida"], caption="Curva de pérdida")
                        if visualizaciones.get("curva_roc"):
                            st.image(visualizaciones["curva_roc"], caption="Curva ROC")
                    with col_nn2:
                        if visualizaciones.get("matriz_confusion"):
                            st.image(visualizaciones["matriz_confusion"], caption="Matriz de confusión")
                        if visualizaciones.get("real_vs_prediccion"):
                            st.image(visualizaciones["real_vs_prediccion"], caption="Real vs predicción")

                if es_knn_resultado:
                    metricas_knn = metricas_interfaz
                    st.markdown("#### Resultados de KNN")
                    st.write(f"**Tipo de problema:** {metricas_knn.get('tipo_problema')}")
                    visualizaciones = metricas_knn.get("visualizaciones", {})
                    col_knn1, col_knn2 = st.columns(2)
                    with col_knn1:
                        if visualizaciones.get("error_vs_k"):
                            st.image(visualizaciones["error_vs_k"], caption="Error vs K")
                        if visualizaciones.get("matriz_confusion"):
                            st.image(visualizaciones["matriz_confusion"], caption="Matriz de confusión")
                    with col_knn2:
                        if visualizaciones.get("curva_roc"):
                            st.image(visualizaciones["curva_roc"], caption="Curva ROC")
                        if visualizaciones.get("real_vs_prediccion"):
                            st.image(visualizaciones["real_vs_prediccion"], caption="Predicción vs valores reales")

                if es_arbol_resultado:
                    metricas_arbol = metricas_interfaz
                    st.markdown("#### Resultados de Árboles")
                    st.write(f"**Modelo seleccionado:** {metricas_arbol.get('modelo_seleccionado')}")
                    st.write(f"**Tipo de problema:** {metricas_arbol.get('tipo_problema')}")
                    visualizaciones = metricas_arbol.get("visualizaciones", {})
                    col_tree1, col_tree2 = st.columns(2)
                    with col_tree1:
                        if visualizaciones.get("arbol"):
                            st.image(visualizaciones["arbol"], caption="Árbol")
                        if visualizaciones.get("matriz_confusion"):
                            st.image(visualizaciones["matriz_confusion"], caption="Matriz de confusión")
                    with col_tree2:
                        if visualizaciones.get("feature_importance"):
                            st.image(visualizaciones["feature_importance"], caption="Feature importance")
                        if visualizaciones.get("real_vs_prediccion"):
                            st.image(visualizaciones["real_vs_prediccion"], caption="Real vs predicción")

                if es_logistica_resultado:
                    metricas_log = metricas_interfaz
                    st.markdown("#### Resultados de Regresión Logística")
                    st.write(f"**Tipo de problema:** {metricas_log.get('tipo_problema')}")
                    visualizaciones = metricas_log.get("visualizaciones", {})
                    col_log1, col_log2 = st.columns(2)
                    with col_log1:
                        if visualizaciones.get("curva_roc"):
                            st.image(visualizaciones["curva_roc"], caption="Curva ROC")
                        if visualizaciones.get("matriz_confusion"):
                            st.image(visualizaciones["matriz_confusion"], caption="Matriz de confusión")
                    with col_log2:
                        if visualizaciones.get("coeficientes"):
                            st.image(visualizaciones["coeficientes"], caption="Coeficientes")

                if es_credit_resultado:
                    metricas_credit = metricas_interfaz
                    st.markdown("#### Resultados de Credit Scoring")
                    st.write(f"**Tipo de problema:** {metricas_credit.get('tipo_problema')}")
                    visualizaciones = metricas_credit.get("visualizaciones", {})
                    col_credit1, col_credit2 = st.columns(2)
                    with col_credit1:
                        if visualizaciones.get("score_distribution"):
                            st.image(visualizaciones["score_distribution"], caption="Distribución de score")
                        if visualizaciones.get("curva_roc"):
                            st.image(visualizaciones["curva_roc"], caption="Curva ROC")
                        if visualizaciones.get("matriz_confusion"):
                            st.image(visualizaciones["matriz_confusion"], caption="Matriz de confusión")
                    with col_credit2:
                        if visualizaciones.get("segmentos_riesgo"):
                            st.image(visualizaciones["segmentos_riesgo"], caption="Segmentos de riesgo")
                        if visualizaciones.get("coeficientes"):
                            st.image(visualizaciones["coeficientes"], caption="Coeficientes")

        with tab_pipe:
            st.markdown("#### Resumen del Pipeline")
            rp1, rp2 = st.columns(2)
            with rp1:
                st.markdown(f"**Columnas usadas:** {resumen_pipeline['columnas_usadas']}")
                st.markdown(f"**Outliers corregidos:** {resumen_pipeline['outliers_corregidos']}")
                if resumen_pipeline["tiempo_total"] != "N/D":
                    st.markdown(f"**Tiempo total:** {resumen_pipeline['tiempo_total']} s")
            with rp2:
                st.markdown(f"**Nulos tratados:** {resumen_pipeline['nulos_tratados']}")
                if columnas_eliminadas:
                    st.markdown(f"**Columnas eliminadas ({len(columnas_eliminadas)}):**")
                    st.code(", ".join(str(c) for c in columnas_eliminadas))
            with st.expander("Reglas de preprocesamiento (JSON)"):
                st.json(res.get("reglas", {}))

        with tab_ins:
            st.markdown("#### Interpretación del Agente")
            if st.session_state.messages_resultados:
                first_msg = st.session_state.messages_resultados[0]
                st.markdown(first_msg["content"])
            else:
                st.info("El agente aún no ha analizado los resultados.")


# ╔══════════════════════════════════════════════════════════════╗
# ║  APP BOOTSTRAP                                               ║
# ╚══════════════════════════════════════════════════════════════╝

st.set_page_config(page_title="DM Autopilot", page_icon="⚡", layout="wide")
load_css("styles.css")

# ── Session state initialisation ──────────────────────────────
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
if "proposal_update_notice" not in st.session_state:
    st.session_state.proposal_update_notice = False
if "modo_app" not in st.session_state:
    st.session_state.modo_app = verificar_modo_operacion()
if "assistant_collapsed" not in st.session_state:
    st.session_state.assistant_collapsed = False

# ── Onboarding: block until user provides a name ──────────────
if "user_name" not in st.session_state:
    _onboarding_modal()
    st.stop()


# ╔══════════════════════════════════════════════════════════════╗
# ║  MAIN LAYOUT                                                 ║
# ╚══════════════════════════════════════════════════════════════╝

# ── 1. Fixed chrome: TopBar + Stepper + Context Bar ───────────
st.markdown(
    _build_chrome_html(
        st.session_state.phase,
        st.session_state.modo_app,
        st.session_state.df,
        st.session_state.results,
        user_initial=st.session_state.get("user_initial", "?"),
    ),
    unsafe_allow_html=True,
)

# ── 2. Navigation row ─────────────────────────────────────────
_back_map  = {"PROPUESTA": ("CARGA", "← Carga"), "RESULTADOS": ("PROPUESTA", "← Propuesta")}
_show_back = st.session_state.phase in _back_map and st.session_state.phase != "EJECUCION"

_nav_l, _nav_mid, _nav_r = st.columns([1, 5, 1])

with _nav_l:
    if _show_back:
        _target_phase, _back_label = _back_map[st.session_state.phase]
        st.markdown('<div class="ae-back-btn">', unsafe_allow_html=True)
        if st.button(_back_label, key="btn_back_phase"):
            st.session_state.phase = _target_phase
            st.rerun()
        st.markdown("</div>", unsafe_allow_html=True)

with _nav_r:
    _uc, _rc = st.columns([1, 1])
    with _uc:
        _uname = st.session_state.get("user_name", "Usuario")
        _uinitial = st.session_state.get("user_initial", "?")
        with st.popover(_uinitial, use_container_width=True):
            st.markdown(f"**{_uname}**")
            if st.button("✏️ Cambiar nombre", key="btn_change_name", use_container_width=True):
                st.session_state.pop("user_name", None)
                st.session_state.pop("user_initial", None)
                st.rerun()
    with _rc:
        st.markdown('<div class="ae-reset-btn">', unsafe_allow_html=True)
        if st.button("↺", key="btn_reset_system", help="Reiniciar aplicación"):
            st.session_state.clear()
            st.rerun()
        st.markdown("</div>", unsafe_allow_html=True)

# ── 3. Workspace (75%) + Assistant panel (25% or strip) ───────
_collapsed = st.session_state.assistant_collapsed
_col_ws, _col_asst = st.columns([11, 1] if _collapsed else [3, 1])

with _col_ws:
    _render_workspace()

with _col_asst:
    _render_assistant_panel(_collapsed)
