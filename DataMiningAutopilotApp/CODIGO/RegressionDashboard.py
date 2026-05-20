import math

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


PLOTLY_TEMPLATE = "plotly_dark"
COLOR_GOOD = "#10b981"
COLOR_WARN = "#f59e0b"
COLOR_BAD = "#ef4444"
COLOR_INFO = "#38bdf8"


def _safe_float(value):
    try:
        if value is None or pd.isna(value):
            return None
        return float(value)
    except Exception:
        return None


def _fmt(value, digits=4):
    value = _safe_float(value)
    if value is None:
        return "N/D"
    if abs(value) >= 1000:
        return f"{value:,.2f}"
    return f"{value:,.{digits}f}"


def _score_color(metric_name, value):
    value = _safe_float(value)
    if value is None:
        return COLOR_INFO
    name = metric_name.lower()
    if name in {"r2", "modelo.score"}:
        if value >= 0.75:
            return COLOR_GOOD
        if value >= 0.45:
            return COLOR_WARN
        return COLOR_BAD
    if name == "mape":
        if value <= 0.10:
            return COLOR_GOOD
        if value <= 0.25:
            return COLOR_WARN
        return COLOR_BAD
    return COLOR_INFO


def _metric_delta(metric_name, value):
    value = _safe_float(value)
    if value is None:
        return None, "off"
    name = metric_name.lower()
    if name in {"r2", "modelo.score"}:
        if value >= 0.75:
            return "alto", "normal"
        if value >= 0.45:
            return "medio", "off"
        return "bajo", "inverse"
    if name == "mape":
        if value <= 0.10:
            return "bajo error", "normal"
        if value <= 0.25:
            return "error medio", "off"
        return "error alto", "inverse"
    return None, "off"


def _section(title, subtitle=None):
    st.markdown(f"### {title}")
    if subtitle:
        st.caption(subtitle)


def _empty_state(text):
    st.info(text)


def _diagnostics_frame(metricas):
    diag = metricas.get("diagnostico", {}) if isinstance(metricas, dict) else {}
    y_true = diag.get("y_test", [])
    y_pred = diag.get("y_pred", [])
    if not y_true or not y_pred or len(y_true) != len(y_pred):
        return pd.DataFrame()
    df = pd.DataFrame(
        {
            "real": pd.to_numeric(pd.Series(y_true), errors="coerce"),
            "prediccion": pd.to_numeric(pd.Series(y_pred), errors="coerce"),
        }
    ).dropna()
    if df.empty:
        return df
    df["residual"] = df["real"] - df["prediccion"]
    df["error_absoluto"] = df["residual"].abs()
    df["error_pct"] = np.where(
        df["real"].abs() > 1e-9,
        df["error_absoluto"] / df["real"].abs(),
        np.nan,
    )
    df["registro"] = np.arange(1, len(df) + 1)
    return df


def _coefficient_frame(modelo, cols):
    if modelo is None or not cols:
        return pd.DataFrame()
    coef = None
    try:
        if hasattr(modelo, "coef_"):
            coef = np.asarray(modelo.coef_).flatten()
        elif hasattr(modelo, "regressor_") and hasattr(modelo.regressor_, "coef_"):
            coef = np.asarray(modelo.regressor_.coef_).flatten()
        elif hasattr(modelo, "regressor") and hasattr(modelo.regressor, "coef_"):
            coef = np.asarray(modelo.regressor.coef_).flatten()
    except Exception:
        coef = None
    if coef is None or len(coef) != len(cols):
        return pd.DataFrame()
    df = pd.DataFrame({"variable": cols, "coeficiente": coef})
    df["magnitud_abs"] = df["coeficiente"].abs()
    return df.sort_values("magnitud_abs", ascending=False)


def _correlation_frame(df_original, target, top_n=12):
    if df_original is None or not target or target not in df_original.columns:
        return pd.DataFrame()
    numeric = df_original.select_dtypes(include=[np.number]).copy()
    if target not in numeric.columns or numeric.shape[1] < 2:
        return pd.DataFrame()
    corr = numeric.corr(numeric_only=True)[target].drop(labels=[target], errors="ignore")
    corr = corr.replace([np.inf, -np.inf], np.nan).dropna()
    if corr.empty:
        return pd.DataFrame()
    out = corr.abs().sort_values(ascending=False).head(top_n).reset_index()
    out.columns = ["variable", "correlacion_abs"]
    out["correlacion"] = out["variable"].map(corr)
    return out


def _dataset_health(df_original, resumen_pipeline):
    if df_original is None or df_original.empty:
        return {}
    null_pct = float(df_original.isna().mean().mean() * 100)
    num_cols = int(df_original.select_dtypes(include=[np.number]).shape[1])
    cat_cols = int(df_original.select_dtypes(include=["object", "category", "string", "bool"]).shape[1])
    total_rows = int(len(df_original))
    clean_logs = resumen_pipeline.get("logs", []) if isinstance(resumen_pipeline, dict) else []
    rows_removed = 0
    for log in clean_logs:
        detail = str(log.get("detalle", ""))
        if "filas" in detail:
            try:
                rows_removed += int(detail.split()[0])
            except Exception:
                pass
    return {
        "nulos_pct": null_pct,
        "columnas_numericas": num_cols,
        "columnas_categoricas": cat_cols,
        "filas_originales": total_rows,
        "filas_eliminadas": rows_removed,
        "outliers": resumen_pipeline.get("outliers_corregidos", 0),
        "columnas_eliminadas": len(resumen_pipeline.get("columnas_eliminadas", [])),
    }


def _render_kpis(metricas, n_registros, n_variables):
    precision = metricas.get("metricas_precision", {}) if isinstance(metricas, dict) else {}
    items = [
        ("R2", precision.get("R2")),
        ("RMSE", precision.get("RMSE")),
        ("MAE", precision.get("MAE")),
        ("MAPE", precision.get("MAPE")),
        ("Registros", n_registros),
        ("Variables usadas", n_variables),
    ]
    cols = st.columns(6)
    for col, (label, value) in zip(cols, items):
        with col:
            color = _score_color(label, value)
            delta, delta_color = _metric_delta(label, value)
            st.markdown(
                f"<div class='lm-kpi-accent' style='border-color:{color}'></div>",
                unsafe_allow_html=True,
            )
            st.metric(label, _fmt(value, 4) if label not in {"Registros", "Variables usadas"} else f"{int(value):,}", delta=delta, delta_color=delta_color)


def _plot_real_vs_pred(df_diag):
    min_val = float(np.nanmin([df_diag["real"].min(), df_diag["prediccion"].min()]))
    max_val = float(np.nanmax([df_diag["real"].max(), df_diag["prediccion"].max()]))
    fig = px.scatter(
        df_diag,
        x="real",
        y="prediccion",
        color="error_absoluto",
        color_continuous_scale="Viridis",
        hover_data={"registro": True, "residual": ":.4f", "error_absoluto": ":.4f"},
        template=PLOTLY_TEMPLATE,
        opacity=0.72,
    )
    fig.add_trace(
        go.Scatter(
            x=[min_val, max_val],
            y=[min_val, max_val],
            mode="lines",
            name="Prediccion ideal",
            line=dict(color=COLOR_GOOD, dash="dash", width=2),
        )
    )
    fig.update_layout(height=520, margin=dict(l=10, r=10, t=35, b=10), coloraxis_colorbar_title="Error abs.")
    fig.update_xaxes(title="Valor real")
    fig.update_yaxes(title="Prediccion")
    return fig


def _plot_residual_hist(df_diag):
    fig = px.histogram(
        df_diag,
        x="residual",
        nbins=min(60, max(15, int(math.sqrt(len(df_diag))))),
        marginal="box",
        template=PLOTLY_TEMPLATE,
        color_discrete_sequence=[COLOR_INFO],
    )
    fig.add_vline(x=0, line_dash="dash", line_color=COLOR_GOOD)
    fig.update_layout(height=380, margin=dict(l=10, r=10, t=35, b=10))
    return fig


def _plot_residual_scatter(df_diag, x_col, x_title):
    fig = px.scatter(
        df_diag,
        x=x_col,
        y="residual",
        color="error_absoluto",
        color_continuous_scale="Turbo",
        hover_data={"registro": True, "real": ":.4f", "prediccion": ":.4f"},
        template=PLOTLY_TEMPLATE,
        opacity=0.72,
    )
    fig.add_hline(y=0, line_dash="dash", line_color=COLOR_GOOD)
    fig.update_layout(height=420, margin=dict(l=10, r=10, t=35, b=10), coloraxis_colorbar_title="Error abs.")
    fig.update_xaxes(title=x_title)
    fig.update_yaxes(title="Residual")
    return fig


def _plot_target_distribution(df_original, target):
    if df_original is None or not target or target not in df_original.columns:
        return None
    target_series = pd.to_numeric(df_original[target], errors="coerce").dropna()
    if target_series.empty:
        return None
    fig = px.histogram(
        x=target_series,
        nbins=min(70, max(15, int(math.sqrt(len(target_series))))),
        template=PLOTLY_TEMPLATE,
        color_discrete_sequence=[COLOR_GOOD],
    )
    fig.update_layout(height=390, margin=dict(l=10, r=10, t=35, b=10), showlegend=False)
    fig.update_xaxes(title=target)
    fig.update_yaxes(title="Frecuencia")
    return fig


def _plot_horizontal_bar(df, x, y, color=None, title=None):
    fig = px.bar(
        df.sort_values(x, ascending=True),
        x=x,
        y=y,
        orientation="h",
        color=color,
        template=PLOTLY_TEMPLATE,
        color_continuous_scale="Viridis",
        title=title,
    )
    fig.update_layout(height=max(360, min(640, 28 * len(df) + 120)), margin=dict(l=10, r=10, t=45, b=10))
    return fig


def _render_model_comparison(metricas):
    validacion = metricas.get("validacion", {}) if isinstance(metricas, dict) else {}
    comparacion = validacion.get("comparacion_modelos", [])
    if not comparacion:
        _empty_state("No hay comparacion de candidatos disponible para este entrenamiento.")
        return
    df_comp = pd.DataFrame(comparacion)
    if df_comp.empty:
        _empty_state("No hay datos comparativos para mostrar.")
        return
    score_col = "mejor_score_r2" if "mejor_score_r2" in df_comp.columns else "mejor_score"
    df_comp[score_col] = pd.to_numeric(df_comp[score_col], errors="coerce")
    df_comp = df_comp.sort_values(score_col, ascending=False).reset_index(drop=True)
    df_comp.insert(0, "ranking", np.arange(1, len(df_comp) + 1))
    best = metricas.get("modelo_seleccionado")
    df_comp["seleccionado"] = np.where(df_comp["modelo"].eq(best), "Mejor modelo", "Candidato")

    st.dataframe(df_comp, use_container_width=True, hide_index=True)
    fig = px.bar(
        df_comp,
        x="modelo",
        y=score_col,
        color="seleccionado",
        text=score_col,
        template=PLOTLY_TEMPLATE,
        color_discrete_map={"Mejor modelo": COLOR_GOOD, "Candidato": COLOR_INFO},
    )
    fig.update_traces(texttemplate="%{text:.4f}", textposition="outside")
    fig.update_layout(height=420, margin=dict(l=10, r=10, t=35, b=10), yaxis_title="R2 validacion cruzada")
    st.plotly_chart(fig, use_container_width=True)


def _automatic_insights(metricas, df_diag, coef_df):
    precision = metricas.get("metricas_precision", {}) if isinstance(metricas, dict) else {}
    r2 = _safe_float(precision.get("R2"))
    rmse = _safe_float(precision.get("RMSE"))
    mape = _safe_float(precision.get("MAPE"))
    insights = []

    if r2 is not None:
        if r2 >= 0.75:
            insights.append(("success", "El modelo presenta buen ajuste general segun R2."))
        elif r2 >= 0.45:
            insights.append(("warning", "El ajuste es moderado; puede mejorar con variables adicionales o transformaciones."))
        else:
            insights.append(("error", "El ajuste global es bajo; la relacion lineal podria no capturar suficiente informacion."))

    if mape is not None:
        if mape <= 0.10:
            insights.append(("success", "El error porcentual promedio es bajo."))
        elif mape <= 0.25:
            insights.append(("warning", "El error porcentual es aceptable, pero conviene revisar segmentos con mayor desviacion."))
        else:
            insights.append(("error", "El MAPE es alto; revisa escala del target, outliers o no linealidad."))

    if not df_diag.empty:
        abs_error = df_diag["error_absoluto"]
        q95 = abs_error.quantile(0.95)
        median = abs_error.median()
        if median > 0 and q95 / median > 4:
            insights.append(("warning", "Hay errores extremos: el percentil 95 supera varias veces el error mediano."))
        corr = df_diag[["prediccion", "residual"]].corr().iloc[0, 1]
        if pd.notna(corr) and abs(corr) > 0.30:
            insights.append(("warning", "Los residuales muestran patron frente a la prediccion; podria existir sesgo o no linealidad."))
        if rmse is not None and median > 0 and rmse / median > 2:
            insights.append(("warning", "RMSE es mucho mayor que el error mediano, senal de observaciones dificiles u outliers."))

    if not coef_df.empty:
        top = coef_df.iloc[0]
        direction = "positiva" if top["coeficiente"] >= 0 else "negativa"
        insights.append(("info", f"La variable con mayor influencia lineal es {top['variable']} con relacion {direction}."))

    if not insights:
        insights.append(("info", "No se detectaron alertas fuertes con la informacion disponible."))
    return insights


def _render_insight_cards(insights):
    for level, text in insights:
        color = {
            "success": COLOR_GOOD,
            "warning": COLOR_WARN,
            "error": COLOR_BAD,
            "info": COLOR_INFO,
        }.get(level, COLOR_INFO)
        st.markdown(
            f"""
            <div class="lm-insight-card" style="border-left-color:{color}">
                <div class="lm-insight-dot" style="background:{color}"></div>
                <div>{text}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def _render_pipeline(resumen_pipeline, res, health):
    steps = [
        ("Validacion de columnas", True),
        ("Eliminacion / imputacion de nulos", resumen_pipeline.get("nulos_tratados", 0) > 0),
        ("Tratamiento de outliers", resumen_pipeline.get("outliers_corregidos", 0) > 0),
        ("Encoding categorico", resumen_pipeline.get("categoricas_codificadas", 0) > 0),
        ("Escalado numerico", True),
        ("PCA", bool(res.get("es_pca"))),
        ("Entrenamiento", True),
        ("Evaluacion", True),
    ]
    cols = st.columns(4)
    for idx, (label, active) in enumerate(steps):
        with cols[idx % 4]:
            state = "Completo" if active else "No aplicado"
            cls = "active" if active else "muted"
            st.markdown(
                f"<div class='lm-pipeline-step {cls}'><span>{'OK' if active else '--'}</span><strong>{label}</strong><small>{state}</small></div>",
                unsafe_allow_html=True,
            )

    st.markdown("#### Salud del dataset")
    hcols = st.columns(6)
    hcols[0].metric("% nulos", f"{health.get('nulos_pct', 0):.2f}%")
    hcols[1].metric("Outliers tratados", f"{health.get('outliers', 0):,}")
    hcols[2].metric("Numéricas", f"{health.get('columnas_numericas', 0):,}")
    hcols[3].metric("Categóricas", f"{health.get('columnas_categoricas', 0):,}")
    hcols[4].metric("Filas eliminadas", f"{health.get('filas_eliminadas', 0):,}")
    hcols[5].metric("Columnas eliminadas", f"{health.get('columnas_eliminadas', 0):,}")

    with st.expander("Reglas de preprocesamiento"):
        st.json(res.get("reglas", {}))
    logs = resumen_pipeline.get("logs", [])
    if logs:
        with st.expander("Logs tecnicos de limpieza"):
            st.dataframe(pd.DataFrame(logs), use_container_width=True, hide_index=True)


def render_linear_regression_dashboard(res, df_original, resumen_pipeline, ia_messages=None, on_predict=None):
    metricas = res.get("metricas", {})
    cols = res.get("cols", [])
    target = res.get("target")
    modelo = res.get("modelo")
    df_diag = _diagnostics_frame(metricas)
    coef_df = _coefficient_frame(modelo, cols)
    corr_df = _correlation_frame(df_original, target)
    health = _dataset_health(df_original, resumen_pipeline)

    st.markdown(
        "<div class='lm-dashboard-title'><span>Linear Regression AutoML</span><small>Diagnostico, explicabilidad y calidad operativa</small></div>",
        unsafe_allow_html=True,
    )
    _render_kpis(metricas, len(df_original) if df_original is not None else 0, len(cols))

    tab_metrics, tab_viz, tab_ai, tab_comp, tab_pred, tab_pipe = st.tabs(
        [
            "Metricas",
            "Visualizaciones",
            "Insights IA",
            "Comparacion de Modelos",
            "Predicciones",
            "Pipeline Ejecutado",
        ]
    )

    with tab_metrics:
        _section("Resumen de rendimiento", "Metricas principales y diagnostico estadistico del conjunto de validacion.")
        precision = metricas.get("metricas_precision", {})
        left, right = st.columns([1, 1])
        with left:
            st.dataframe(
                pd.DataFrame(
                    [{"metrica": k, "valor": v} for k, v in precision.items()]
                ),
                use_container_width=True,
                hide_index=True,
            )
        with right:
            if not df_diag.empty:
                err_summary = df_diag[["residual", "error_absoluto", "error_pct"]].describe().T
                st.dataframe(err_summary, use_container_width=True)
            else:
                _empty_state("No hay vectores de validacion para resumir errores.")

        _section("Variables usadas")
        st.dataframe(pd.DataFrame({"variable": cols}), use_container_width=True, hide_index=True)

    with tab_viz:
        if df_diag.empty:
            _empty_state("No hay datos de validacion para generar visualizaciones interactivas.")
        else:
            _section("Real vs prediccion", "La linea punteada representa una prediccion perfecta.")
            st.plotly_chart(_plot_real_vs_pred(df_diag), use_container_width=True)

            c1, c2 = st.columns(2)
            with c1:
                _section("Distribucion de residuales")
                st.plotly_chart(_plot_residual_hist(df_diag), use_container_width=True)
            with c2:
                _section("Distribucion del target")
                fig_target = _plot_target_distribution(df_original, target)
                if fig_target is not None:
                    st.plotly_chart(fig_target, use_container_width=True)
                else:
                    _empty_state("Target no disponible como numerico en el dataset original.")

            c3, c4 = st.columns(2)
            with c3:
                _section("Residuales vs prediccion")
                st.plotly_chart(_plot_residual_scatter(df_diag, "prediccion", "Prediccion"), use_container_width=True)
            with c4:
                _section("Residuales vs real")
                st.plotly_chart(_plot_residual_scatter(df_diag, "real", "Valor real"), use_container_width=True)

            _section("Top errores mas grandes")
            top_n = st.slider("Cantidad de errores a mostrar", min_value=5, max_value=min(50, len(df_diag)), value=min(10, len(df_diag)))
            top_errors = df_diag.nlargest(top_n, "error_absoluto")[
                ["registro", "real", "prediccion", "residual", "error_absoluto", "error_pct"]
            ]
            st.dataframe(top_errors, use_container_width=True, hide_index=True)

        c5, c6 = st.columns(2)
        with c5:
            _section("Correlaciones principales")
            if not corr_df.empty:
                st.plotly_chart(_plot_horizontal_bar(corr_df, "correlacion_abs", "variable", "correlacion"), use_container_width=True)
                st.dataframe(corr_df, use_container_width=True, hide_index=True)
            else:
                _empty_state("No hay suficientes columnas numericas originales para correlacionar con el target.")
        with c6:
            _section("Importancia por coeficientes")
            if not coef_df.empty:
                top_coef = coef_df.head(15)
                st.plotly_chart(_plot_horizontal_bar(top_coef, "magnitud_abs", "variable", "coeficiente"), use_container_width=True)
                st.dataframe(top_coef, use_container_width=True, hide_index=True)
            else:
                _empty_state("El modelo seleccionado no expone coeficientes compatibles.")

    with tab_ai:
        _section("Insights automaticos", "Reglas dinamicas generadas a partir de metricas, residuales y coeficientes.")
        _render_insight_cards(_automatic_insights(metricas, df_diag, coef_df))
        st.markdown("#### Interpretacion del agente")
        if ia_messages:
            st.markdown(ia_messages[0].get("content", ""))
        else:
            _empty_state("El agente aun no ha generado una interpretacion.")

    with tab_comp:
        _section("Ranking de candidatos", "Comparacion de modelos evaluados durante GridSearchCV.")
        _render_model_comparison(metricas)

    with tab_pred:
        _section("Predicciones", "Usa el modal existente para predecir desde texto o archivo con el mismo pipeline entrenado.")
        c1, c2 = st.columns([1, 2])
        with c1:
            if st.button("Abrir prediccion", use_container_width=True, type="primary"):
                if on_predict is not None:
                    on_predict()
        with c2:
            st.caption("Las nuevas filas pasan por transformaciones memorizadas: imputacion, encoding, escalado y PCA si aplica.")

        if not df_diag.empty:
            export_df = df_diag[["registro", "real", "prediccion", "residual", "error_absoluto", "error_pct"]]
            st.download_button(
                "Exportar diagnostico de validacion",
                data=export_df.to_csv(index=False).encode("utf-8"),
                file_name="diagnostico_regresion_lineal.csv",
                mime="text/csv",
                use_container_width=True,
            )

    with tab_pipe:
        _section("Pipeline ejecutado", "Resumen visual de transformaciones y salud del dataset.")
        _render_pipeline(resumen_pipeline, res, health)
