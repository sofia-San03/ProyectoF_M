import math
from pathlib import Path

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


def _problem_family(metricas, tipo_modelo):
    tipo = str(metricas.get("tipo_problema", "")).lower()
    modelo = str(tipo_modelo).lower()
    if "cluster" in tipo or "cluster" in modelo:
        return "clustering"
    if "regresion" in tipo or "regression" in tipo:
        return "regression"
    return "classification"


def _section(title, subtitle=None):
    st.markdown(f"### {title}")
    if subtitle:
        st.caption(subtitle)


def _empty(text):
    st.info(text)


def _metric_color(name, value):
    value = _safe_float(value)
    if value is None:
        return COLOR_INFO
    key = str(name).lower()
    if key in {"r2", "accuracy", "precision", "recall", "f1-score", "roc-auc", "pr-auc", "silhouette score"}:
        if value >= 0.75:
            return COLOR_GOOD
        if value >= 0.45:
            return COLOR_WARN
        return COLOR_BAD
    if key in {"mape", "davies-bouldin index"}:
        if value <= 0.10:
            return COLOR_GOOD
        if value <= 0.30:
            return COLOR_WARN
        return COLOR_BAD
    return COLOR_INFO


def _render_kpis(items, columns=6):
    if not items:
        return
    cols = st.columns(min(columns, len(items)))
    for col, (label, value) in zip(cols, items):
        with col:
            st.markdown(
                f"<div class='lm-kpi-accent' style='border-color:{_metric_color(label, value)}'></div>",
                unsafe_allow_html=True,
            )
            if isinstance(value, (int, np.integer)) and label.lower() not in {"r2"}:
                shown = f"{int(value):,}"
            else:
                shown = _fmt(value)
            st.metric(label, shown)


def _diagnostics_frame(metricas):
    diag = metricas.get("diagnostico", {}) if isinstance(metricas, dict) else {}
    y_true = diag.get("y_test") or diag.get("y_true") or []
    y_pred = diag.get("y_pred") or []
    if not y_true or not y_pred or len(y_true) != len(y_pred):
        return pd.DataFrame()
    df = pd.DataFrame({"real": y_true, "prediccion": y_pred})
    df["registro"] = np.arange(1, len(df) + 1)
    real_num = pd.to_numeric(df["real"], errors="coerce")
    pred_num = pd.to_numeric(df["prediccion"], errors="coerce")
    if real_num.notna().all() and pred_num.notna().all():
        df["real"] = real_num
        df["prediccion"] = pred_num
        df["residual"] = df["real"] - df["prediccion"]
        df["error_absoluto"] = df["residual"].abs()
        df["error_pct"] = np.where(df["real"].abs() > 1e-9, df["error_absoluto"] / df["real"].abs(), np.nan)
    return df


def _probability_frame(metricas):
    diag = metricas.get("diagnostico", {}) if isinstance(metricas, dict) else {}
    y_score = diag.get("y_score") or diag.get("probabilidades") or []
    if not y_score:
        return pd.DataFrame()
    series = pd.Series(y_score)
    return pd.DataFrame({"registro": np.arange(1, len(series) + 1), "probabilidad": pd.to_numeric(series, errors="coerce")}).dropna()


def _coef_or_importance_frame(modelo, cols):
    if modelo is None or not cols:
        return pd.DataFrame()
    values = None
    kind = "coeficiente"
    try:
        if hasattr(modelo, "feature_importances_"):
            values = np.asarray(modelo.feature_importances_).flatten()
            kind = "importancia"
        elif hasattr(modelo, "coef_"):
            values = np.asarray(modelo.coef_).flatten()
            kind = "coeficiente"
        elif hasattr(modelo, "regressor_") and hasattr(modelo.regressor_, "coef_"):
            values = np.asarray(modelo.regressor_.coef_).flatten()
            kind = "coeficiente"
    except Exception:
        values = None
    if values is None or len(values) != len(cols):
        return pd.DataFrame()
    df = pd.DataFrame({"variable": cols, kind: values})
    df["magnitud_abs"] = np.abs(values)
    df["tipo"] = kind
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


def _dataset_health(df_original, resumen_pipeline, res):
    if df_original is None or df_original.empty:
        return {}
    logs = resumen_pipeline.get("logs", []) if isinstance(resumen_pipeline, dict) else []
    rows_removed = 0
    for log in logs:
        detail = str(log.get("detalle", ""))
        if "filas" in detail:
            try:
                rows_removed += int(detail.split()[0])
            except Exception:
                pass
    cardinality = df_original.nunique(dropna=True)
    return {
        "nulos_pct": float(df_original.isna().mean().mean() * 100),
        "columnas_numericas": int(df_original.select_dtypes(include=[np.number]).shape[1]),
        "columnas_categoricas": int(df_original.select_dtypes(include=["object", "category", "string", "bool"]).shape[1]),
        "cardinalidad_media": float(cardinality.mean()) if not cardinality.empty else 0,
        "features_generadas": int(max(0, len(res.get("cols", [])) - df_original.shape[1])),
        "filas_eliminadas": rows_removed,
        "outliers": resumen_pipeline.get("outliers_corregidos", 0),
        "columnas_eliminadas": len(resumen_pipeline.get("columnas_eliminadas", [])),
        "pca": bool(res.get("es_pca")),
    }


def _plot_bar(df, x, y, color=None, title=None):
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
    fig.update_layout(height=max(360, min(680, 30 * len(df) + 120)), margin=dict(l=10, r=10, t=45, b=10))
    return fig


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
    fig.add_trace(go.Scatter(x=[min_val, max_val], y=[min_val, max_val], mode="lines", name="Ideal", line=dict(color=COLOR_GOOD, dash="dash")))
    fig.update_layout(height=520, margin=dict(l=10, r=10, t=35, b=10), coloraxis_colorbar_title="Error abs.")
    return fig


def _plot_hist(series, title, color=COLOR_INFO, nbins=None):
    series = pd.to_numeric(pd.Series(series), errors="coerce").dropna()
    if series.empty:
        return None
    fig = px.histogram(
        x=series,
        nbins=nbins or min(70, max(15, int(math.sqrt(len(series))))),
        template=PLOTLY_TEMPLATE,
        color_discrete_sequence=[color],
        title=title,
    )
    fig.update_layout(height=390, margin=dict(l=10, r=10, t=45, b=10), showlegend=False)
    return fig


def _render_existing_visuals(visualizaciones):
    if not isinstance(visualizaciones, dict) or not visualizaciones:
        return
    shown = 0
    cols = st.columns(2)
    for key, path in visualizaciones.items():
        if not path:
            continue
        with cols[shown % 2]:
            if Path(str(path)).exists():
                st.image(path, caption=str(key).replace("_", " ").title(), use_container_width=True)
            else:
                st.caption(f"{key}: {path}")
        shown += 1


def _render_regression(res, df_original, resumen_pipeline, ia_messages=None):
    metricas = res.get("metricas", {})
    precision = metricas.get("metricas_precision", {})
    df_diag = _diagnostics_frame(metricas)
    coef_df = _coef_or_importance_frame(res.get("modelo"), res.get("cols", []))
    corr_df = _correlation_frame(df_original, res.get("target"))
    _render_kpis(
        [
            ("R2", precision.get("R2")),
            ("RMSE", precision.get("RMSE")),
            ("MAE", precision.get("MAE")),
            ("MAPE", precision.get("MAPE")),
            ("Variables", len(res.get("cols", []))),
            ("Registros", len(df_original) if df_original is not None else 0),
        ]
    )
    tab_resumen, tab_viz, tab_interp, tab_ia, tab_pipe = st.tabs(["Resumen", "Visualizaciones", "Interpretabilidad", "Insights IA", "Pipeline"])
    with tab_resumen:
        _section("Metricas de regresion")
        st.dataframe(pd.DataFrame([{"metrica": k, "valor": v} for k, v in precision.items()]), use_container_width=True, hide_index=True)
        _render_leaderboard(metricas)
    with tab_viz:
        if not df_diag.empty and {"residual", "error_absoluto"}.issubset(df_diag.columns):
            st.plotly_chart(_plot_real_vs_pred(df_diag), use_container_width=True)
            c1, c2 = st.columns(2)
            with c1:
                st.plotly_chart(_plot_hist(df_diag["residual"], "Distribucion de residuales", COLOR_INFO), use_container_width=True)
            with c2:
                st.plotly_chart(_plot_hist(df_diag["error_absoluto"], "Error absoluto", COLOR_WARN), use_container_width=True)
            c3, c4 = st.columns(2)
            with c3:
                fig = px.scatter(df_diag, x="prediccion", y="residual", color="error_absoluto", template=PLOTLY_TEMPLATE, opacity=0.75)
                fig.add_hline(y=0, line_dash="dash", line_color=COLOR_GOOD)
                st.plotly_chart(fig, use_container_width=True)
            with c4:
                fig = px.scatter(df_diag, x="real", y="residual", color="error_absoluto", template=PLOTLY_TEMPLATE, opacity=0.75)
                fig.add_hline(y=0, line_dash="dash", line_color=COLOR_GOOD)
                st.plotly_chart(fig, use_container_width=True)
            st.dataframe(df_diag.nlargest(min(20, len(df_diag)), "error_absoluto"), use_container_width=True, hide_index=True)
        else:
            _render_existing_visuals(metricas.get("visualizaciones", {}))
    with tab_interp:
        _render_importance_and_correlations(coef_df, corr_df)
        _render_insights(_regression_insights(precision, df_diag, coef_df))
    with tab_ia:
        _render_ai_insights(ia_messages, _regression_insights(precision, df_diag, coef_df), "regresion")
    with tab_pipe:
        _render_pipeline(resumen_pipeline, res, df_original)


def _render_classification(res, df_original, resumen_pipeline, ia_messages=None):
    metricas = res.get("metricas", {})
    precision = metricas.get("metricas_precision", {})
    matriz = metricas.get("matriz_confusion")
    df_diag = _diagnostics_frame(metricas)
    prob_df = _probability_frame(metricas)
    coef_df = _coef_or_importance_frame(res.get("modelo"), res.get("cols", []))
    n_classes = len(np.unique(df_diag["real"])) if not df_diag.empty and "real" in df_diag else None
    if n_classes is None and matriz is not None:
        n_classes = len(matriz)
    _render_kpis(
        [
            ("Accuracy", precision.get("Accuracy")),
            ("Precision", precision.get("Precision")),
            ("Recall", precision.get("Recall")),
            ("F1-Score", precision.get("F1-Score")),
            ("ROC-AUC", precision.get("ROC-AUC")),
            ("Clases", n_classes or 0),
        ]
    )
    tab_resumen, tab_viz, tab_interp, tab_ia, tab_pipe = st.tabs(["Resumen", "Visualizaciones", "Interpretabilidad", "Insights IA", "Pipeline"])
    with tab_resumen:
        _section("Metricas de clasificacion")
        st.dataframe(pd.DataFrame([{"metrica": k, "valor": v} for k, v in precision.items()]), use_container_width=True, hide_index=True)
        _render_leaderboard(metricas)
    with tab_viz:
        if matriz is not None:
            z = np.asarray(matriz)
            fig = px.imshow(z, text_auto=True, template=PLOTLY_TEMPLATE, color_continuous_scale="Blues", title="Matriz de confusion")
            fig.update_layout(height=450, margin=dict(l=10, r=10, t=45, b=10))
            st.plotly_chart(fig, use_container_width=True)
        if not df_diag.empty:
            c1, c2 = st.columns(2)
            with c1:
                st.plotly_chart(px.histogram(df_diag, x="real", template=PLOTLY_TEMPLATE, title="Clases reales"), use_container_width=True)
            with c2:
                st.plotly_chart(px.histogram(df_diag, x="prediccion", template=PLOTLY_TEMPLATE, title="Predicciones"), use_container_width=True)
            errors = df_diag[df_diag["real"].astype(str) != df_diag["prediccion"].astype(str)]
            st.dataframe(errors.head(100), use_container_width=True, hide_index=True)
        if not prob_df.empty:
            st.plotly_chart(px.histogram(prob_df, x="probabilidad", template=PLOTLY_TEMPLATE, title="Distribucion de probabilidades"), use_container_width=True)
            _render_binary_curves(prob_df)
        _render_existing_visuals(metricas.get("visualizaciones", {}))
    with tab_interp:
        _render_importance_and_correlations(coef_df, pd.DataFrame())
        _render_insights(_classification_insights(precision, matriz, df_diag))
    with tab_ia:
        _render_ai_insights(ia_messages, _classification_insights(precision, matriz, df_diag), "clasificacion")
    with tab_pipe:
        _render_pipeline(resumen_pipeline, res, df_original)


def _render_clustering(res, df_original, resumen_pipeline, ia_messages=None):
    metricas = res.get("metricas", {})
    precision = metricas.get("metricas_precision", {})
    perfiles = res.get("perfiles", {})
    _render_kpis(
        [
            ("Silhouette Score", precision.get("Silhouette Score")),
            ("Davies-Bouldin Index", precision.get("Davies-Bouldin Index")),
            ("Clusters", metricas.get("mejor_numero_clusters")),
            ("Inercia", precision.get("Inercia")),
            ("Distancia promedio", precision.get("Distancia Euclidiana Media")),
            ("Variables", len(res.get("cols", []))),
        ]
    )
    tab_resumen, tab_viz, tab_perfiles, tab_ia, tab_pipe = st.tabs(["Resumen", "Visualizaciones", "Perfiles", "Insights IA", "Pipeline"])
    with tab_resumen:
        st.dataframe(pd.DataFrame([{"metrica": k, "valor": v} for k, v in precision.items()]), use_container_width=True, hide_index=True)
        _render_leaderboard(metricas)
        _render_insights(_clustering_insights(precision, perfiles))
    with tab_viz:
        _render_existing_visuals(metricas.get("visualizaciones", {}))
        comps = metricas.get("metricas_por_algoritmo", [])
        if comps:
            rows = []
            for item in comps:
                row = {"algoritmo": item.get("algoritmo"), "clusters": item.get("clusters")}
                row.update(item.get("metricas", {}))
                rows.append(row)
            df = pd.DataFrame(rows)
            st.dataframe(df, use_container_width=True, hide_index=True)
            if "Silhouette Score" in df.columns:
                st.plotly_chart(px.bar(df, x="algoritmo", y="Silhouette Score", color="clusters", template=PLOTLY_TEMPLATE), use_container_width=True)
    with tab_perfiles:
        _render_cluster_profiles(perfiles)
    with tab_ia:
        _render_ai_insights(ia_messages, _clustering_insights(precision, perfiles), "clustering")
    with tab_pipe:
        _render_pipeline(resumen_pipeline, res, df_original)


def _render_neural_network_extra(metricas):
    visualizaciones = metricas.get("visualizaciones", {})
    if visualizaciones.get("perdida"):
        st.image(visualizaciones["perdida"], caption="Evolucion de loss", use_container_width=True)
    _render_insights([("info", "Revisa la curva de perdida: una estabilizacion gradual sugiere convergencia; oscilaciones fuertes sugieren sensibilidad a hiperparametros o escala.")])


def _render_importance_and_correlations(coef_df, corr_df):
    c1, c2 = st.columns(2)
    with c1:
        _section("Importancia de variables")
        if not coef_df.empty:
            top = coef_df.head(20)
            st.plotly_chart(_plot_bar(top, "magnitud_abs", "variable", top.columns[1]), use_container_width=True)
            st.dataframe(top, use_container_width=True, hide_index=True)
        else:
            _empty("El estimador no expone coeficientes ni feature_importances_.")
    with c2:
        _section("Correlaciones con target")
        if not corr_df.empty:
            st.plotly_chart(_plot_bar(corr_df, "correlacion_abs", "variable", "correlacion"), use_container_width=True)
            st.dataframe(corr_df, use_container_width=True, hide_index=True)
        else:
            _empty("No hay correlaciones numericas disponibles para el target.")


def _render_leaderboard(metricas):
    validacion = metricas.get("validacion", {}) if isinstance(metricas, dict) else {}
    rows = validacion.get("comparacion_modelos") or metricas.get("metricas_por_algoritmo") or []
    if not rows:
        return
    df = pd.DataFrame(rows)
    if "metricas" in df.columns:
        expanded = pd.json_normalize(df["metricas"]).add_prefix("")
        df = pd.concat([df.drop(columns=["metricas"]), expanded], axis=1)
    score_cols = [c for c in ["mejor_score_r2", "mejor_score", "Silhouette Score", "Accuracy", "R2"] if c in df.columns]
    if score_cols:
        score_col = score_cols[0]
        df[score_col] = pd.to_numeric(df[score_col], errors="coerce")
        df = df.sort_values(score_col, ascending=False)
        st.markdown("#### Leaderboard AutoML")
        st.dataframe(df, use_container_width=True, hide_index=True)
        label_col = "modelo" if "modelo" in df.columns else "algoritmo" if "algoritmo" in df.columns else df.columns[0]
        st.plotly_chart(px.bar(df, x=label_col, y=score_col, template=PLOTLY_TEMPLATE, color=score_col, color_continuous_scale="Viridis"), use_container_width=True)


def _render_binary_curves(prob_df):
    if prob_df.empty:
        return
    sorted_df = prob_df.sort_values("probabilidad", ascending=False).reset_index(drop=True)
    sorted_df["percentil"] = (np.arange(len(sorted_df)) + 1) / len(sorted_df)
    sorted_df["lift_proxy"] = sorted_df["probabilidad"] / max(sorted_df["probabilidad"].mean(), 1e-9)
    fig = px.line(sorted_df, x="percentil", y="lift_proxy", template=PLOTLY_TEMPLATE, title="Lift proxy por score")
    st.plotly_chart(fig, use_container_width=True)


def _render_cluster_profiles(perfiles):
    if not perfiles:
        _empty("No hay perfilamiento de clusters disponible.")
        return
    for cluster_id, info in perfiles.items():
        with st.expander(f"Cluster {cluster_id}", expanded=False):
            st.write(f"Tamano: {info.get('tamaño_grupo', info.get('tamano_grupo', 'N/D'))}")
            nums = info.get("estadisticas_numericas", {})
            if nums:
                st.dataframe(pd.DataFrame(nums).T.head(30), use_container_width=True)
            cats = info.get("distribucion_categorias", {})
            if cats:
                st.json(cats)


def _render_pipeline(resumen_pipeline, res, df_original):
    steps = [
        ("Validacion", True),
        ("Nulos", resumen_pipeline.get("nulos_tratados", 0) > 0),
        ("Outliers", resumen_pipeline.get("outliers_corregidos", 0) > 0),
        ("Encoding", resumen_pipeline.get("categoricas_codificadas", 0) > 0),
        ("Escalado", True),
        ("PCA", bool(res.get("es_pca"))),
        ("Entrenamiento", True),
        ("Evaluacion", True),
    ]
    cols = st.columns(4)
    for idx, (name, active) in enumerate(steps):
        with cols[idx % 4]:
            cls = "active" if active else "muted"
            st.markdown(f"<div class='lm-pipeline-step {cls}'><span>{'OK' if active else '--'}</span><strong>{name}</strong><small>{'Aplicado' if active else 'No aplicado'}</small></div>", unsafe_allow_html=True)
    health = _dataset_health(df_original, resumen_pipeline, res)
    _section("Salud del dataset")
    _render_kpis(
        [
            ("% nulos", health.get("nulos_pct", 0)),
            ("Outliers", health.get("outliers", 0)),
            ("Numericas", health.get("columnas_numericas", 0)),
            ("Categoricas", health.get("columnas_categoricas", 0)),
            ("Filas eliminadas", health.get("filas_eliminadas", 0)),
            ("Features gen.", health.get("features_generadas", 0)),
        ]
    )
    with st.expander("Reglas y logs"):
        st.json(res.get("reglas", {}))
        logs = resumen_pipeline.get("logs", [])
        if logs:
            st.dataframe(pd.DataFrame(logs), use_container_width=True, hide_index=True)


def _render_insights(insights):
    for level, text in insights:
        color = {"success": COLOR_GOOD, "warning": COLOR_WARN, "error": COLOR_BAD, "info": COLOR_INFO}.get(level, COLOR_INFO)
        st.markdown(f"<div class='lm-insight-card' style='border-left-color:{color}'><div class='lm-insight-dot' style='background:{color}'></div><div>{text}</div></div>", unsafe_allow_html=True)


def _render_ai_insights(ia_messages, automatic_insights, family):
    st.markdown("### Insights automaticos")
    st.caption("Alertas dinamicas generadas a partir de metricas, comportamiento del modelo y senales del pipeline.")
    _render_insights(_expand_featured_alerts(automatic_insights, family))

    st.markdown("#### Interpretacion de Gemini")
    if not ia_messages:
        st.warning("La interpretacion IA aun no esta disponible. Espera a que Autopilot termine de analizar el modelo o revisa la configuracion de Gemini.")
        return

    content = str(ia_messages[0].get("content", "")).strip()
    if not content:
        st.warning("Gemini devolvio una respuesta vacia. Intenta regenerar el analisis desde el asistente.")
        return

    first_lines = [line.strip() for line in content.splitlines() if line.strip()]
    executive = first_lines[0] if first_lines else "Analisis IA disponible"
    st.markdown(
        f"""
        <div class="lm-ai-summary">
            <strong>Conclusion ejecutiva</strong>
            <span>{executive}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )
    with st.container(border=True):
        st.markdown(content)
    with st.expander("Contexto: como usar estos insights"):
        st.markdown(
            """
            - Usa la conclusion ejecutiva para explicar el valor del modelo a negocio.
            - Contrasta las recomendaciones con las metricas y graficas del dashboard.
            - Si el modelo muestra alertas, prioriza revisar datos, variables y segmentaciones antes de productivizar.
            """
        )


def _expand_featured_alerts(insights, family):
    expanded = list(insights or [])
    templates = {
        "regresion": [
            ("info", "Compara RMSE y MAE: una brecha grande suele indicar errores extremos."),
            ("warning", "Revisa residuales vs prediccion para detectar no linealidad o sesgo."),
            ("info", "Las variables de mayor magnitud ayudan a explicar que factores mueven la prediccion."),
        ],
        "clasificacion": [
            ("warning", "Revisa la matriz de confusion para ubicar las clases con mayor error."),
            ("info", "Precision y recall deben interpretarse segun el costo de falsos positivos y falsos negativos."),
            ("warning", "Si una clase domina la distribucion, valida desbalanceo antes de productivizar."),
        ],
        "clustering": [
            ("info", "Contrasta Silhouette y Davies-Bouldin para evaluar separacion y compacidad."),
            ("warning", "Clusters con tamanos muy distintos pueden requerir segmentacion adicional."),
            ("info", "Usa los perfiles promedio para convertir grupos en segmentos accionables."),
        ],
    }
    expanded.extend(templates.get(family, []))

    seen = set()
    unique = []
    for level, text in expanded:
        key = str(text).strip().lower()
        if key and key not in seen:
            seen.add(key)
            unique.append((level, text))
    return unique[:8]


def _regression_insights(precision, df_diag, coef_df):
    out = []
    r2 = _safe_float(precision.get("R2"))
    if r2 is not None:
        out.append(("success" if r2 >= 0.75 else "warning" if r2 >= 0.45 else "error", "Ajuste general alto." if r2 >= 0.75 else "Ajuste moderado; conviene revisar variables o no linealidad." if r2 >= 0.45 else "Ajuste bajo; el patron lineal podria ser insuficiente."))
    if not df_diag.empty and "error_absoluto" in df_diag:
        median = df_diag["error_absoluto"].median()
        q95 = df_diag["error_absoluto"].quantile(.95)
        if median > 0 and q95 / median > 4:
            out.append(("warning", "Se detectan errores extremos en la cola superior."))
    if not coef_df.empty:
        out.append(("info", f"Variable mas influyente: {coef_df.iloc[0]['variable']}."))
    return out or [("info", "No se detectaron alertas fuertes.")]


def _classification_insights(precision, matriz, df_diag):
    out = []
    f1 = _safe_float(precision.get("F1-Score"))
    if f1 is not None:
        out.append(("success" if f1 >= .75 else "warning" if f1 >= .45 else "error", "Balance precision-recall solido." if f1 >= .75 else "F1 moderado; revisa umbral, clases o variables." if f1 >= .45 else "F1 bajo; el modelo confunde clases relevantes."))
    if matriz is not None:
        arr = np.asarray(matriz)
        if arr.size:
            errors = arr.sum() - np.trace(arr)
            out.append(("info", f"Errores fuera de diagonal: {int(errors):,}."))
    if not df_diag.empty:
        counts = df_diag["real"].value_counts(normalize=True)
        if not counts.empty and counts.max() > .75:
            out.append(("warning", "Posible desbalanceo de clases en la muestra de validacion."))
    return out or [("info", "No se detectaron alertas fuertes.")]


def _clustering_insights(precision, perfiles):
    out = []
    sil = _safe_float(precision.get("Silhouette Score"))
    if sil is not None:
        out.append(("success" if sil >= .5 else "warning" if sil >= .25 else "error", "Clusters bien separados." if sil >= .5 else "Separacion moderada; los segmentos pueden solaparse." if sil >= .25 else "Separacion debil entre clusters."))
    if perfiles:
        sizes = []
        for info in perfiles.values():
            size = info.get("tamaño_grupo", info.get("tamano_grupo"))
            if isinstance(size, (int, float)):
                sizes.append(size)
        if sizes and max(sizes) / max(min(sizes), 1) > 5:
            out.append(("warning", "Los clusters estan desbalanceados en tamano."))
    return out or [("info", "No se detectaron alertas fuertes.")]


def render_model_dashboard(res, df_original, resumen_pipeline, ia_messages=None, on_predict=None):
    metricas = res.get("metricas", {})
    family = _problem_family(metricas, res.get("tipo_modelo"))
    title = {
        "regression": "Dashboard AutoML de Regresion",
        "classification": "Dashboard AutoML de Clasificacion",
        "clustering": "Dashboard AutoML de Clustering",
    }[family]
    st.markdown(
        f"<div class='lm-dashboard-title'><span>{title}</span><small>{res.get('tipo_modelo')} · resultados dinamicos e interpretables</small></div>",
        unsafe_allow_html=True,
    )
    if family == "clustering":
        _render_clustering(res, df_original, resumen_pipeline, ia_messages=ia_messages)
    elif family == "regression":
        _render_regression(res, df_original, resumen_pipeline, ia_messages=ia_messages)
        if str(res.get("tipo_modelo")).lower() == "redes_neuronales":
            _render_neural_network_extra(metricas)
    else:
        _render_classification(res, df_original, resumen_pipeline, ia_messages=ia_messages)
        if str(res.get("tipo_modelo")).lower() == "redes_neuronales":
            _render_neural_network_extra(metricas)
    if on_predict is not None and family != "clustering":
        st.markdown("### Predicciones")
        if st.button("Abrir prediccion", type="primary", use_container_width=True):
            on_predict()
