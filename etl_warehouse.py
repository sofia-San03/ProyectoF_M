# etl_warehouse

import hashlib
import json
import logging
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import pandas as pd
import numpy as np

DWH_DIR  = Path("dwh")
DWH_PATH = DWH_DIR / "warehouse.db"
LOG_PATH = DWH_DIR / "etl.log"

DWH_DIR.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [ETL] %(levelname)s — %(message)s",
    handlers=[
        logging.FileHandler(LOG_PATH, encoding="utf-8"),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger("etl_warehouse")

# DDL Star Schema
_DDL = """
CREATE TABLE IF NOT EXISTS dim_dataset (
    dataset_id   INTEGER PRIMARY KEY AUTOINCREMENT,
    name         TEXT    NOT NULL,
    source_path  TEXT    DEFAULT '',
    file_hash    TEXT    UNIQUE,
    n_rows       INTEGER,
    n_cols       INTEGER,
    columns_json TEXT,
    loaded_at    TEXT    NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS dim_model (
    model_id      INTEGER PRIMARY KEY AUTOINCREMENT,
    model_name    TEXT    NOT NULL,
    model_type    TEXT,
    hyperparams   TEXT,
    registered_at TEXT    NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS dim_time (
    time_id   INTEGER PRIMARY KEY AUTOINCREMENT,
    full_date TEXT    NOT NULL UNIQUE,
    year      INTEGER,
    quarter   INTEGER,
    month     INTEGER,
    week      INTEGER,
    day       INTEGER,
    weekday   TEXT
);

CREATE TABLE IF NOT EXISTS dim_feature (
    feature_id   INTEGER PRIMARY KEY AUTOINCREMENT,
    dataset_id   INTEGER REFERENCES dim_dataset(dataset_id),
    feature_name TEXT    NOT NULL,
    dtype        TEXT,
    is_target    INTEGER DEFAULT 0,
    importance   REAL    DEFAULT NULL,
    created_at   TEXT    NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS dim_prediction (
    prediction_id   INTEGER PRIMARY KEY AUTOINCREMENT,
    analysis_id     INTEGER,
    input_json      TEXT,
    predicted_value TEXT,
    confidence      REAL    DEFAULT NULL,
    predicted_at    TEXT    NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS fact_analysis (
    analysis_id      INTEGER PRIMARY KEY AUTOINCREMENT,
    dataset_id       INTEGER REFERENCES dim_dataset(dataset_id),
    model_id         INTEGER REFERENCES dim_model(model_id),
    time_id          INTEGER REFERENCES dim_time(time_id),
    target_col       TEXT,
    task_type        TEXT,
    -- Clasificacion
    accuracy         REAL DEFAULT NULL,
    precision_val    REAL DEFAULT NULL,
    recall_val       REAL DEFAULT NULL,
    f1_score         REAL DEFAULT NULL,
    roc_auc          REAL DEFAULT NULL,
    -- Regresion
    rmse             REAL DEFAULT NULL,
    mae              REAL DEFAULT NULL,
    mape             REAL DEFAULT NULL,
    r2_score         REAL DEFAULT NULL,
    -- Clustering
    silhouette       REAL DEFAULT NULL,
    n_clusters       INTEGER DEFAULT NULL,
    -- Cross-Validation
    cv_mean          REAL DEFAULT NULL,
    cv_std           REAL DEFAULT NULL,
    cv_folds         INTEGER DEFAULT NULL,
    -- Features
    n_features_orig  INTEGER DEFAULT NULL,
    n_features_used  INTEGER DEFAULT NULL,
    cols_used_json   TEXT,
    -- Meta
    n_predictions    INTEGER DEFAULT 0,
    ai_conclusion    TEXT,
    execution_secs   REAL DEFAULT NULL,
    created_at       TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_fact_dataset  ON fact_analysis(dataset_id);
CREATE INDEX IF NOT EXISTS idx_fact_model    ON fact_analysis(model_id);
CREATE INDEX IF NOT EXISTS idx_fact_time     ON fact_analysis(time_id);
CREATE INDEX IF NOT EXISTS idx_pred_analysis ON dim_prediction(analysis_id);
CREATE INDEX IF NOT EXISTS idx_feat_dataset  ON dim_feature(dataset_id);
"""

def _m(metrics: Dict, key: str) -> Optional[float]:
    val = (
        metrics.get(key)
        or metrics.get("metricas_precision", {}).get(key)
    )
    try:
        return float(val) if val is not None else None
    except (TypeError, ValueError):
        return None


# Clase Principal
class ETLWarehouse:
    def __init__(self, db_path: Union[str, Path] = DWH_PATH):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()
        log.info("ETLWarehouse listo — DB: %s", self.db_path)

    def _conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, check_same_thread=False)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA foreign_keys=ON")
        conn.row_factory = sqlite3.Row
        return conn

    def _init_schema(self) -> None:
        with self._conn() as conn:
            conn.executescript(_DDL)

    @staticmethod
    def _hash_df(df: pd.DataFrame) -> str:
        raw = pd.util.hash_pandas_object(df, index=True).values.tobytes()
        return hashlib.sha256(raw).hexdigest()

    def _get_or_create_time(self, conn: sqlite3.Connection,
                            dt: Optional[datetime] = None) -> int:
        dt = dt or datetime.now()
        full_date = dt.strftime("%Y-%m-%d")
        row = conn.execute(
            "SELECT time_id FROM dim_time WHERE full_date=?", (full_date,)
        ).fetchone()
        if row:
            return row["time_id"]
        quarter = (dt.month - 1) // 3 + 1
        cur = conn.execute(
            """INSERT INTO dim_time(full_date,year,quarter,month,week,day,weekday)
               VALUES (?,?,?,?,?,?,?)""",
            (full_date, dt.year, quarter, dt.month,
             int(dt.strftime("%W")), dt.day, dt.strftime("%A")),
        )
        return cur.lastrowid

    # dim_dataset
    def load_dataset(self, df: pd.DataFrame, name: str, source_path: str = "") -> int:
        file_hash = self._hash_df(df)
        with self._conn() as conn:
            row = conn.execute(
                "SELECT dataset_id FROM dim_dataset WHERE file_hash=?",
                (file_hash,)
            ).fetchone()
            if row:
                log.info("Dataset '%s' ya registrado → id=%d", name, row["dataset_id"])
                return row["dataset_id"]

            cur = conn.execute(
                """INSERT INTO dim_dataset(name,source_path,file_hash,n_rows,n_cols,columns_json)
                   VALUES (?,?,?,?,?,?)""",
                (name, source_path, file_hash,
                 len(df), df.shape[1], json.dumps(df.columns.tolist())),
            )
            ds_id = cur.lastrowid
        log.info("Dataset '%s' registrado → id=%d (%dx%d)", name, ds_id, len(df), df.shape[1])
        return ds_id

    # dim_model
    def load_model(self, model_name: str, model_type: str = "",
                   hyperparams: Optional[Dict] = None) -> int:
        with self._conn() as conn:
            cur = conn.execute(
                "INSERT INTO dim_model(model_name,model_type,hyperparams) VALUES(?,?,?)",
                (model_name, model_type, json.dumps(hyperparams or {})),
            )
            model_id = cur.lastrowid
        log.info("Modelo '%s' registrado → id=%d", model_name, model_id)
        return model_id

    # dim_feature
    def load_features(self, dataset_id: int, df: pd.DataFrame,
                      target_col: str = "",
                      importances: Optional[Dict[str, float]] = None) -> None:
        importances = importances or {}
        rows = [
            (dataset_id, col, str(df[col].dtype),
             1 if col == target_col else 0, importances.get(col))
            for col in df.columns
        ]
        with self._conn() as conn:
            conn.executemany(
                """INSERT OR IGNORE INTO dim_feature
                   (dataset_id,feature_name,dtype,is_target,importance)
                   VALUES (?,?,?,?,?)""",
                rows,
            )
        log.info("%d features registradas para dataset_id=%d", len(rows), dataset_id)

    def save_analysis(
        self,
        dataset_id: int,
        model_id: int,
        target_col: str,
        task_type: str,
        metrics: Dict[str, Any],
        cols_used: List[str],
        ai_conclusion: str = "",
        execution_secs: Optional[float] = None,
        n_features_orig: Optional[int] = None,
    ) -> int:
        m = metrics
        sel = m.get("seleccion_variables", {})
        n_used = sel.get("cantidad_final") or len(cols_used)

        cv = (m.get("cross_validation_train_R2")
              or m.get("cross_validation_train_Accuracy")
              or {})

        silhouette_val = (
            _m(m, "Silueta")
            or _m(m, "silhouette_optimo")
            or _m(m, "silhouette")
        )

        n_clusters_val = m.get("k_optimo") or m.get("n_clusters")

        with self._conn() as conn:
            time_id = self._get_or_create_time(conn)
            cur = conn.execute(
                """INSERT INTO fact_analysis (
                    dataset_id,model_id,time_id,target_col,task_type,
                    accuracy,precision_val,recall_val,f1_score,roc_auc,
                    rmse,mae,mape,r2_score,
                    silhouette,n_clusters,
                    cv_mean,cv_std,cv_folds,
                    n_features_orig,n_features_used,cols_used_json,
                    ai_conclusion,execution_secs
                ) VALUES (
                    ?,?,?,?,?,
                    ?,?,?,?,?,
                    ?,?,?,?,
                    ?,?,
                    ?,?,?,
                    ?,?,?,
                    ?,?
                )""",
                (
                    dataset_id, model_id, time_id, target_col, task_type,
                    _m(m, "Accuracy"),  _m(m, "Precision"),
                    _m(m, "Recall"),    _m(m, "F1-Score"), _m(m, "ROC-AUC"),
                    _m(m, "RMSE"), _m(m, "MAE"), _m(m, "MAPE"), _m(m, "R2"),
                    silhouette_val, n_clusters_val,
                    cv.get("media"), cv.get("desviacion_estandar"), cv.get("folds"),
                    n_features_orig or sel.get("cantidad_original"),
                    n_used, json.dumps(cols_used),
                    ai_conclusion, execution_secs,
                ),
            )
            analysis_id = cur.lastrowid
        log.info("Análisis guardado → analysis_id=%d task=%s", analysis_id, task_type)
        return analysis_id

    # dim_prediction
    def save_predictions(self, analysis_id: int,
                         input_records: List[Dict],
                         predictions: List[Any],
                         confidences: Optional[List[float]] = None) -> None:
        confidences = confidences or [None] * len(predictions)
        rows = [
            (analysis_id, json.dumps(rec, default=str), str(pred), conf)
            for rec, pred, conf in zip(input_records, predictions, confidences)
        ]
        with self._conn() as conn:
            conn.executemany(
                """INSERT INTO dim_prediction(analysis_id,input_json,predicted_value,confidence)
                   VALUES (?,?,?,?)""",
                rows,
            )
            conn.execute(
                "UPDATE fact_analysis SET n_predictions=n_predictions+? WHERE analysis_id=?",
                (len(rows), analysis_id),
            )
        log.info("%d predicciones guardadas → analysis_id=%d", len(rows), analysis_id)

    def get_analysis_history(self, limit: int = 50) -> pd.DataFrame:
        sql = """
        SELECT
            f.analysis_id,
            d.name         AS dataset,
            m.model_name   AS modelo,
            t.full_date    AS fecha,
            f.task_type,
            f.target_col,
            ROUND(COALESCE(f.accuracy, f.r2_score, f.silhouette), 4) AS metrica_principal,
            f.n_features_orig,
            f.n_features_used,
            f.n_predictions,
            ROUND(f.execution_secs, 2)                               AS secs,
            SUBSTR(f.ai_conclusion, 1, 100)                          AS conclusion
        FROM fact_analysis f
        LEFT JOIN dim_dataset d ON f.dataset_id = d.dataset_id
        LEFT JOIN dim_model   m ON f.model_id   = m.model_id
        LEFT JOIN dim_time    t ON f.time_id    = t.time_id
        ORDER BY f.analysis_id DESC
        LIMIT ?
        """
        with self._conn() as conn:
            return pd.read_sql_query(sql, conn, params=(limit,))

    def get_best_models(self, task_type: str = "clasificacion",
                        metric: str = "accuracy", top_n: int = 5) -> pd.DataFrame:
        col_map = {
            "accuracy":   "f.accuracy",
            "f1":         "f.f1_score",
            "r2":         "f.r2_score",
            "rmse":       "f.rmse",
            "silhouette": "f.silhouette",
        }
        order_col = col_map.get(metric.lower(), "f.accuracy")
        order_dir = "ASC" if metric.lower() == "rmse" else "DESC"
        sql = f"""
        SELECT f.analysis_id, d.name AS dataset, m.model_name,
               t.full_date, {order_col} AS metrica
        FROM fact_analysis f
        LEFT JOIN dim_dataset d ON f.dataset_id = d.dataset_id
        LEFT JOIN dim_model   m ON f.model_id   = m.model_id
        LEFT JOIN dim_time    t ON f.time_id    = t.time_id
        WHERE LOWER(f.task_type) LIKE ? AND {order_col} IS NOT NULL
        ORDER BY {order_col} {order_dir}
        LIMIT ?
        """
        with self._conn() as conn:
            return pd.read_sql_query(sql, conn,
                                     params=(f"%{task_type.lower()}%", top_n))

    def get_predictions(self, analysis_id: int) -> pd.DataFrame:
        sql = """
        SELECT prediction_id, input_json, predicted_value, confidence, predicted_at
        FROM dim_prediction WHERE analysis_id=? ORDER BY prediction_id
        """
        with self._conn() as conn:
            return pd.read_sql_query(sql, conn, params=(analysis_id,))

    def get_dataset_stats(self) -> pd.DataFrame:
        sql = """
        SELECT d.dataset_id, d.name, d.n_rows, d.n_cols,
               d.loaded_at, COUNT(f.analysis_id) AS n_analisis
        FROM dim_dataset d
        LEFT JOIN fact_analysis f ON d.dataset_id = f.dataset_id
        GROUP BY d.dataset_id ORDER BY d.loaded_at DESC
        """
        with self._conn() as conn:
            return pd.read_sql_query(sql, conn)

    def export_star_schema_excel(self,
                                  output_path: str = "dwh/star_schema_export.xlsx") -> str:
        tables = ["fact_analysis", "dim_dataset", "dim_model",
                  "dim_time", "dim_feature", "dim_prediction"]
        with self._conn() as conn, \
             pd.ExcelWriter(output_path, engine="openpyxl") as writer:
            for t in tables:
                pd.read_sql_query(f"SELECT * FROM {t}", conn).to_excel(
                    writer, sheet_name=t[:31], index=False)
        log.info("Star Schema exportado → %s", output_path)
        return output_path


def registrar_resultado_en_warehouse(
    df: pd.DataFrame,
    dataset_name: str,
    model_name: str,
    model_type: str,
    target_col: str,
    task_type: str,
    metrics_json: Dict,
    cols_used: List[str],
    ai_conclusion: str = "",
    hyperparams: Optional[Dict] = None,
    execution_secs: Optional[float] = None,
) -> int:
    wh = ETLWarehouse()
    ds_id  = wh.load_dataset(df, name=dataset_name)
    mod_id = wh.load_model(model_name, model_type=model_type, hyperparams=hyperparams)
    wh.load_features(ds_id, df, target_col=target_col)
    an_id  = wh.save_analysis(
        dataset_id=ds_id,
        model_id=mod_id,
        target_col=target_col,
        task_type=task_type,
        metrics=metrics_json,
        cols_used=cols_used,
        ai_conclusion=ai_conclusion,
        execution_secs=execution_secs,
        n_features_orig=df.shape[1],
    )
    return an_id

if __name__ == "__main__":
    print("─── ETL Warehouse Smoke Test ───")
    df_test = pd.DataFrame({
        "edad":     np.random.randint(18, 65, 100),
        "ingresos": np.random.uniform(5000, 50000, 100),
        "churn":    np.random.randint(0, 2, 100),
    })

    an_id = registrar_resultado_en_warehouse(
        df=df_test, dataset_name="clientes_demo",
        model_name="Red Neuronal Clasificador", model_type="clasificacion",
        target_col="churn", task_type="clasificacion",
        metrics_json={
            "metricas_precision": {
                "Accuracy": 0.85, "Precision": 0.83,
                "Recall": 0.81, "F1-Score": 0.82,
                "Matriz_Confusion": [[42, 8], [7, 43]],
            },
            "cross_validation_train_Accuracy": {
                "media": 0.84, "desviacion_estandar": 0.02, "folds": 5
            }
        },
        cols_used=["edad", "ingresos"],
        ai_conclusion="MLP con accuracy > 0.85 en test.",
        execution_secs=3.12,
    )
    print(f"✅ Clasificación analysis_id: {an_id}")

    an_id_cl = registrar_resultado_en_warehouse(
        df=df_test, dataset_name="clientes_demo",
        model_name="KMedianas", model_type="clustering",
        target_col="", task_type="clustering",
        metrics_json={
            "modelo": "KMedianas",
            "k_optimo": 3,
            "Silueta": 0.4123,
            "silhouette_por_k": {2: 0.38, 3: 0.4123, 4: 0.35},
            "inercia_L1_por_k": {2: 120.5, 3: 98.2, 4: 85.1},
            "variables_utilizadas": ["edad", "ingresos"],
        },
        cols_used=["edad", "ingresos"],
        ai_conclusion="K=3 óptimo por Silhouette.",
        execution_secs=1.05,
    )
    print(f"✅ Clustering analysis_id: {an_id_cl}")

    wh = ETLWarehouse()
    print(wh.get_analysis_history().to_string(index=False))
    print(f"\n📊 Excel: {wh.export_star_schema_excel()}")
