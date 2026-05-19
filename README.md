# Data Mining Autopilot

Aplicación empresarial de **Streamlit** que automatiza el ciclo completo de la ciencia de datos: carga de archivos, análisis exploratorio, limpieza inteligente, entrenamiento de modelos de ML e interpretación estratégica de resultados mediante **IA (Gemini)**. Soporta operación híbrida: procesamiento en la nube con **Google Cloud Platform (GCP)** o totalmente local sin dependencias externas.

---

## Tabla de contenidos

1. [Arquitectura del sistema](#arquitectura-del-sistema)
2. [Módulos principales](#módulos-principales)
3. [Modos de operación](#modos-de-operación)
4. [Guía de instalación y levantamiento](#guía-de-instalación-y-levantamiento)
5. [ETL — Esquema estrella (C5 CDMX)](#etl--esquema-estrella-c5-cdmx)
6. [Capacidades de preprocesamiento](#capacidades-de-preprocesamiento)
7. [Modelos de Machine Learning](#modelos-de-machine-learning)
8. [Guía técnica para la IA (`reglas_dict`)](#guía-técnica-para-la-ia-reglas_dict)
9. [Alcance y limitaciones](#alcance-y-limitaciones)

---

## Arquitectura del sistema

```
DataMiningAutopilotApp/
├── app_simple.py               # Frontend y orquestador principal
├── CODIGO/
│   ├── Funcionalidades.py      # Conectores GCP, clientes GCS/BQ, Gemini, formateo
│   ├── CleanData.py            # Pipeline de limpieza y transformación
│   ├── MODELS.py               # Suite de modelos de ML
│   └── CargarDatos.py          # EDA y reportes de calidad de datos
├── etl/
│   └── build_star_schema.py    # ETL: CSV crudo C5 → esquema estrella (5 dims + fact)
├── credenciales/
│   ├── BigQuery_credentials.json         # Service account GCP (no se versiona)
│   └── BigQuery_credentials.example.json # Plantilla de referencia
├── data/
│   ├── c5_raw/                 # Datos crudos (no versionados)
│   ├── c5_processed/           # Salida del ETL (no versionada)
│   └── cache/                  # Caché de sesión (no versionado)
├── Resultados/                 # Artefactos de modelos generados (dataset_limpio.xlsx, etc.)
├── .streamlit/config.toml      # Configuración de Streamlit (maxUploadSize = 2 GB)
├── requirements.txt
└── packages.txt
```

**Principio de diseño:** `app_simple.py` actúa como capa UI pura y delega toda la lógica de negocio a `CODIGO/Funcionalidades.py`. Esto mantiene el frontend liviano y facilita el testing independiente del backend.

---

## Módulos principales

| Archivo | Responsabilidad |
|---|---|
| `app_simple.py` | Interfaz Streamlit, detección automática de modo (cloud/local), orquestación del flujo completo |
| `CODIGO/Funcionalidades.py` | Constantes de configuración de nube, conectores BigQuery y GCS, cliente Gemini, formateadores de métricas para dashboard |
| `CODIGO/CleanData.py` | Clase `Transformar_Df`: normalización Unicode, outliers (IQR), imputación, lematización NLP (spaCy), WOE, Target Encoding, Ordinal Encoding, PCA, alineación resiliente de tipos (Dtype Alignment) |
| `CODIGO/MODELS.py` | Suite de modelos: Regresión Lineal, Logística, Árbol de Decisión, Redes Neuronales MLP, KNN, Clustering K-Means y Credit Scoring con optimización GridSearchCV |
| `CODIGO/CargarDatos.py` | Clase `AnalizarDatos`: reportes interactivos de calidad de datos y análisis exploratorio automático (EDA) |
| `etl/build_star_schema.py` | ETL standalone: procesa el dataset C5 CDMX (incidentes viales), construye esquema estrella con 5 tablas dimensionales y enriquece con datos meteorológicos (Open-Meteo API) |

---

## Modos de operación

La app detecta el entorno al iniciar y elige el modo automáticamente. No se requiere configuración manual.

### Modo Cloud (producción)

**Se activa cuando** se cumple al menos una condición:
- `credenciales/BigQuery_credentials.json` contiene un service account válido (`"type": "service_account"`).
- Las Application Default Credentials (ADC) están disponibles (`gcloud auth application-default login` o Workload Identity en GCP).

**Flujo de datos:**
1. Archivos subidos → **Google Cloud Storage** (bucket `archivos_back`).
2. Trigger GCS → ingesta en **BigQuery** (dataset `Cubo`).
3. **Cloud Function `armar-cubo`** une la tabla de hechos con las dimensiones y crea la vista `cubo_analitico`.
4. La app lee el cubo desde BigQuery como DataFrame.

### Modo Local (desarrollo / fallback)

**Se activa cuando** no hay credenciales GCP ni ADC.

**Flujo de datos:**
1. Archivos subidos se leen directamente en memoria como DataFrames.
2. Si se suben dimensiones, el cubo se construye localmente con `LEFT JOIN` sobre columnas de nombre coincidente.
3. El DataFrame resultante se almacena en sesión y el flujo continúa igual que en modo cloud.

**Funcionalidades activas en modo local:**
- Chat estratégico con Gemini (requiere `GEMINI_KEY.txt`).
- Todos los modelos de ML (limpieza, entrenamiento, predicción).
- Reportes exploratorios de calidad de datos.
- Predicción desde texto libre o archivo CSV.

> **Nota:** El cubo local usa un JOIN genérico. Si la Cloud Function `armar-cubo` aplica transformaciones adicionales, el resultado puede diferir del cubo en producción.

> **Forzar modo local manualmente:** `export FORCE_LOCAL_MODE=1` antes de iniciar la app. Útil cuando la Cloud Function tiene bugs o se quiere desarrollar sin dependencias cloud.

---

## Guía de instalación y levantamiento

### Requisitos previos

- **Python 3.9 o superior**
- `pip` actualizado (`pip install --upgrade pip`)
- (Opcional, para modo cloud) Google Cloud SDK instalado y autenticado

### 1. Clonar el repositorio

```bash
git clone <url-del-repositorio>
cd ProyectoF_M
```

### 2. Crear entorno virtual

Se recomienda encarecidamente usar un entorno virtual para evitar conflictos de dependencias:

```bash
python -m venv venv

# Linux / macOS
source venv/bin/activate

# Windows
venv\Scripts\activate
```

### 3. Instalar dependencias

```bash
pip install -r DataMiningAutopilotApp/requirements.txt
```

El archivo `requirements.txt` incluye automáticamente la descarga del modelo de español de spaCy (`es_core_news_sm`). Si la instalación falla por problemas de red con la URL de spaCy, instálalo manualmente:

```bash
pip install spacy
python -m spacy download es_core_news_sm
```

### 4. Configurar credenciales de Gemini

La app busca la clave de Gemini en dos lugares (en orden de prioridad):

1. **Archivo de texto** — crea `DataMiningAutopilotApp/credenciales/GEMINI_KEY.txt` con tu clave en la primera línea.
2. **Variable de entorno** — `export GEMINI_API_KEY="tu_clave_aqui"`.

Sin esta clave, el chat estratégico y la interpretación de resultados con IA no estarán disponibles, pero el resto de la app funciona normalmente.

### 5. Configurar credenciales de Google Cloud (solo modo cloud)

Copia la plantilla de ejemplo y reemplaza los campos con los valores reales de tu service account:

```bash
cp DataMiningAutopilotApp/credenciales/BigQuery_credentials.example.json \
   DataMiningAutopilotApp/credenciales/BigQuery_credentials.json
# Editar BigQuery_credentials.json con los datos reales del service account
```

Los campos que debes reemplazar son:
- `project_id`
- `private_key_id`
- `private_key`
- `client_email`
- `client_id`
- `client_x509_cert_url`

Alternativamente, autentícate con las ADC del SDK de Google:

```bash
gcloud auth application-default login
```

> **Seguridad:** `BigQuery_credentials.json` y `GEMINI_KEY.txt` están en `.gitignore` y **nunca deben commitearse** al repositorio.

### 6. Levantar la aplicación

Desde la raíz del repositorio, ejecutar:

```bash
cd DataMiningAutopilotApp
python -m streamlit run app_simple.py
```

La app abre automáticamente en el navegador en `http://localhost:8501`.

> **Tamaño máximo de carga:** configurado en `.streamlit/config.toml` a **2 GB** por archivo.

### 7. Flujo de uso recomendado

```
1. Subir dataset (CSV o XLSX)
        ↓
2. Ejecutar EDA — revisar calidad de datos y distribuciones
        ↓
3. Consultar a la IA — obtener propuesta de limpieza y modelo
        ↓
4. Ajustar reglas_dict si es necesario y confirmar
        ↓
5. Ejecutar pipeline de limpieza + entrenamiento
        ↓
6. Revisar métricas y visualizaciones del modelo
        ↓
7. Interpretar resultados con la IA (traducción a lenguaje de negocio)
        ↓
8. Predicción: subir CSV nuevo o describir caso en lenguaje natural por chat
```

---

## ETL — Esquema estrella (C5 CDMX)

El módulo `etl/build_star_schema.py` es un script **standalone** (independiente de la app) que transforma el dataset crudo de incidentes viales de la CDMX (C5) en un esquema estrella listo para cargar en BigQuery.

**Dimensiones generadas:**
- `dim_tiempo` — año, mes, día, día de la semana, hora
- `dim_ubicacion` — alcaldía, colonia, coordenadas validadas dentro del bounding box CDMX
- `dim_incidente` — tipo y clasificación del incidente
- `dim_respuesta` — cuerpo de atención y tiempo de respuesta
- `dim_clima` — temperatura, precipitación, visibilidad y código meteorológico (enriquecido con Open-Meteo API)

**Uso:**

```bash
# Desde la raíz del repositorio
python -m etl.build_star_schema \
    --input  DataMiningAutopilotApp/data/c5_raw/inViales_2022_2024.csv \
    --output DataMiningAutopilotApp/data/c5_processed \
    [--sample 50000]
```

El flag `--sample` es útil para pruebas rápidas con un subconjunto de datos. Los archivos de salida (CSV por tabla) se generan en el directorio `--output`.

> Los directorios `data/c5_raw/`, `data/c5_processed/` y `data/cache/` están en `.gitignore` para evitar que datasets pesados se versionen.

---

## Capacidades de preprocesamiento

El pipeline `CleanData.Transformar_Df` aplica, en orden:

| Paso | Descripción |
|---|---|
| Pre-validación | Detección de columnas UUID/hash, conversión automática string → numérico |
| Análisis de outliers | Detección y winsorización por IQR antes de imputar nulos |
| Imputación de nulos | Media, mediana, moda, valor fijo, drop-row o drop-column según configuración |
| Limpieza de texto | Normalización Unicode, eliminación de caracteres no alfanuméricos, minúsculas |
| Lematización NLP | spaCy `es_core_news_sm`: tokenización, remoción de stopwords, generación de variables de palabras |
| Codificación categórica | One-Hot (Dummies), Target Encoding, WOE+IV, Ordinal Encoding |
| Fechas | Extracción de año/mes/día/día_semana + columna `t_paso` (días desde inicio) para series de tiempo |
| Alineación de tipos | Dtype Alignment que previene la dummificación errónea de variables numéricas nulas en inferencia |
| Reducción dimensional | PCA automático cuando la densidad de variables es alta (`EsPCA: true`) |
| Escalado | `StandardScaler` aplicado antes del entrenamiento |

---

## Modelos de Machine Learning

| Modelo | Tipo de problema | Métricas principales |
|---|---|---|
| `Regresion_lineal` | Regresión continua | R², MSE, RMSE, MAE, MAPE |
| `Regresion_logistica` | Clasificación binaria | Accuracy, Precision, Recall, F1, ROC-AUC |
| `Arbol_decision` | Clasificación o regresión | Accuracy/R², Precision/RMSE, Recall/MAE, F1, ROC-AUC |
| `Redes_neuronales` (MLP) | Clasificación o regresión | Accuracy/R², F1, ROC-AUC |
| `KNN` | Clasificación o regresión | Accuracy/R², F1 |
| `Clustering_optimizacion` | No supervisado | Silhouette Score, Davies-Bouldin, Inercia |
| `Credit_scoring` | Clasificación binaria (crédito) | Accuracy, F1, ROC-AUC, KS, IV por variable |

Todos los modelos supervisados usan `GridSearchCV` con validación cruzada para optimización de hiperparámetros. Los artefactos del modelo (`.pkl`) y las visualizaciones se guardan en `Resultados/`.

---

## Guía técnica para la IA (`reglas_dict`)

El agente Gemini genera una configuración JSON que controla el pipeline de preprocesamiento. Esta sección documenta el contrato exacto que debe seguir esa configuración.

### Estructura del JSON

```json
{
  "target": "nombre_columna_objetivo",
  "modelo": "Regresion_lineal",
  "n_clusters": null,
  "reglas_dict": {
    "nombre_columna": {
      "metodo": "mean",
      "valorDefecto": null,
      "tolSustitucion": 0.05,
      "tolMantenerCols": 0.1,
      "Lematizar": false,
      "Dummies": false,
      "MaxDummies": 20,
      "TargetEncoding": false,
      "WOE": false,
      "bins_woe": 5,
      "Ordinal": false,
      "orden": [],
      "Fecha": false
    }
  },
  "EsPCA": false
}
```

### Parámetros por columna

| Parámetro | Valores | Descripción |
|---|---|---|
| `metodo` | `"mean"`, `"median"`, `"mode"`, `"drop-values"`, `"drop-column"` | Estrategia de imputación. Usar `median` cuando hay outliers |
| `valorDefecto` | cualquier valor o `null` | Si se especifica, tiene prioridad sobre `metodo` |
| `tolSustitucion` | float (0-1) | Umbral máximo de nulos para intentar sustitución |
| `tolMantenerCols` | float (0-1) | Si la proporción de nulos supera este valor, la columna se elimina |
| `Lematizar` | `true`/`false` | Solo para texto libre: limpia, remueve stopwords y crea variables de palabras |
| `Dummies` | `true`/`false` | Fuerza One-Hot Encoding en variables categóricas |
| `MaxDummies` | entero | Límite de categorías; las menos frecuentes se agrupan en "otros" |
| `TargetEncoding` | `true`/`false` | Reemplaza categorías por la media del target (solo modelos supervisados) |
| `WOE` | `true`/`false` | Weight of Evidence: solo para problemas binarios supervisados |
| `bins_woe` | entero | Número de bins para calcular WOE/IV en variables numéricas |
| `Ordinal` | `true`/`false` | Codificación ordinal; requiere especificar `orden` |
| `orden` | lista de strings | Orden lógico de categorías, ej: `["bajo", "medio", "alto"]` |
| `Fecha` | `true`/`false` | Trata la columna como fecha, extrae año/mes/día/día_semana y crea `t_paso` |

### Restricciones y reglas de oro

- **No incluir** la columna `target` ni columnas ID en `reglas_dict`.
- `Clustering_optimizacion` requiere `"target": null`; no usar WOE ni TargetEncoding.
- `Regresion_logistica` y `Credit_scoring` requieren target con exactamente 2 clases.
- `Lematizar: true` es **obligatorio** en columnas con múltiples etiquetas separadas por `|` o `,`.
- Columnas con nombres de personas, RUT, DNI, correos o IDs siempre deben usar `"metodo": "drop-column"`.
- `WOE` no debe aplicarse a texto libre.
- `Credit_scoring`: priorizar `WOE: true` y `bins_woe: 5` en variables numéricas o categóricas de sentido crediticio.

### Ejemplo — Regresión lineal (precios)

```json
{
  "target": "precio_casa",
  "modelo": "Regresion_lineal",
  "reglas_dict": {
    "descripcion": { "Lematizar": true },
    "calidad": { "Ordinal": true, "orden": ["mala", "buena", "excelente"] },
    "alcaldia": { "TargetEncoding": true },
    "metros": { "metodo": "median" },
    "fecha_construccion": { "Fecha": true }
  },
  "EsPCA": false
}
```

### Ejemplo — Clustering

```json
{
  "target": null,
  "modelo": "Clustering_optimizacion",
  "n_clusters": 5,
  "reglas_dict": {
    "segmento": { "Dummies": true, "MaxDummies": 12 },
    "fecha_alta": { "metodo": "drop-column" },
    "ingreso": { "metodo": "median" }
  },
  "EsPCA": false
}
```

### Ejemplo — Credit Scoring

```json
{
  "target": "default",
  "modelo": "Credit_scoring",
  "reglas_dict": {
    "edad":                 { "WOE": true, "bins_woe": 5, "metodo": "median" },
    "ingresos":             { "WOE": true, "bins_woe": 5, "metodo": "median" },
    "historial_crediticio": { "Dummies": true, "MaxDummies": 10 },
    "nivel_riesgo":         { "Ordinal": true, "orden": ["bajo", "medio", "alto"] }
  },
  "EsPCA": false
}
```

---

## Alcance y limitaciones

### Qué puede hacer

- Carga de archivos CSV y XLSX (hasta 2 GB por archivo).
- EDA interactivo automático con `ydata-profiling`.
- Consultoría estratégica con Gemini: propuesta de limpieza y modelo personalizada.
- Pipeline de preprocesamiento completo (ver tabla de capacidades más arriba).
- Entrenamiento de 7 tipos de modelos con optimización de hiperparámetros.
- Interpretación de resultados en lenguaje de negocio mediante Gemini.
- Predicción desde CSV nuevo o desde descripción en lenguaje natural por chat.
- Operación completamente local sin acceso a internet (excepto funciones de IA).

### Limitaciones conocidas

- **Sin modelos de ensamble avanzados**: no incluye XGBoost, LightGBM, Random Forest ni Deep Learning.
- **Clasificación multiclase limitada**: el flujo principal está optimizado para targets binarios; árboles y MLP soportan multiclase pero sin métricas por clase.
- **Solo datos tabulares**: no procesa imágenes, audio, video ni texto no estructurado (salvo lematización dentro de columnas de un dataset tabular).
- **Escala in-memory**: basado en Pandas/Streamlit; datasets de decenas de millones de filas pueden superar la RAM disponible. Para Big Data usar el modo cloud con BigQuery.
- **Dependencia de API**: las funciones de IA (estrategia, interpretación, predicción por chat) requieren internet y clave válida de Gemini.
- **Cubo local simplificado**: el JOIN genérico en modo local puede diferir de la Cloud Function `armar-cubo` si esta aplica transformaciones adicionales.
