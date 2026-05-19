# ⚡ Data Mining Autopilot

Este proyecto es una aplicación de **Streamlit** de nivel empresarial diseñada para automatizar y optimizar todo el ciclo de vida de la ciencia de datos. Combina técnicas avanzadas de Machine Learning, procesamiento robusto en la nube con **Google Cloud Platform (GCP)** y la orquestación estratégica de modelos predictivos mediante **Inteligencia Artificial (Gemini)**.

---

## 🚀 Novedades y Capacidades Avanzadas

### 1. Arquitectura Modular Desacoplada (Frontend Limpio)
* **Frontend Puro (`app_simple.py`)**: Rediseñado para actuar exclusivamente como una capa de interfaz de usuario sumamente rápida y ligera.
* **Módulo de Backend (`CODIGO/Funcionalidades.py`)**: Centraliza todas las constantes de configuración de nube, conectores a bases de datos, lógica de comunicación externa y formateadores avanzados de métricas.

### 2. Integración y Orquestación Serverless con GCP
* **Google Cloud Storage (GCS)**: Carga ágil y segura del dataset de hechos principal y tablas dimensionales directamente al bucket `"archivos_back"`.
* **Google BigQuery**: Ingesta automatizada y sincronizada de los datos limpios y procesados en el dataset `"Cubo"`, permitiendo consultas optimizadas de alto rendimiento.
* **Orquestación en Cloud Run**: Ejecución y entrenamiento pesado delegados a un microservicio en la nube (`armar-cubo`), liberando de carga computacional a la memoria local de la aplicación y eliminando los límites de RAM de Streamlit.

### 3. Predicción Conversacional mediante Agentes de IA
* **Traducción de Lenguaje Natural**: Los usuarios pueden describir una situación de negocio en texto libre (ej. *"Tengo una casita en Miguel Hidalgo con 3 recámaras..."*).
* **Agente Estructurador**: Un agente de Gemini procesa el texto e interpreta las características complejas, mapeándolas automáticamente al esquema y dtypes exactos del dataset original.
* **Alineación Resiliente de Tipos (Dtype Alignment)**: Implementación de seguridad en `CleanData.py` que previene que valores nulos numéricos (como `latitud` o `longitud`) sean dummificados por error al ingresar registros por chat. Las variables se alinean con sus tipos de entrenamiento, garantizando predicciones geográficas y numéricas 100% correctas.

### 4. Interfaz Premium de Alta Gama
* **Estética Aetheris UI**: Diseño visual minimalista con tema oscuro personalizado mediante HSL y CSS de precisión.
* **Micro-animaciones y Tarjetas Dinámicas**: Diseño interactivo sin bordes toscos, con estados visuales suaves para un look moderno y profesional.
* **Soporte de Carga de Gran Escala**: Configuración interna optimizada (`maxUploadSize = 1024` en `.streamlit/config.toml`) para permitir la carga local de datasets de hasta **1 GB** de capacidad.

---

## 🛠️ Estructura del Proyecto

* 💻 **`app_simple.py`**: Interfaz de usuario y orquestador principal del flujo del sistema (Carga, Propuesta, Ejecución y Resultados).
* ⚙️ **`CODIGO/Funcionalidades.py`**: Conectores de Google Cloud, cliente BigQuery, cliente GCS, APIs de Gemini, formateadores avanzados de dashboards y utilidades generales.
* 🧼 **`CODIGO/CleanData.py`**: Clase `Transformar_Df` encargada de la lematización NLP (Spacy), manejo de outliers, imputación, WOE, Target Encoding y normalización de inferencia.
* 🤖 **`CODIGO/MODELS.py`**: Suite de modelos predictivos que incluye Regresión Lineal, Regresión Logística, Árboles de Decisión, Redes Neuronales MLP, KNN, Clustering y Credit Scoring con optimización GridSearchCV.
* 📊 **`CODIGO/CargarDatos.py`**: Clase `AnalizarDatos` para la generación de reportes interactivos de calidad de datos y análisis exploratorio (EDA) inmediato.
* 🔑 **`credenciales/`**: Carpeta segura para almacenar los archivos JSON de credenciales de Google Cloud (`BigQuery_credentials.json`).

---

## 🔀 Modos de operación

La aplicación detecta automáticamente el entorno al iniciar y elige el modo de operación adecuado. No se requiere configuración manual.

### 🟢 Modo Cloud (producción)

**Se activa cuando** existe alguna de estas condiciones:
- El archivo `credenciales/BigQuery_credentials.json` es un service account válido (`"type": "service_account"`).
- Las Application Default Credentials (ADC) están disponibles en el entorno (`gcloud auth application-default login` o credenciales de Workload Identity en GCP).

**Flujo de datos:**
1. Los archivos se suben a **Google Cloud Storage** (bucket `archivos_back`).
2. Un trigger en GCS carga los datos en **BigQuery** (dataset `Cubo`).
3. La **Cloud Function `armar-cubo`** une la tabla de hechos con las dimensiones y crea la vista `cubo_analitico`.
4. La app lee el cubo desde BigQuery como DataFrame.

**Cómo configurarlo:**
- Copia `credenciales/BigQuery_credentials.example.json` → `credenciales/BigQuery_credentials.json` y reemplaza los campos con los valores reales de tu service account.
- O bien ejecuta `gcloud auth application-default login` antes de iniciar la app.

### 🔵 Modo Local (desarrollo / fallback)

**Se activa cuando** no hay credenciales GCP ni ADC disponibles (entorno de desarrollo sin acceso a la nube).

**Flujo de datos:**
1. Los archivos subidos se leen directamente en memoria como DataFrames.
2. Si se suben dimensiones, se construye el cubo local con `LEFT JOIN` sobre columnas de nombre coincidente (equivalente simplificado de la Cloud Function).
3. El DataFrame resultante se almacena en sesión y el flujo continúa igual que en modo cloud.

**Funcionalidades que siguen activas en modo local:**
- Chat estratégico con Gemini (requiere `GEMINI_KEY.txt`).
- Todos los modelos de ML (limpieza, entrenamiento, predicción).
- Reportes exploratorios de calidad de datos.
- Predicción desde texto o archivo.

> **Nota:** El cubo construido localmente usa un JOIN genérico. Si la Cloud Function `armar-cubo` aplica transformaciones adicionales, el resultado puede diferir del cubo en producción.

> **Forzar modo local manualmente:** ejecuta `export FORCE_LOCAL_MODE=1` antes de iniciar la app para saltarte la detección de credenciales y operar siempre en modo local. Útil cuando la Cloud Function tiene bugs o se quiere desarrollar sin dependencias cloud.

---

## ⚙️ Instrucciones de Configuración y Uso

### 1. Requisitos Previos
* **Python 3.9+** instalado en tu sistema.
* Una clave API de **Google Gemini** guardada en un archivo llamado `GEMINI_KEY.txt` en la raíz del proyecto, o declarada como variable de entorno `GEMINI_API_KEY`.
* Un archivo de credenciales de GCP en `credenciales/BigQuery_credentials.json` con permisos de escritura/lectura en Storage, BigQuery y Cloud Run.

### 2. Instalación de Dependencias
Instala los requerimientos ejecutando en la terminal de la raíz del proyecto:
```bash
pip install -r requirements.txt
```

### 3. Ejecución de la Aplicación
Para iniciar el Data Mining Autopilot localmente, ejecuta:
```bash
python -m streamlit run app_simple.py
```
El servidor web local se abrirá en la dirección estándar `http://localhost:8501`.

---

## 📈 Alcance y Capacidades del Preprocesamiento
* **Limpieza de Cadenas**: Normalización Unicode, eliminación de caracteres no alfanuméricos y conversión a minúsculas.
* **Codificaciones Inteligentes**:
  * *Weight of Evidence (WOE)*: Agrupación en bins basada en la fuerza de predicción del target.
  * *Target Encoding*: Reemplazo de categorías de alta cardinalidad por promedios del target.
  * *Ordinal Encoding*: Asignación jerárquica programable.
* **Reducción PCA**: Generación automática de componentes principales cuando la densidad de variables es excesivamente alta.
