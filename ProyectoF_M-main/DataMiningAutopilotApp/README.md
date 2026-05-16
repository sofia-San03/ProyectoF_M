# ⚡ Data Mining Autopilot

Este proyecto es una aplicación de **Streamlit** diseñada para automatizar el ciclo de vida de la ciencia de datos, desde la carga y limpieza de datos hasta el entrenamiento de modelos de Machine Learning y la interpretación estratégica de los resultados utilizando Inteligencia Artificial.

## 🚀 Instrucciones de Uso

### Requisitos Previos
- Tener instalado Python.
- Configurar una clave de API de Google Gemini (puedes crear un archivo `GEMINI_KEY.txt` en la raíz con tu clave o configurarla en los Secrets de Streamlit como `GEMINI_API_KEY`).

### Ejecución
Para iniciar la aplicación, abre una terminal en la carpeta del proyecto y ejecuta el siguiente comando:

```bash
python -m streamlit run app_simple.py
```

La función principal y punto de entrada de la aplicación es `app_simple.py`.

---

## 🎯 Alcance del Proyecto

### ✅ ¿Qué puede hacer?
- **Carga Versátil**: Soporta archivos en formato `.csv` y `.xlsx`.
- **Análisis Exploratorio (EDA)**: Genera reportes interactivos detallados sobre la calidad y distribución de los datos de forma automática.
- **Consultoría de IA**: Integra un agente de IA (Gemini) que analiza los metadatos de tu archivo para proponer una estrategia de limpieza y modelado personalizada.
- **Preprocesamiento Automatizado**:
    - Manejo de valores atípicos (outliers).
    - Imputación inteligente de valores nulos (Media, Mediana, Moda, o eliminación).
    - Limpieza de texto, normalización y lematización mediante NLP (Spacy).
    - Codificación de variables categóricas (One-Hot Encoding, Target Encoding, Ordinal Encoding, WOE).
    - Escalado de datos y Reducción de Dimensionalidad (PCA).
- **Entrenamiento de Modelos**:
    - **Regresión Lineal**: Para predicción de valores continuos.
    - **Regresión Logística**: Para clasificación binaria.
    - **Árboles de Decisión**: Para clasificación binaria (incluye visualización del árbol).
    - Optimización de hiperparámetros mediante `GridSearchCV` y validación cruzada.
- **Interpretación Estratégica**: La IA traduce las métricas técnicas (R², Accuracy, F1, etc.) a lenguaje de negocio y recomendaciones accionables.
- **Predicciones en Tiempo Real**: Permite realizar predicciones sobre nuevos datos ingresados en formato CSV.

### ❌ ¿Qué no puede hacer? (Limitaciones)
- **Modelos Complejos**: Actualmente no soporta modelos de ensamble avanzados (XGBoost, Random Forest) o Deep Learning.
- **Clasificación Multiclase**: El flujo de clasificación actual está optimizado principalmente para variables objetivo binarias.
- **Datos no Tabulares**: No está diseñado para procesar imágenes, audio o video; se enfoca exclusivamente en datos estructurados (tablas).
- **Big Data**: Al ser una aplicación in-memory basada en Pandas y Streamlit, puede tener limitaciones de rendimiento con conjuntos de datos extremadamente grandes (millones de filas).
- **Dependencia de API**: Requiere una conexión activa a internet y una clave válida de Gemini para las funciones de estrategia e interpretación.

---

## 🛠️ Estructura Principal
- `app_simple.py`: Interfaz de usuario y orquestador principal.
- `CleanData.py`: Clase `Transformar_Df` encargada de toda la lógica de limpieza y transformación.
- `MODELS.py`: Funciones para el entrenamiento y evaluación de modelos de ML.
- `CargarDatos.py`: Lógica para el análisis inicial y generación de reportes EDA.
