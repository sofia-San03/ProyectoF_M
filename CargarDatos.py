import os
import shutil
import tkinter as tk
from tkinter import filedialog, messagebox
import pandas as pd

try:
    from ydata_profiling import ProfileReport
    _YDATA_AVAILABLE = True
except ModuleNotFoundError:
    _YDATA_AVAILABLE = False

def CargarDatos():
    root = tk.Tk()
    root.withdraw()

    file_paths = filedialog.askopenfilenames(
        title="Selecciona los archivos para cargar",
        filetypes=[("Todos los archivos", "*.*")]
    )

    if not file_paths:
        print("No se seleccionó ningún archivo. Operación cancelada.")
        return

    data_dir = os.path.join(os.getcwd(), 'data')
    os.makedirs(data_dir, exist_ok=True)
    print(f"Carpeta de destino: {data_dir}")

    copied_files = 0
    for file_path in file_paths:
        try:
            filename = os.path.basename(file_path)
            destination = os.path.join(data_dir, filename)
            shutil.copy2(file_path, destination)
            copied_files += 1
            print(f"Archivo copiado exitosamente: {filename}")
        except Exception as e:
            print(f"Error al copiar el archivo {file_path}: {e}")

    if copied_files > 0:
        mensaje = f"Se han cargado correctamente {copied_files} archivo(s) en la carpeta 'data'."
        print(mensaje)
    else:
        mensaje = "No se pudo copiar ningún archivo."
        print(mensaje)


def seleccionar_y_cargar_df():
    root = tk.Tk()
    root.withdraw()
    file_path = filedialog.askopenfilename(
        title="Selecciona el archivo CSV para analizar",
        filetypes=[("Archivos CSV", "*.csv"), ("Archivos Excel", "*.xlsx")]
    )
    if not file_path:
        print("No se seleccionó ningún archivo.")
        return None

    if file_path.endswith('.csv'):
        return pd.read_csv(file_path)
    else:
        return pd.read_excel(file_path)


def obtener_dataframe_reciente():
    data_dir = os.path.join(os.getcwd(), 'data')
    if not os.path.exists(data_dir):
        return None
    files = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith('.csv')]
    if not files:
        return None
    latest_file = max(files, key=os.path.getctime)
    print(f"Cargando archivo más reciente: {latest_file}")
    return pd.read_csv(latest_file)


def _fallback_profile_html(df: pd.DataFrame) -> str:
    shape     = df.shape
    nulls     = df.isnull().sum().reset_index()
    nulls.columns = ["Columna", "Nulos"]
    nulls["% Nulos"] = (nulls["Nulos"] / shape[0] * 100).round(2)

    describe_html = df.describe(include="all").T.to_html(classes="table", border=0)
    nulls_html    = nulls.to_html(index=False, classes="table", border=0)
    head_html     = df.head(10).to_html(index=False, classes="table", border=0)

    css = """
    <style>
      body { font-family: Arial, sans-serif; padding: 20px; background: #0f172a; color: #e2e8f0; }
      h2   { color: #10b981; }
      .table { border-collapse: collapse; width: 100%; margin-bottom: 20px; }
      .table th, .table td { border: 1px solid #334155; padding: 6px 10px; text-align: left; }
      .table th { background: #1e293b; color: #10b981; }
      .table tr:nth-child(even) { background: #1e293b55; }
    </style>
    """
    warn = (
        "<div style='background:#b45309;padding:10px;border-radius:8px;margin-bottom:16px;'>"
        "⚠️ <b>ydata-profiling no está instalado</b> — mostrando reporte básico. "
        "Instala con: <code>pip install ydata-profiling</code>"
        "</div>"
    )
    return f"""
    {css}
    {warn}
    <h2>📊 Reporte Exploratorio (Fallback)</h2>
    <p><b>Filas:</b> {shape[0]} &nbsp;|&nbsp; <b>Columnas:</b> {shape[1]}</p>
    <h2>🔍 Nulos por Columna</h2>{nulls_html}
    <h2>📈 Estadísticas Descriptivas</h2>{describe_html}
    <h2>👀 Primeras 10 Filas</h2>{head_html}
    """


def AnalizarDatos(df: pd.DataFrame) -> str:
    if _YDATA_AVAILABLE:
        profile = ProfileReport(
            df,
            title="Análisis Exploratorio Autopilot",
            explorative=True,
            # Mejoras de rendimiento para datasets grandes
            samples={"head": 10, "tail": 10},
            correlations={
                "auto": {"calculate": True},
                "pearson": {"calculate": True},
                "spearman": {"calculate": False},
                "kendall": {"calculate": False},
                "phi_k": {"calculate": False},
                "cramers": {"calculate": False},
            },
        )
        return profile.to_html()
    else:
        return _fallback_profile_html(df)

if __name__ == "__main__":
    CargarDatos()
    df = pd.read_csv("data/movies.csv")
    html = AnalizarDatos(df)
    print(html[:500])
