import os
import shutil
import tkinter as tk
from tkinter import filedialog, messagebox
import pandas as pd

def CargarDatos():
    root = tk.Tk()
    root.withdraw()

    file_paths = filedialog.askopenfilenames(
        title="Selecciona los archivos para cargar",
        filetypes=[("Todos los archivos", "*.*")]
    )

    if not file_paths:
        return

    data_dir = os.path.join(os.getcwd(), 'data')
    
    os.makedirs(data_dir, exist_ok=True)

    copied_files = 0
    for file_path in file_paths:
        try:
            filename = os.path.basename(file_path)
            destination = os.path.join(data_dir, filename)
            

            shutil.copy2(file_path, destination)
            copied_files += 1
        except Exception as e:
            pass

    if copied_files > 0:
        mensaje = f"Se han cargado correctamente {copied_files} archivo(s) en la carpeta 'data'."
    else:
        mensaje = "No se pudo copiar ningún archivo."

def seleccionar_y_cargar_df():
    """Abre el explorador de archivos y devuelve el DataFrame del archivo seleccionado."""
    root = tk.Tk()
    root.withdraw()
    file_path = filedialog.askopenfilename(
        title="Selecciona el archivo para analizar",
        filetypes=[("Archivos CSV", "*.csv"), ("Archivos Excel", "*.xlsx"), ("Archivos JSON", "*.json")]
    )
    if not file_path:
        return None
    
    if file_path.endswith('.csv'):
        return pd.read_csv(file_path)
    elif file_path.endswith('.json'):
        return pd.read_json(file_path)
    else:
        return pd.read_excel(file_path)

def obtener_dataframe_reciente():
    data_dir = os.path.join(os.getcwd(), 'data')
    if not os.path.exists(data_dir):
        return None
    files = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith(('.csv', '.xlsx', '.json'))]
    if not files:
        return None
    latest_file = max(files, key=os.path.getctime)
    if latest_file.endswith('.csv'):
        return pd.read_csv(latest_file)
    elif latest_file.endswith('.json'):
        return pd.read_json(latest_file)
    else:
        return pd.read_excel(latest_file)

def AnalizarDatos(df):
    from ydata_profiling import ProfileReport
    if len(df) > 100000:
        sample_size = 200000 if len(df) > 500000 else len(df)
        df_sample = df.sample(sample_size, random_state=42) if len(df) > sample_size else df
        profile = ProfileReport(df_sample, title="Análisis Exploratorio (Modo Optimizado)", minimal=True)
    else:
        profile = ProfileReport(df, title="Análisis Exploratorio Autopilot", explorative=True)
    return profile.to_html()


if __name__ == "__main__":
    CargarDatos()
    df= pd.read_csv("data/movies.csv")
    AnalizarDatos(df)
