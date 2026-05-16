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
    """Abre el explorador de archivos y devuelve el DataFrame del archivo seleccionado."""
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

def AnalizarDatos(df):
    from ydata_profiling import ProfileReport
    profile = ProfileReport(df, title="Análisis Exploratorio Autopilot", explorative=True)
    return profile.to_html()


if __name__ == "__main__":
    CargarDatos()
    df= pd.read_csv("data/movies.csv")
    AnalizarDatos(df)