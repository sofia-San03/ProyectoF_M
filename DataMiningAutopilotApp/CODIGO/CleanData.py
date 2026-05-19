import pandas as pd
import numpy as np
import unicodedata
import warnings
import nltk
import re
from nltk.corpus import stopwords
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import pandas.api.types as ptypes

try:
    import spacy
except ImportError:
    spacy = None

nltk.download('stopwords', quiet=True)

def pre_validar_columnas(df):
    df_validado = df.copy()
    logs = []

    for col in df_validado.columns:
        try:
            serie = df_validado[col]
            no_nulos = serie.dropna()
            total_no_nulos = len(no_nulos)

            if total_no_nulos == 0:
                logs.append({
                    "columna": col,
                    "accion": "posible_columna_basura",
                    "estado": "warning",
                    "detalle": "Columna completamente vacia"
                })
                continue

            valores_texto = no_nulos.astype(str).str.strip()
            unique_ratio = no_nulos.nunique(dropna=True) / total_no_nulos if total_no_nulos else 0
            patron_uuid_hash = valores_texto.str.match(
                r"^[a-fA-F0-9]{32}$|^[a-fA-F0-9]{40}$|^[a-fA-F0-9]{64}$|^[a-fA-F0-9]{8}-[a-fA-F0-9]{4}-[a-fA-F0-9]{4}-[a-fA-F0-9]{4}-[a-fA-F0-9]{12}$"
            ).mean()

            if unique_ratio > 0.95 or patron_uuid_hash > 0.80:
                logs.append({
                    "columna": col,
                    "accion": "posible_columna_basura",
                    "estado": "warning",
                    "detalle": "Alta cardinalidad, UUID o hash probable"
                })

            if ptypes.is_object_dtype(serie) or ptypes.is_string_dtype(serie):
                numerica = pd.to_numeric(serie, errors="coerce")
                ratio_numerico = numerica.notna().sum() / total_no_nulos
                if ratio_numerico >= 0.80:
                    df_validado[col] = numerica
                    logs.append({
                        "columna": col,
                        "accion": "convertido_string_a_numerico",
                        "estado": "success"
                    })
                    continue

                # Optimización: comprobar muestra antes de parsear toda la columna (evita warnings y lentitud)
                muestra = no_nulos.head(100)
                es_probable_fecha = False
                if not muestra.empty:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore", UserWarning)
                        try:
                            fecha_muestra = pd.to_datetime(muestra, errors="coerce", format="mixed")
                        except TypeError:
                            fecha_muestra = pd.to_datetime(muestra, errors="coerce")
                    ratio_fecha_muestra = fecha_muestra.notna().sum() / len(muestra)
                    if ratio_fecha_muestra >= 0.50:
                        es_probable_fecha = True
                
                if es_probable_fecha:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore", UserWarning)
                        try:
                            fecha = pd.to_datetime(serie, errors="coerce", format="mixed")
                        except TypeError:
                            fecha = pd.to_datetime(serie, errors="coerce")
                    ratio_fecha = fecha.notna().sum() / total_no_nulos
                    if ratio_fecha >= 0.70:
                        df_validado[col] = fecha
                        logs.append({
                            "columna": col,
                            "accion": "convertido_string_a_fecha",
                            "estado": "success"
                        })
                        continue

                tiene_numeros = valores_texto.str.match(r"^-?\d+(\.\d+)?$").any()
                tiene_texto = valores_texto.str.contains(r"[A-Za-z]", regex=True).any()
                if tiene_numeros and tiene_texto:
                    df_validado[col] = serie.astype("string")
                    logs.append({
                        "columna": col,
                        "accion": "columna_mixta_convertida_a_texto",
                        "estado": "success"
                    })
        except Exception as e:
            logs.append({
                "columna": col,
                "accion": "pre_validacion",
                "estado": "error",
                "detalle": str(e)
            })

    return df_validado, logs

class Transformar_Df:
    def __init__(self, dataFrame, col_target=None, id_column=None, idioma='spanish', modelo_nlp='es_core_news_sm'):
        self.df = dataFrame.copy()
        
        # Validar existencia de la columna objetivo cuando el flujo sea supervisado.
        if col_target and col_target not in self.df.columns:
            columnas_disponibles = list(self.df.columns)
            raise ValueError(f"La columna objetivo '{col_target}' no se encuentra en el DataFrame. "
                             f"Columnas disponibles: {columnas_disponibles}")
        
        # Validar existencia de la columna ID (si se proporciona)
        if id_column and id_column not in self.df.columns:
            columnas_disponibles = list(self.df.columns)
            raise ValueError(f"La columna ID '{id_column}' no se encuentra en el DataFrame. "
                             f"Columnas disponibles: {columnas_disponibles}")

        self.col_target_name = col_target 
        self.id_column = id_column
        self.y = None
        
        self.stop_words = set(stopwords.words(idioma))
        self.nlp = self._cargar_modelo_nlp(modelo_nlp)
        self.scaler = None
        self.pca = None
        self.target_encodings = {}
        self.columnas_entrenamiento = None
        self.dtypes_entrenamiento = {}
        
        # --- VARIABLES DE "MEMORIA" PARA INFERENCIA ---
        self.reglas_entrenamiento = {}
        self.imputaciones_nulos = {}
        self.backup_imputaciones = {}
        self.categorias_validas = {}
        self.columnas_texto_separadas = {}
        self.woe_mappings = {} 
        self.ordinal_mappings = {} 
        self.fecha_referencia = {}
        self.logs_limpieza = []
        self.df_limpio_usuario = None

    def _registrar_log(self, columna, accion, estado="success", detalle=None):
        log = {
            "columna": columna,
            "accion": accion,
            "estado": estado
        }
        if detalle is not None:
            log["detalle"] = str(detalle)
        self.logs_limpieza.append(log)

    def _cargar_modelo_nlp(self, modelo_nlp):
        if spacy is None:
            return None
        try:
            return spacy.load(modelo_nlp)
        except Exception:
            try:
                return spacy.blank("es")
            except Exception:
                return None

    def separar_columna(self, columna_nombre, umbral=0.05):
        df_dummies = self.df[columna_nombre].str.get_dummies(sep=' ')
        frecuencias = df_dummies.mean()
        columnas_a_mantener = frecuencias[frecuencias >= umbral].index.tolist()
        
        self.columnas_texto_separadas[columna_nombre] = columnas_a_mantener
        
        df_filtrado = df_dummies[columnas_a_mantener]
        self.df = pd.concat([self.df.drop(columns=[columna_nombre]), df_filtrado], axis=1)

    def lematizar(self, Registro):
        if self.nlp is None:
            return self.ExtraerTexto(Registro)
        doc = self.nlp(str(Registro))
        lemas = [(token.lemma_ if token.lemma_ else token.text) for token in doc if token.is_alpha]
        return " ".join(lemas) 

    def ExtraerTexto(self, Registro):
        palabras = str(Registro).split()
        palabras = [p for p in palabras if p not in self.stop_words]
        return " ".join(palabras)

    def Limpiar_solo_cadenas(self, columna, lematizar=False):
        if columna not in self.df.columns: return
        try:
            self.df[columna] = self.df[columna].astype("string")
        except Exception as e:
            self._registrar_log(columna, "limpieza_texto", "error", e)
            return
        self.df[columna] = self.df[columna].apply(lambda x: unicodedata.normalize('NFKD', str(x)).encode('ASCII', 'ignore').decode('utf-8') if pd.notnull(x) else x)
        self.df[columna] = self.df[columna].str.lower()
        self.df[columna] = self.df[columna].apply(lambda x: re.sub(r'[^a-zA-Z0-9áéíóúÁÉÍÓÚñÑ\s]', ' ', str(x)) if pd.notnull(x) else x)
        
        if lematizar:
            self.df[columna] = self.df[columna].apply(self.ExtraerTexto)
            self.df[columna] = self.df[columna].apply(self.lematizar)
            self.separar_columna(columna)

    def Manejo_Atipicos(self, columna):
        try:
            serie = pd.to_numeric(self.df[columna], errors='coerce')
            Q1 = serie.quantile(0.25)
            Q3 = serie.quantile(0.75)
            IQR = Q3 - Q1

            if pd.isna(IQR) or IQR == 0:
                self._registrar_log(columna, "outliers_sin_cambios", "success")
                return

            lower = Q1 - 1.5 * IQR
            upper = Q3 + 1.5 * IQR
            outliers = int(((serie < lower) | (serie > upper)).sum())
            self.df[columna] = serie.clip(lower, upper)
            self._registrar_log(columna, "outliers_recortados", "success", f"{outliers} valores recortados")
        except Exception as e:
            self._registrar_log(columna, "outliers_recortados", "error", e)

    def _ajustar_tipo_numerico_post_imputacion(self, columna, original):
        if not ptypes.is_numeric_dtype(self.df[columna]):
            return

        if len(original) > 0 and (original % 1 == 0).all():
            self.df[columna] = self.df[columna].round().astype("Int64")
        else:
            self.df[columna] = self.df[columna].astype(float)

    def No_nulos(self, columna, metodo=None, toleranciaSustitucion=0.05, toleranciaMantenerColumna=0.1, valorPorDefecto=None):
        try:
            if columna not in self.df.columns:
                return {'columna': columna, 'metodo': 'No encontrada', 'Valor_de_relleno': None}

            original_numerico = self.df[columna].dropna() if ptypes.is_numeric_dtype(self.df[columna]) else None

            if metodo == 'drop-column':
                self.df.drop(columns=[columna], inplace=True)
                self._registrar_log(columna, "eliminacion_columna", "success")
                return {'columna': columna, 'metodo': 'drop-column', 'Valor_de_relleno': None}

            total_rows = len(self.df)
            null_count = int(self.df[columna].isna().sum())
            proporcion_nulls = null_count / total_rows if total_rows > 0 else 0

            if total_rows == 0 or null_count == total_rows:
                self.df.drop(columns=[columna], inplace=True)
                self._registrar_log(columna, "eliminacion_columna_vacia", "success")
                return {'columna': columna, 'metodo': 'drop-column', 'Valor_de_relleno': None}

            if null_count == 0:
                self.imputaciones_nulos[columna] = None
                if original_numerico is not None:
                    self._ajustar_tipo_numerico_post_imputacion(columna, original_numerico)
                self._registrar_log(columna, "sin_nulos", "success")
                return {'columna': columna, 'metodo': 'Ninguno', 'Valor_de_relleno': None}

            if proporcion_nulls >= toleranciaMantenerColumna:
                self.df.drop(columns=[columna], inplace=True)
                self._registrar_log(columna, "eliminacion_columna_por_nulos", "success", f"{proporcion_nulls:.2%} nulos")
                return {'columna': columna, 'metodo': 'drop-column', 'Valor_de_relleno': None}

            if valorPorDefecto is not None:
                self.df[columna] = self.df[columna].fillna(valorPorDefecto)
                if original_numerico is not None:
                    self._ajustar_tipo_numerico_post_imputacion(columna, original_numerico)
                self.imputaciones_nulos[columna] = valorPorDefecto
                self.backup_imputaciones[columna] = valorPorDefecto
                self._registrar_log(columna, "imputacion_valor_preestablecido", "success")
                return {'columna': columna, 'metodo': 'valor_preestablecido', 'Valor_de_relleno': valorPorDefecto}

            es_numerica = ptypes.is_numeric_dtype(self.df[columna])
            chosen = metodo
            if chosen is None:
                if proporcion_nulls >= toleranciaSustitucion:
                    chosen = 'drop-values'
                elif es_numerica:
                    skew = self.df[columna].dropna().skew()
                    chosen = 'mean' if abs(skew) < 1 else 'median'
                else:
                    chosen = 'mode'

            if not es_numerica and chosen in ['mean', 'median']:
                chosen = 'mode'

            fill_value = None
            if es_numerica:
                skew = self.df[columna].dropna().skew()
                self.backup_imputaciones[columna] = self.df[columna].mean() if abs(skew) < 1 else self.df[columna].median()
            else:
                modas_backup = self.df[columna].mode(dropna=True)
                self.backup_imputaciones[columna] = modas_backup.iloc[0] if not modas_backup.empty else "Desconocido"

            if chosen == 'drop-values':
                self.df.dropna(subset=[columna], inplace=True)
                self._registrar_log(columna, "eliminacion_filas_nulas", "success", f"{null_count} filas")
            elif es_numerica:
                if chosen == 'median':
                    fill_value = self.df[columna].median()
                    accion = "imputacion_mediana"
                else:
                    fill_value = self.df[columna].mean()
                    accion = "imputacion_media"
                self.df[columna] = self.df[columna].fillna(fill_value)
                if original_numerico is not None:
                    self._ajustar_tipo_numerico_post_imputacion(columna, original_numerico)
                self._registrar_log(columna, accion, "success")
            else:
                modas = self.df[columna].mode(dropna=True)
                fill_value = modas.iloc[0] if not modas.empty else "Desconocido"
                self.df[columna] = self.df[columna].fillna(fill_value)
                chosen = 'mode'
                self._registrar_log(columna, "imputacion_moda", "success")

            self.imputaciones_nulos[columna] = fill_value
            return {'columna': columna, 'metodo': chosen, 'Valor_de_relleno': fill_value}
        except Exception as e:
            self._registrar_log(columna, "imputacion", "error", e)
            return {'columna': columna, 'metodo': 'error-imputacion', 'Valor_de_relleno': None}


    def calcular_woe_iv(self, df_temp, col, target_name):

        target_series = pd.to_numeric(df_temp[target_name], errors='coerce').fillna(0).astype(float)
        df_temp[target_name] = target_series
        
        stats = df_temp.groupby(col)[target_name].agg(['count', 'sum'])
        stats.columns = ['Total', 'Events']
        
        stats = stats.astype(float)
        stats['NonEvents'] = stats['Total'] - stats['Events']
        
        total_events = stats['Events'].sum()
        total_non_events = stats['NonEvents'].sum()
        
        dist_events = stats['Events'] / (float(total_events) if total_events > 0 else 1.0)
        dist_non_events = stats['NonEvents'] / (float(total_non_events) if total_non_events > 0 else 1.0)
        
        dist_events = dist_events.replace(0, 0.0001)
        dist_non_events = dist_non_events.replace(0, 0.0001)
        
        woe_series = np.log(dist_non_events / dist_events)
        iv_series = (dist_non_events - dist_events) * woe_series
        
        iv_total = float(iv_series.sum())
        woe_dict = woe_series.to_dict()
        
        return woe_dict, iv_total

    def aplicar_woe_columna(self, col, bins_count=5):
        """Aplica la transformación WOE si el IV es > 0.1."""
        if col not in self.df.columns or self.df[col].nunique(dropna=True) < 2:
            self._registrar_log(col, "woe_omitido", "warning", "Menos de 2 valores unicos")
            return None
        if not self.col_target_name or self.col_target_name not in self.df.columns:
            self._registrar_log(col, "woe_omitido", "warning", "No hay target disponible")
            return None

        try:
            serie = pd.to_numeric(self.df[col], errors='coerce')
            if serie.nunique(dropna=True) < 2:
                self._registrar_log(col, "woe_omitido", "warning", "Columna no numerica para binning")
                return None
            binned_col, bins_edges = pd.qcut(serie, q=bins_count, retbins=True, duplicates='drop')
        except Exception:
            try:
                binned_col, bins_edges = pd.cut(serie, bins=bins_count, retbins=True)
            except Exception as e:
                self._registrar_log(col, "woe", "error", e)
                return None

        try:
            df_temp = pd.DataFrame({col: binned_col, 'target': self.df[self.col_target_name]})
            woe_dict, iv_total = self.calcular_woe_iv(df_temp, col, 'target')

            if iv_total > 0.1:
                self.woe_mappings[col] = {
                    'bins_edges': bins_edges,
                    'woe_dict': woe_dict,
                    'iv': iv_total
                }
                self.df[col] = binned_col.map(woe_dict).astype(float)
                self._registrar_log(col, "woe", "success", f"IV {iv_total:.3f}")
                return f"WOE (IV: {iv_total:.3f})"
            else:
                self._registrar_log(col, "woe_omitido", "warning", f"IV {iv_total:.3f}")
                return f"WOE-Omitido (IV: {iv_total:.3f} <= 0.1)"
        except Exception as e:
            self._registrar_log(col, "woe", "error", e)
            return None

    def SePuedeCategorizar(self, columna, max_categorias=20, min_prop=0.05):
        if columna not in self.df.columns: return False
        
        freqs_norm = self.df[columna].value_counts(normalize=True)
        if freqs_norm.empty or freqs_norm.count() > max_categorias or freqs_norm.count() <= 1:
            return False
        
        categorias_invalidas = freqs_norm[freqs_norm <= min_prop].index.tolist()
        suma_otros = freqs_norm[freqs_norm <= min_prop].sum()
        if suma_otros > 0.10:
            return False
        
        num_categorias_finales = freqs_norm.count() - len(categorias_invalidas)
        if len(categorias_invalidas) > 0:
            num_categorias_finales += 1 
            
        columnas_extra = num_categorias_finales - 1 
        
        if (self.df.shape[1] + columnas_extra) > self.df.shape[0]:
            return False

        categorias_validas = freqs_norm[freqs_norm > min_prop].index.tolist()
        self.categorias_validas[columna] = categorias_validas

        # Cast to object first so string "otros" can be assigned regardless of the column's dtype
        # (e.g. Int64 nullable integers raised "Invalid value 'otros' for dtype 'Int64'")
        col_obj = self.df[columna].astype(object)
        self.df[columna] = col_obj.where(col_obj.isin(categorias_validas), "otros")
        return True

    def aplicar_target_encoding(self, columna):
        """Aplica Target Encoding (reemplaza categorías por la media del target)."""
        if columna not in self.df.columns or self.df[columna].nunique(dropna=True) < 2:
            self._registrar_log(columna, "target_encoding_omitido", "warning", "Menos de 2 valores unicos")
            return
        if not self.col_target_name or self.col_target_name not in self.df.columns:
            self._registrar_log(columna, "target_encoding_omitido", "warning", "No hay target disponible")
            return
        medias = self.df.groupby(columna)[self.col_target_name].mean()
        media_global = self.df[self.col_target_name].mean()
        self.target_encodings[columna] = {'medias': medias.to_dict(), 'global': media_global}
        self.df[columna] = self.df[columna].map(medias).fillna(media_global)
        self._registrar_log(columna, "target_encoding", "success")

    def aplicar_ordinal_encoding(self, columna, orden=None):
        """Aplica Ordinal Encoding a una columna basada en un orden lógico."""
        if columna not in self.df.columns or self.df[columna].nunique(dropna=True) < 2:
            self._registrar_log(columna, "ordinal_omitido", "warning", "Menos de 2 valores unicos")
            return "Ordinal Omitido"
        if orden is None:
            orden = sorted(self.df[columna].unique().tolist())
        
        mapping = {cat: i for i, cat in enumerate(orden)}
        self.ordinal_mappings[columna] = mapping
        
        self.df[columna] = self.df[columna].map(mapping).fillna(-1).astype(int)
        self._registrar_log(columna, "ordinal_encoding", "success")
        return "Ordinal Encoding"

    def Manejo_Fechas(self, columna):
        """Procesa columnas de fecha: ordena, extrae características y calcula t_paso."""
        if columna not in self.df.columns:
            return None

        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                try:
                    fecha_convertida = pd.to_datetime(self.df[columna], errors='coerce', format='mixed')
                except TypeError:
                    fecha_convertida = pd.to_datetime(self.df[columna], errors='coerce')
            validas = int(fecha_convertida.notna().sum())
            total_no_nulos = int(self.df[columna].notna().sum())

            if total_no_nulos == 0 or validas == 0:
                self._registrar_log(columna, "fecha_no_convertible", "warning")
                return None

            self.df[columna] = fecha_convertida
            min_date = self.df[columna].dropna().min()
            self.fecha_referencia[columna] = min_date

            self.df[f"{columna}_t_paso"] = (self.df[columna] - min_date).dt.days
            self.df[f"{columna}_anio"] = self.df[columna].dt.year
            self.df[f"{columna}_mes"] = self.df[columna].dt.month
            self.df[f"{columna}_dia"] = self.df[columna].dt.day

            self.df.drop(columns=[columna], inplace=True)
            self._registrar_log(columna, "fecha_transformada", "success", f"{validas} fechas validas")
            return "Procesamiento de Fecha (t_paso + features)"
        except Exception as e:
            self._registrar_log(columna, "fecha_no_convertible", "error", e)
            return None
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            try:
                self.df[columna] = pd.to_datetime(self.df[columna], errors='coerce', format='mixed')
            except TypeError:
                self.df[columna] = pd.to_datetime(self.df[columna], errors='coerce')

        if self.df[columna].isnull().any():
            self.df.dropna(subset=[columna], inplace=True)

        if self.df.empty:
            return None

        self.df.sort_values(by=columna, inplace=True)

        # Guardar fecha de referencia (Día 0)
        min_date = self.df[columna].min()
        self.fecha_referencia[columna] = min_date

        # Crear t_paso (días transcurridos)
        self.df[f"{columna}_t_paso"] = (self.df[columna] - min_date).dt.days

        # Extraer características
        self.df[f"{columna}_anio"] = self.df[columna].dt.year
        self.df[f"{columna}_mes"] = self.df[columna].dt.month
        
        # Solo extraer día si hay información real de días (si no todos son día 1)
        if self.df[columna].dt.day.nunique() > 1:
            self.df[f"{columna}_dia"] = self.df[columna].dt.day
            self.df[f"{columna}_dia_semana"] = self.df[columna].dt.dayofweek
            self.df[f"{columna}_es_fin_semana"] = self.df[columna].dt.dayofweek.isin([5, 6]).astype(int)
        
        # Eliminar la original
        self.df.drop(columns=[columna], inplace=True)
        return "Procesamiento de Fecha (t_paso + features)"

    def Clean_All_Rows(self, reglas_dict=None, EsPCA=False):
        # Guardar tipos de datos originales de entrenamiento para la inferencia
        self.dtypes_entrenamiento = {col: str(dtype) for col, dtype in self.df.dtypes.items()}
        
        print("Reglas de Entrada:")
        print(reglas_dict)
        self.df, logs_prevalidacion = pre_validar_columnas(self.df)
        self.logs_limpieza.extend(logs_prevalidacion)
        try:
            duplicados = int(self.df.duplicated().sum())
            self.df.drop_duplicates(inplace=True)
            self._registrar_log("__dataset__", "duplicados_eliminados", "success", f"{duplicados} filas")
        except Exception as e:
            self._registrar_log("__dataset__", "duplicados_eliminados", "error", e)
        reporte = []
        if reglas_dict is None: reglas_dict = {}
        
        # Normalizar claves para que la búsqueda sea insensible a mayúsculas y espacios
        reglas_normalizadas = {str(k).strip().lower(): v for k, v in reglas_dict.items()}
        self.reglas_entrenamiento = reglas_dict.copy()

        df_id = None
        if self.id_column and self.id_column in self.df.columns:
            df_id = self.df[[self.id_column]].copy()
            self.df.drop(columns=[self.id_column], inplace=True)

        columnas_actuales = [c for c in self.df.columns if c != self.col_target_name]

        # --- PREPARAR TARGET ANTES DEL LOOP (Necesario para Target Encoding / WOE) ---
        if self.col_target_name:
            self.df.dropna(subset=[self.col_target_name], inplace=True)
            if ptypes.is_object_dtype(self.df[self.col_target_name]) or ptypes.is_string_dtype(self.df[self.col_target_name]):
                from sklearn.preprocessing import LabelEncoder
                le = LabelEncoder()
                self.df[self.col_target_name] = le.fit_transform(self.df[self.col_target_name].astype(str))

        for col in columnas_actuales:
            if col not in self.df.columns:
                continue
            col_norm = str(col).strip().lower()
            regla = reglas_normalizadas.get(col_norm, {})
            
            es_fecha = ptypes.is_datetime64_any_dtype(self.df[col]) or \
                       (ptypes.is_object_dtype(self.df[col]) and "fecha" in col_norm) or \
                       regla.get('Fecha', False)

            if es_fecha:
                try:
                    res_fecha = self.Manejo_Fechas(col)
                    if res_fecha:
                        reporte.append({'columna': col, 'metodo': res_fecha, 'Valor_de_relleno': None})
                        continue
                except Exception as e:
                    self._registrar_log(col, "fecha_transformada", "error", e)

            if ptypes.is_numeric_dtype(self.df[col]):
                self.Manejo_Atipicos(col)

            res_nulos = self.No_nulos(col, 
                                      metodo=regla.get('metodo'), 
                                      toleranciaSustitucion=regla.get('tolSustitucion', 0.1), 
                                      toleranciaMantenerColumna=regla.get('tolMantenerCols', 0.5),
                                      valorPorDefecto=regla.get('valorDefecto'))
            reporte.append(res_nulos)

            if res_nulos['metodo'] == 'drop-column' or col not in self.df.columns:
                continue

            if regla.get('WOE', False):
                try:
                    res_woe = self.aplicar_woe_columna(col, bins_count=regla.get('bins_woe', 5))
                    if res_woe:
                        reporte[-1]['metodo'] += f" / {res_woe}"
                except Exception as e:
                    self._registrar_log(col, "woe", "error", e)
                    if col in self.df.columns and self.df[col].nunique(dropna=True) <= regla.get('MaxDummies', 20):
                        try:
                            self.df = pd.get_dummies(self.df, columns=[col], drop_first=True)
                            reporte[-1]['metodo'] += ' / Dummies fallback WOE'
                            self._registrar_log(col, "dummies_fallback_woe", "success")
                        except Exception as fallback_error:
                            self._registrar_log(col, "dummies_fallback_woe", "error", fallback_error)
        
            if col in self.woe_mappings:
                continue

            tiene_regla_categorica = regla.get('TargetEncoding', False) or regla.get('Ordinal', False) or regla.get('Dummies', False)
            es_texto_bool = ptypes.is_object_dtype(self.df[col]) or ptypes.is_string_dtype(self.df[col]) or ptypes.is_bool_dtype(self.df[col])

            if es_texto_bool or tiene_regla_categorica:
                if es_texto_bool:
                    self.Limpiar_solo_cadenas(col, lematizar=regla.get('Lematizar', False))
                
                if col not in self.df.columns:
                    continue
                
                es_dummificable = self.SePuedeCategorizar(col, max_categorias=regla.get('MaxDummies', 20))
                
                if regla.get('TargetEncoding', False):
                    try:
                        self.aplicar_target_encoding(col)
                        reporte[-1]['metodo'] += ' / Target Encoding'
                    except Exception as e:
                        self._registrar_log(col, "target_encoding", "error", e)
                elif regla.get('Ordinal', False):
                    try:
                        res_ord = self.aplicar_ordinal_encoding(col, orden=regla.get('orden'))
                        reporte[-1]['metodo'] += f' / {res_ord}'
                    except Exception as e:
                        self._registrar_log(col, "ordinal_encoding", "error", e)
                elif regla.get('Dummies', False):
                    try:
                        nunique = self.df[col].nunique(dropna=True)
                        max_dummies = regla.get('MaxDummies', 20)
                        if nunique < 2:
                            reporte[-1]['metodo'] += ' / Dummies omitido'
                            self._registrar_log(col, "dummies_omitido", "warning", "Menos de 2 valores unicos")
                        elif nunique > max_dummies:
                            reporte[-1]['metodo'] += ' / Dummies omitido (Alta cardinalidad)'
                            self._registrar_log(col, "dummies_omitido", "warning", f"{nunique} valores unicos")
                        else:
                            freq_max = self.df[col].value_counts(normalize=True).max() if not self.df[col].empty else 0
                            if freq_max < 0.05:
                                self.df.drop(columns=[col], inplace=True)
                                reporte[-1]['metodo'] += ' / Borrada (Frecuencia < 5%)'
                                self._registrar_log(col, "eliminacion_columna_frecuencia_baja", "success")
                            else:
                                self.df = pd.get_dummies(self.df, columns=[col], drop_first=True)
                                reporte[-1]['metodo'] += ' / Dummies'
                                self._registrar_log(col, "dummies", "success")
                    except Exception as e:
                        self._registrar_log(col, "dummies", "error", e)
                elif es_dummificable:
                    try:
                        self.df = pd.get_dummies(self.df, columns=[col], drop_first=True)
                        reporte[-1]['metodo'] += ' / Dummies'
                        self._registrar_log(col, "dummies", "success")
                    except Exception as e:
                        self._registrar_log(col, "dummies", "error", e)
                elif es_texto_bool:
                    self.df.drop(columns=[col], inplace=True)
                    reporte[-1]['metodo'] += ' / Borrada (No dummificable o Frecuencia < 5%)'
           
        self.df_limpio_usuario = self.df.copy()

        if self.col_target_name:
            self.y = self.df[self.col_target_name].copy()
            self.df.drop(columns=[self.col_target_name], inplace=True)
        else:
            self.y = None
        
        for col in self.df.columns:
            if ptypes.is_numeric_dtype(self.df[col]) or ptypes.is_bool_dtype(self.df[col]):
                try:
                    self.df[col] = pd.to_numeric(self.df[col], errors='coerce').fillna(0).astype(float)
                except Exception as e:
                    self._registrar_log(col, "conversion_numerica_final", "error", e)

        self.columnas_entrenamiento = self.df.columns.tolist()

        cols_num = self.df.select_dtypes(include=[np.number]).columns
        if not cols_num.empty:
            self.scaler = StandardScaler()
            self.df[cols_num] = self.scaler.fit_transform(self.df[cols_num])
            self.columnas_escaladas = cols_num.tolist()
            
        if EsPCA:
            self.df = self.df.select_dtypes(include=[np.number])
            self.columnas_entrenamiento = self.df.columns.tolist()
            
            if self.df.empty:
                raise ValueError("No quedan columnas numéricas para aplicar PCA.")

            pca_obj = PCA()
            pca_obj.fit(self.df)
            var_acc = np.cumsum(pca_obj.explained_variance_ratio_)
            n_comp = np.argmax(var_acc >= 0.7) + 1
            self.pca = PCA(n_components=n_comp)
            componentes = self.pca.fit_transform(self.df)
            self.df = pd.DataFrame(componentes, columns=[f'PC{i+1}' for i in range(n_comp)], index=self.df.index)
        
        print("Reglas Finales:")
        print(reporte)
        return reporte

    def transformar_nueva_tupla(self, nuevo_df):
        df_pred = nuevo_df.copy()
        
        # Alinear tipos de datos con los tipos originales de entrenamiento
        if hasattr(self, "dtypes_entrenamiento") and self.dtypes_entrenamiento:
            for col in df_pred.columns:
                if col in self.dtypes_entrenamiento:
                    dtype_str = self.dtypes_entrenamiento[col]
                    try:
                        if "int" in dtype_str.lower() or "float" in dtype_str.lower():
                            df_pred[col] = pd.to_numeric(df_pred[col], errors='coerce')
                        elif "datetime" in dtype_str.lower():
                            df_pred[col] = pd.to_datetime(df_pred[col], errors='coerce')
                        else:
                            df_pred[col] = df_pred[col].astype(dtype_str)
                    except Exception:
                        pass
        
        df_id_pred = None
        if self.id_column and self.id_column in df_pred.columns:
            df_id_pred = df_pred[[self.id_column]].copy()
            df_pred.drop(columns=[self.id_column], inplace=True)
        
        if self.col_target_name in df_pred.columns:
            df_pred.drop(columns=[self.col_target_name], inplace=True)

        for col in self.backup_imputaciones.keys():
            if col in df_pred.columns:
                val_relleno = self.imputaciones_nulos.get(col)
                if val_relleno is None:
                    val_relleno = self.backup_imputaciones.get(col)
                
                if val_relleno is not None:
                    df_pred[col] = df_pred[col].fillna(val_relleno)

        for col in list(df_pred.columns):
            regla = self.reglas_entrenamiento.get(col, {})
            
            if col in self.columnas_texto_separadas or regla.get('Lematizar', False):
                df_pred[col] = df_pred[col].apply(lambda x: unicodedata.normalize('NFKD', str(x)).encode('ASCII', 'ignore').decode('utf-8') if pd.notnull(x) else x)
                df_pred[col] = df_pred[col].str.lower()
                df_pred[col] = df_pred[col].apply(lambda x: re.sub(r'[^a-zA-Z0-9áéíóúÁÉÍÓÚñÑ\s]', ' ', str(x)) if pd.notnull(x) else x)
                
                if regla.get('Lematizar', False):
                    df_pred[col] = df_pred[col].apply(self.ExtraerTexto)
                    df_pred[col] = df_pred[col].apply(self.lematizar)
            

            if col in self.columnas_texto_separadas:
                df_dummies_texto = df_pred[col].str.get_dummies(sep=' ')
                cols_a_mantener = self.columnas_texto_separadas[col]
                for c_mantener in cols_a_mantener:
                    if c_mantener not in df_dummies_texto.columns:
                        df_dummies_texto[c_mantener] = 0
                df_filtrado = df_dummies_texto[cols_a_mantener]
                df_pred = pd.concat([df_pred.drop(columns=[col]), df_filtrado], axis=1)

        #Reemplazar por "otros" si son categorías nuevas o no pasaron el umbral en entrenamiento
        for col, categorias_validas in self.categorias_validas.items():
            if col in df_pred.columns:
                df_pred[col] = df_pred[col].where(df_pred[col].isin(categorias_validas), "otros")

        # Target Encoding
        for col, info in self.target_encodings.items():
            if col in df_pred.columns:
                df_pred[col] = df_pred[col].map(info['medias']).fillna(info['global'])

        # --- APLICAR WOE
        for col, mapping in self.woe_mappings.items():
            if col in df_pred.columns:
                binned = pd.cut(df_pred[col], bins=mapping['bins_edges'], include_lowest=True)
                df_pred[col] = binned.map(mapping['woe_dict']).astype(float)

        # --- APLICAR ORDINAL ENCODING 
        for col, mapping in self.ordinal_mappings.items():
            if col in df_pred.columns:
                df_pred[col] = df_pred[col].map(mapping).fillna(-1).astype(int)

        # --- APLICAR MANEJO DE FECHAS EN PREDICCIÓN
        for col, min_date in self.fecha_referencia.items():
            if col in df_pred.columns:
                df_pred[col] = pd.to_datetime(df_pred[col], errors='coerce')
                df_pred[col] = df_pred[col].fillna(min_date)
                df_pred[f"{col}_t_paso"] = (df_pred[col] - min_date).dt.days
                df_pred[f"{col}_anio"] = df_pred[col].dt.year
                df_pred[f"{col}_mes"] = df_pred[col].dt.month
                
                # Usamos la misma lógica: si la columna de entrenamiento tenía días, los extraemos
                if f"{col}_dia" in self.columnas_entrenamiento:
                    df_pred[f"{col}_dia"] = df_pred[col].dt.day
                    df_pred[f"{col}_dia_semana"] = df_pred[col].dt.dayofweek
                    df_pred[f"{col}_es_fin_semana"] = df_pred[col].dt.dayofweek.isin([5, 6]).astype(int)
                
                df_pred.drop(columns=[col], inplace=True)

        # 5. Dummies Normales
        df_pred = pd.get_dummies(df_pred)

        # Si una columna de entrenamiento no existe la crea con 0.
        # Si hay columnas extra en los datos nuevos, las borra.
        df_pred = df_pred.reindex(columns=self.columnas_entrenamiento, fill_value=0)

        # Convertir booleanos residuales a enteros
        columnas_bool = df_pred.select_dtypes(include=['bool']).columns
        if not columnas_bool.empty:
            df_pred[columnas_bool] = df_pred[columnas_bool].astype(int)

        # Garantizar que no quede NINGÚN NaN residual antes de escalar o aplicar PCA para evitar fallos en estimadores como MLPRegressor
        df_pred = df_pred.fillna(0)

        # Escalado
        if self.scaler is not None:
            cols_a_escalar = getattr(self, "columnas_escaladas", None)
            if cols_a_escalar is None and hasattr(self.scaler, "feature_names_in_"):
                cols_a_escalar = list(self.scaler.feature_names_in_)
            
            if cols_a_escalar:
                df_pred[cols_a_escalar] = self.scaler.transform(df_pred[cols_a_escalar])
            else:
                df_pred[self.columnas_entrenamiento] = self.scaler.transform(df_pred[self.columnas_entrenamiento])

        # PCA
        if self.pca is not None:
            componentes = self.pca.transform(df_pred)
            df_pred = pd.DataFrame(componentes, columns=[f'PC{i+1}' for i in range(componentes.shape[1])], index=df_pred.index)
        
        # Re-adjuntar ID si existía
        if df_id_pred is not None:
            df_pred = pd.concat([df_id_pred.reset_index(drop=True), df_pred.reset_index(drop=True)], axis=1)

        return df_pred
