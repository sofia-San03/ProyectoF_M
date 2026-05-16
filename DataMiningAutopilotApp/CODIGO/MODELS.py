import numpy as np
import json
import joblib
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (mean_squared_error, mean_absolute_error, mean_absolute_percentage_error, r2_score,
                             accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
                             silhouette_score, davies_bouldin_score, pairwise_distances, confusion_matrix,
                             ConfusionMatrixDisplay, RocCurveDisplay)
from sklearn.feature_selection import SelectKBest, f_regression, f_classif
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor, plot_tree
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.cluster import KMeans, DBSCAN
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage
import warnings
from uuid import uuid4
import os

from sklearn.compose import TransformedTargetRegressor

def Regresion_lineal(X, y, test_size=0.3, cv_folds=5, model_filename='modelo_lineal.pkl', seed=123, p_value_threshold=0.05):
    X = X.select_dtypes(include=[np.number]).copy()
    if X.empty:
        raise ValueError("Regresión lineal requiere columnas numéricas.")
 
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=seed)


    columnas_validas = X_train.columns[X_train.nunique() > 1]
    
    if len(columnas_validas) < len(X_train.columns):
        columnas_eliminadas = set(X_train.columns) - set(columnas_validas)   
    X_train = X_train[columnas_validas]
    X_test = X_test[columnas_validas]

    kb = SelectKBest(k="all", score_func=f_regression)
    kb.fit(X_train, y_train)
    
    mascara_significativas = kb.pvalues_ < p_value_threshold
    columnas_significativas = X_train.columns[mascara_significativas].tolist()
    
    # Fallback de seguridad
    if len(columnas_significativas) == 0:
        print(f"Advertencia: Ninguna variable cumplió el umbral p<{p_value_threshold}. Seleccionando las 3 mejores.")
        kb_fallback = SelectKBest(k=min(3, X.shape[1]), score_func=f_regression)
        kb_fallback.fit(X_train, y_train)
        columnas_significativas = X_train.columns[kb_fallback.get_support()].tolist()

    X_train_sel = X_train[columnas_significativas]
    X_test_sel = X_test[columnas_significativas]


    y_min = np.nanmin(y_train)
    if y_min > -1:
        modelo_base = TransformedTargetRegressor(
            regressor=LinearRegression(),
            func=np.log1p,
            inverse_func=np.expm1
        )
        parametros = {
            'regressor__fit_intercept': [True, False],
            'regressor__copy_X': [True, False]
        }
    else:
        modelo_base = LinearRegression()
        parametros = {
            'fit_intercept': [True, False],
            'copy_X': [True, False]
        }
    
    grid_search = GridSearchCV(estimator=modelo_base, param_grid=parametros, 
                               scoring='neg_mean_squared_error', cv=cv_folds, n_jobs=-1)

    grid_search.fit(X_train_sel, y_train)
    mejor_modelo = grid_search.best_estimator_


    # 4. Predicciones y Métricas de Precisión
    # Score en Entrenamiento (Train)
    score=mejor_modelo.score(X_train_sel, y_train)
    
    # Score en Prueba (Test)
    y_pred = mejor_modelo.predict(X_test_sel)

    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    mape = mean_absolute_percentage_error(y_test, y_pred)
    r2_test = r2_score(y_test, y_pred)

    cv_scores = cross_val_score(mejor_modelo, X_train_sel, y_train, cv=cv_folds, scoring='r2', n_jobs=-1)
    cv_r2_mean = cv_scores.mean()
    cv_r2_std = cv_scores.std()
    model_path = _ruta_con_id_unico(model_filename)

    resultados = {
        "seleccion_variables": {
            "cantidad_original": X.shape[1],
            "cantidad_final": len(columnas_significativas),
            "variables_utilizadas": columnas_significativas
        },
        "metricas_precision": {
            "modelo.score": float(score),
            "R2": float(r2_test),
            "MSE": float(mse),
            "RMSE": float(rmse),
            "MAE": float(mae),
            "MAPE": float(mape)
        },
        "cross_validation_train_R2": {
            "media": float(cv_r2_mean),
            "desviacion_estandar": float(cv_r2_std),
            "folds": cv_folds
        },
        "mejores_hiperparametros": grid_search.best_params_,
        "ruta_modelo_guardado": model_path
    }
    
    json_resultado = json.dumps(resultados, indent=4, ensure_ascii=False)

    paquete_modelo = {
        "modelo": mejor_modelo,
        "columnas": columnas_significativas
    }
    joblib.dump(paquete_modelo, model_path)

    return json_resultado, mejor_modelo, columnas_significativas

def _ruta_con_id_unico(filename):
    os.makedirs("MODELOS", exist_ok=True)
    base, ext = filename.rsplit('.', 1) if '.' in filename else (filename, '')
    sufijo = uuid4().hex[:8]
    return f"MODELOS/{base}_{sufijo}.{ext}" if ext else f"MODELOS/{base}_{sufijo}"

def _obtener_image_prefix(model_path):
    os.makedirs("Resultados", exist_ok=True)
    base_name = os.path.basename(model_path).replace(".pkl", "")
    return os.path.join("Resultados", base_name)

def _guardar_feature_importance(modelo, columnas, image_prefix):
    if not hasattr(modelo, "feature_importances_"):
        return None

    importancias = np.asarray(modelo.feature_importances_)
    orden = np.argsort(importancias)[::-1][:20]
    path = f"{image_prefix}_feature_importance.png"
    plt.figure(figsize=(10, 7))
    plt.barh(np.array(columnas)[orden][::-1], importancias[orden][::-1])
    plt.title("Feature importance")
    plt.xlabel("Importancia")
    plt.tight_layout()
    plt.savefig(path)
    plt.close()
    return path

def _guardar_coeficientes_logisticos(modelo, columnas, image_prefix):
    coef_path = f"{image_prefix}_coeficientes.png"
    coeficientes = np.asarray(modelo.coef_)
    if coeficientes.ndim == 2:
        valores = coeficientes[0]
    else:
        valores = coeficientes

    orden = np.argsort(np.abs(valores))[::-1][:20]
    plt.figure(figsize=(10, 7))
    plt.barh(np.array(columnas)[orden][::-1], valores[orden][::-1])
    plt.axvline(0, color="black", linewidth=0.8)
    plt.title("Coeficientes de regresión logística")
    plt.xlabel("Coeficiente")
    plt.tight_layout()
    plt.savefig(coef_path)
    plt.close()
    return coef_path

def _guardar_arbol_visual(modelo, columnas, image_prefix, class_names=None):
    path = f"{image_prefix}_tree.png"
    modelo_para_plot = modelo.estimators_[0] if hasattr(modelo, "estimators_") else modelo
    plt.figure(figsize=(22, 12))
    plot_tree(
        modelo_para_plot,
        feature_names=columnas,
        class_names=class_names,
        filled=True,
        rounded=True,
        max_depth=4
    )
    plt.title("Árbol de decisión" if not hasattr(modelo, "estimators_") else "Árbol representativo del RandomForest")
    plt.tight_layout()
    plt.savefig(path)
    plt.close()
    return path

def Arbol_decision(X, y, test_size=0.3, cv_folds=5, model_filename='modelo_arbol.pkl', seed=123, p_value_threshold=0.05):
    if y is None:
        raise ValueError("Arbol_decision requiere una variable objetivo.")

    X_numeric = X.select_dtypes(include=[np.number]).copy()
    columnas_validas = X_numeric.columns[X_numeric.nunique() > 1]
    X_numeric = X_numeric[columnas_validas]
    if X_numeric.empty:
        raise ValueError("Arbol_decision requiere columnas numéricas no constantes.")

    y_array = np.asarray(y)
    es_clf = _es_clasificacion(y_array)
    unique_values = np.sort(np.unique(y_array))

    if es_clf:
        _, class_counts = np.unique(y_array, return_counts=True)
        if np.min(class_counts) < 2:
            raise ValueError("Clasificación con árboles requiere al menos 2 registros por clase.")
        stratify = y_array
        score_func = f_classif
        scoring = "accuracy"
    else:
        stratify = None
        score_func = f_regression
        scoring = "r2"

    X_train, X_test, y_train, y_test = train_test_split(
        X_numeric,
        y_array,
        test_size=test_size,
        random_state=seed,
        stratify=stratify
    )

    kb = SelectKBest(k="all", score_func=score_func)
    kb.fit(X_train, y_train)
    pvalues = np.nan_to_num(kb.pvalues_, nan=1.0)
    mascara_significativas = pvalues < p_value_threshold
    columnas_significativas = X_train.columns[mascara_significativas].tolist()

    if len(columnas_significativas) == 0:
        k_fallback = min(5, X_train.shape[1])
        kb_fallback = SelectKBest(k=k_fallback, score_func=score_func)
        kb_fallback.fit(X_train, y_train)
        columnas_significativas = X_train.columns[kb_fallback.get_support()].tolist()

    X_train_sel = X_train[columnas_significativas]
    X_test_sel = X_test[columnas_significativas]
    cv_seguro = _cv_seguro(y_train, es_clf, cv_folds)

    if es_clf:
        candidatos = [
            (
                "DecisionTreeClassifier",
                DecisionTreeClassifier(random_state=seed),
                {
                    "criterion": ["gini", "entropy"],
                    "max_depth": [None, 5, 10],
                    "min_samples_split": [2, 5],
                    "min_samples_leaf": [1, 2]
                }
            ),
            (
                "RandomForestClassifier",
                RandomForestClassifier(random_state=seed, n_jobs=-1),
                {
                    "n_estimators": [100],
                    "criterion": ["gini", "entropy"],
                    "max_depth": [None, 5, 10],
                    "min_samples_leaf": [1, 2]
                }
            )
        ]
    else:
        candidatos = [
            (
                "DecisionTreeRegressor",
                DecisionTreeRegressor(random_state=seed),
                {
                    "criterion": ["squared_error", "absolute_error"],
                    "max_depth": [None, 5, 10],
                    "min_samples_split": [2, 5],
                    "min_samples_leaf": [1, 2]
                }
            ),
            (
                "RandomForestRegressor",
                RandomForestRegressor(random_state=seed, n_jobs=-1),
                {
                    "n_estimators": [100],
                    "criterion": ["squared_error"],
                    "max_depth": [None, 5, 10],
                    "min_samples_leaf": [1, 2]
                }
            )
        ]

    busquedas = []
    for nombre, estimador, parametros in candidatos:
        grid = GridSearchCV(
            estimator=estimador,
            param_grid=parametros,
            scoring=scoring,
            cv=cv_seguro,
            n_jobs=-1
        )
        grid.fit(X_train_sel, y_train)
        busquedas.append((nombre, grid))

    nombre_mejor, grid_search = sorted(busquedas, key=lambda item: item[1].best_score_, reverse=True)[0]
    mejor_modelo = grid_search.best_estimator_
    y_pred = mejor_modelo.predict(X_test_sel)

    model_path = _ruta_con_id_unico(model_filename)
    image_prefix = _obtener_image_prefix(model_path)
    class_names = [str(v) for v in unique_values] if es_clf else None
    tree_path = _guardar_arbol_visual(mejor_modelo, columnas_significativas, image_prefix, class_names=class_names)
    importance_path = _guardar_feature_importance(mejor_modelo, columnas_significativas, image_prefix)

    visualizaciones = {
        "arbol": tree_path,
        "feature_importance": importance_path,
        "matriz_confusion": None
    }

    if es_clf:
        average_mode = "binary" if len(unique_values) == 2 else "weighted"
        acc = accuracy_score(y_test, y_pred)
        if len(unique_values) == 2:
            pos_label = unique_values[-1]
            prec = precision_score(y_test, y_pred, pos_label=pos_label, zero_division=0)
            rec = recall_score(y_test, y_pred, pos_label=pos_label, zero_division=0)
            f1 = f1_score(y_test, y_pred, pos_label=pos_label, zero_division=0)
        else:
            prec = precision_score(y_test, y_pred, average=average_mode, zero_division=0)
            rec = recall_score(y_test, y_pred, average=average_mode, zero_division=0)
            f1 = f1_score(y_test, y_pred, average=average_mode, zero_division=0)

        y_prob = mejor_modelo.predict_proba(X_test_sel)
        if len(unique_values) == 2:
            auc = roc_auc_score(y_test, y_prob[:, 1])
        else:
            auc = roc_auc_score(y_test, y_prob, multi_class="ovr", average="weighted")

        confusion_path, matriz = _guardar_matriz_confusion(y_test, y_pred, unique_values, image_prefix)
        visualizaciones["matriz_confusion"] = confusion_path

        resultados = {
            "tipo_modelo": "Arbol_decision",
            "modelo_seleccionado": nombre_mejor,
            "tipo_problema": "clasificacion",
            "metricas_precision": {
                "Accuracy": float(acc),
                "Precision": float(prec),
                "Recall": float(rec),
                "F1-Score": float(f1),
                "ROC-AUC": float(auc)
            },
            "matriz_confusion": matriz.tolist(),
            "visualizaciones": visualizaciones
        }
    else:
        r2 = r2_score(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae = mean_absolute_error(y_test, y_pred)
        resultados = {
            "tipo_modelo": "Arbol_decision",
            "modelo_seleccionado": nombre_mejor,
            "tipo_problema": "regresion",
            "metricas_precision": {
                "R2": float(r2),
                "RMSE": float(rmse),
                "MAE": float(mae)
            },
            "visualizaciones": visualizaciones
        }

    resultados.update({
        "seleccion_variables": {
            "cantidad_original": X.shape[1],
            "cantidad_final": len(columnas_significativas),
            "variables_utilizadas": columnas_significativas
        },
        "validacion": {
            "test_size": test_size,
            "cv_folds": cv_seguro,
            "mejor_score_grid": float(grid_search.best_score_),
            "comparacion_modelos": [
                {
                    "modelo": nombre,
                    "mejor_score": float(grid.best_score_),
                    "mejores_hiperparametros": grid.best_params_
                }
                for nombre, grid in busquedas
            ]
        },
        "mejores_hiperparametros": grid_search.best_params_,
        "ruta_modelo_guardado": model_path
    })

    paquete_modelo = {
        "modelo": mejor_modelo,
        "columnas": columnas_significativas,
        "resultados": resultados
    }
    joblib.dump(paquete_modelo, model_path)

    return json.dumps(resultados, indent=4, ensure_ascii=False), mejor_modelo, columnas_significativas

def Regresion_logistica(X, y, test_size=0.3, cv_folds=5, model_filename='modelo_logistico.pkl', seed=123, p_value_threshold=0.05):
    if y is None:
        raise ValueError("Regresion_logistica requiere una variable objetivo.")

    y_array = np.asarray(y)
    unique_values = np.sort(np.unique(y_array))
    if len(unique_values) != 2:
        raise ValueError(f"La Regresión Logística requiere una variable objetivo binaria (2 valores). Se encontraron {len(unique_values)} valores: {unique_values}")

    _, class_counts = np.unique(y_array, return_counts=True)
    if np.min(class_counts) < 2:
        raise ValueError("Regresión logística requiere al menos 2 registros por clase.")
    
    X_numeric = X.select_dtypes(include=[np.number]).copy()
    columnas_validas = X_numeric.columns[X_numeric.nunique() > 1]
    X_numeric = X_numeric[columnas_validas]
    if X_numeric.empty:
        raise ValueError("Regresión logística requiere columnas numéricas no constantes.")

    X_train, X_test, y_train, y_test = train_test_split(
        X_numeric,
        y_array,
        test_size=test_size,
        random_state=seed,
        stratify=y_array
    )

    kb = SelectKBest(k="all", score_func=f_classif)
    kb.fit(X_train, y_train)
    
    pvalues = np.nan_to_num(kb.pvalues_, nan=1.0)
    mascara_significativas = pvalues < p_value_threshold
    columnas_significativas = X_train.columns[mascara_significativas].tolist()
    
    if len(columnas_significativas) == 0:
        print(f"Advertencia: Ninguna variable cumplió el umbral p<{p_value_threshold}. Seleccionando las 3 mejores.")
        kb_fallback = SelectKBest(k=min(5, X_train.shape[1]), score_func=f_classif)
        kb_fallback.fit(X_train, y_train)
        columnas_significativas = X_train.columns[kb_fallback.get_support()].tolist()

    X_train_sel = X_train[columnas_significativas]
    X_test_sel = X_test[columnas_significativas]

    cv_seguro = _cv_seguro(y_train, True, cv_folds)
    parametros = {
        'C': [0.01, 0.1, 1, 10, 100],
        'penalty': ['l1', 'l2'],
        'solver': ['liblinear', 'saga'],
        'class_weight': [None, 'balanced'],
        'max_iter': [500, 1000]
    }
    
    modelo_base = LogisticRegression(random_state=seed)
    grid_search = GridSearchCV(
        estimator=modelo_base,
        param_grid=parametros,
        scoring='roc_auc',
        cv=cv_seguro,
        n_jobs=-1
    )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        grid_search.fit(X_train_sel, y_train)
    mejor_modelo = grid_search.best_estimator_

    if not hasattr(mejor_modelo, "predict_proba"):
        raise ValueError("El modelo de regresión logística seleccionado no expone probabilidades.")

    y_pred = mejor_modelo.predict(X_test_sel)
    probas = mejor_modelo.predict_proba(X_test_sel)
    if probas.shape[1] != 2:
        raise ValueError("La validación de probabilidades esperaba exactamente 2 columnas para clasificación binaria.")
    y_prob = probas[:, 1]

    acc = accuracy_score(y_test, y_pred)
    pos_label = unique_values[-1]
    prec = precision_score(y_test, y_pred, pos_label=pos_label, zero_division=0)
    rec = recall_score(y_test, y_pred, pos_label=pos_label, zero_division=0)
    f1 = f1_score(y_test, y_pred, pos_label=pos_label, zero_division=0)
    auc = roc_auc_score(y_test, y_prob)

    model_path = _ruta_con_id_unico(model_filename)
    image_prefix = _obtener_image_prefix(model_path)
    roc_path = _guardar_roc(y_test, y_prob, image_prefix)
    confusion_path, matriz = _guardar_matriz_confusion(y_test, y_pred, unique_values, image_prefix)
    coeficientes_path = _guardar_coeficientes_logisticos(mejor_modelo, columnas_significativas, image_prefix)

    resultados = {
        "tipo_modelo": "Regresion_logistica",
        "tipo_problema": "clasificacion_binaria",
        "seleccion_variables": {
            "cantidad_original": X.shape[1],
            "cantidad_final": len(columnas_significativas),
            "variables_utilizadas": columnas_significativas
        },
        "metricas_precision": {
            "Accuracy": float(acc),
            "Precision": float(prec),
            "Recall": float(rec),
            "F1-Score": float(f1),
            "ROC-AUC": float(auc)
        },
        "matriz_confusion": matriz.tolist(),
        "visualizaciones": {
            "curva_roc": roc_path,
            "matriz_confusion": confusion_path,
            "coeficientes": coeficientes_path
        },
        "validacion": {
            "test_size": test_size,
            "cv_folds": cv_seguro,
            "mejor_score_grid": float(grid_search.best_score_),
            "class_distribution": {
                str(label): int(count)
                for label, count in zip(unique_values, class_counts)
            }
        },
        "mejores_hiperparametros": grid_search.best_params_,
        "ruta_modelo_guardado": model_path
    }
    
    json_resultado = json.dumps(resultados, indent=4, ensure_ascii=False)

    paquete_modelo = {
        "modelo": mejor_modelo,
        "columnas": columnas_significativas
    }
    joblib.dump(paquete_modelo, model_path)

    return json_resultado, mejor_modelo, columnas_significativas

class OptimizedClusterModel:
    def __init__(self, algoritmo, estimator=None, centers=None):
        self.algoritmo = algoritmo
        self.estimator = estimator
        self.centers = centers

    def predict(self, X):
        if self.estimator is not None and hasattr(self.estimator, "predict"):
            return self.estimator.predict(X)

        if self.centers is None:
            raise ValueError("Este modelo de clustering no tiene centros disponibles para predicción.")

        distances = pairwise_distances(X, self.centers, metric="euclidean")
        return np.argmin(distances, axis=1)

def _validar_labels_clustering(labels):
    labels_validos = labels[labels != -1]
    if len(labels_validos) == 0:
        return False
    return len(np.unique(labels_validos)) >= 2

def _calcular_metricas_clustering(X, labels, centers=None):
    labels = np.asarray(labels)
    mascara = labels != -1
    X_eval = X[mascara]
    labels_eval = labels[mascara]

    metricas = {
        "Silhouette Score": None,
        "Davies-Bouldin Index": None,
        "Inercia": None,
        "Distancia Euclidiana Media": None,
        "Distancia Manhattan Media": None
    }

    if len(X_eval) == 0 or len(np.unique(labels_eval)) < 2:
        return metricas

    metricas["Silhouette Score"] = float(silhouette_score(X_eval, labels_eval))
    metricas["Davies-Bouldin Index"] = float(davies_bouldin_score(X_eval, labels_eval))

    if centers is None:
        centers = np.array([X_eval[labels_eval == label].mean(axis=0) for label in np.unique(labels_eval)])

    label_to_index = {label: idx for idx, label in enumerate(np.unique(labels_eval))}
    assigned_centers = np.array([centers[label_to_index[label]] for label in labels_eval])
    euclidean_distances = np.linalg.norm(X_eval - assigned_centers, axis=1)
    manhattan_distances = np.abs(X_eval - assigned_centers).sum(axis=1)

    metricas["Inercia"] = float(np.sum(euclidean_distances ** 2))
    metricas["Distancia Euclidiana Media"] = float(np.mean(euclidean_distances))
    metricas["Distancia Manhattan Media"] = float(np.mean(manhattan_distances))
    return metricas

def _k_medians(X, n_clusters, seed=123, max_iter=100):
    rng = np.random.default_rng(seed)
    initial_idx = rng.choice(len(X), size=n_clusters, replace=False)
    centers = X[initial_idx].copy()
    labels = np.zeros(len(X), dtype=int)

    for _ in range(max_iter):
        distances = pairwise_distances(X, centers, metric="manhattan")
        new_labels = np.argmin(distances, axis=1)
        new_centers = centers.copy()

        for cluster_id in range(n_clusters):
            cluster_points = X[new_labels == cluster_id]
            if len(cluster_points) > 0:
                new_centers[cluster_id] = np.median(cluster_points, axis=0)

        if np.array_equal(new_labels, labels) and np.allclose(new_centers, centers):
            break
        labels = new_labels
        centers = new_centers

    return labels, centers

def _k_medoids(X, n_clusters, seed=123, max_iter=60):
    rng = np.random.default_rng(seed)
    medoid_indices = rng.choice(len(X), size=n_clusters, replace=False)
    labels = np.zeros(len(X), dtype=int)

    for _ in range(max_iter):
        medoids = X[medoid_indices]
        distances = pairwise_distances(X, medoids, metric="euclidean")
        new_labels = np.argmin(distances, axis=1)
        new_medoid_indices = medoid_indices.copy()

        for cluster_id in range(n_clusters):
            cluster_indices = np.where(new_labels == cluster_id)[0]
            if len(cluster_indices) == 0:
                continue
            cluster_distances = pairwise_distances(X[cluster_indices], X[cluster_indices], metric="euclidean")
            new_medoid_indices[cluster_id] = cluster_indices[np.argmin(cluster_distances.sum(axis=1))]

        if np.array_equal(new_labels, labels) and np.array_equal(new_medoid_indices, medoid_indices):
            break
        labels = new_labels
        medoid_indices = new_medoid_indices

    return labels, X[medoid_indices]

def _guardar_visualizaciones_clustering(X, labels, inercias_codo, k_values, image_prefix):
    scatter_path = f"{image_prefix}_scatter.png"
    elbow_path = f"{image_prefix}_elbow.png"
    dendrogram_path = f"{image_prefix}_dendrograma.png"

    if X.shape[1] > 2:
        puntos_2d = PCA(n_components=2, random_state=123).fit_transform(X)
        x_label, y_label = "PC1", "PC2"
    else:
        puntos_2d = X
        x_label, y_label = "Feature 1", "Feature 2"

    plt.figure(figsize=(10, 7))
    plt.scatter(puntos_2d[:, 0], puntos_2d[:, 1], c=labels, cmap="tab10", s=35, alpha=0.85)
    plt.title("Scatter de clusters")
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.tight_layout()
    plt.savefig(scatter_path)
    plt.close()

    plt.figure(figsize=(10, 6))
    plt.plot(k_values, inercias_codo, marker="o")
    plt.title("Método del codo")
    plt.xlabel("Número de clusters")
    plt.ylabel("Inercia")
    plt.tight_layout()
    plt.savefig(elbow_path)
    plt.close()

    sample_size = min(200, len(X))
    X_dendro = X[:sample_size]
    plt.figure(figsize=(12, 7))
    dendrogram(linkage(X_dendro, method="ward"))
    plt.title("Dendrograma jerárquico")
    plt.xlabel("Observaciones")
    plt.ylabel("Distancia")
    plt.tight_layout()
    plt.savefig(dendrogram_path)
    plt.close()

    return scatter_path, elbow_path, dendrogram_path

def Clustering_optimizacion(X, y=None, min_clusters=2, max_clusters=10, model_filename='modelo_clustering.pkl', seed=123):
    if X is None or X.empty:
        raise ValueError("Clustering requiere al menos una matriz X con columnas numéricas.")

    X_numeric = X.select_dtypes(include=[np.number]).copy()
    if X_numeric.empty:
        raise ValueError("Clustering requiere columnas numéricas después del preprocesamiento.")

    X_values = X_numeric.to_numpy(dtype=float)
    n_samples = len(X_values)
    if n_samples < 3:
        raise ValueError("Clustering requiere al menos 3 registros.")

    max_k = min(max_clusters, n_samples - 1)
    min_k = min(min_clusters, max_k)
    k_values = list(range(min_k, max_k + 1))

    inercias_codo = []
    candidatos = []

    for k in k_values:
        kmeans = KMeans(n_clusters=k, random_state=seed, n_init=10)
        labels_kmeans = kmeans.fit_predict(X_values)
        inercias_codo.append(float(kmeans.inertia_))
        metricas_kmeans = _calcular_metricas_clustering(X_values, labels_kmeans, kmeans.cluster_centers_)
        candidatos.append({
            "algoritmo": "K-medias",
            "k": k,
            "labels": labels_kmeans,
            "centers": kmeans.cluster_centers_,
            "estimator": kmeans,
            "metricas": metricas_kmeans
        })

        labels_medoids, centers_medoids = _k_medoids(X_values, k, seed=seed)
        candidatos.append({
            "algoritmo": "K-medioides",
            "k": k,
            "labels": labels_medoids,
            "centers": centers_medoids,
            "estimator": None,
            "metricas": _calcular_metricas_clustering(X_values, labels_medoids, centers_medoids)
        })

        labels_medians, centers_medians = _k_medians(X_values, k, seed=seed)
        candidatos.append({
            "algoritmo": "K-medianas",
            "k": k,
            "labels": labels_medians,
            "centers": centers_medians,
            "estimator": None,
            "metricas": _calcular_metricas_clustering(X_values, labels_medians, centers_medians)
        })

    if n_samples > 4:
        nearest_distances = np.sort(pairwise_distances(X_values, metric="euclidean"), axis=1)[:, 1]
        eps_values = np.unique(np.percentile(nearest_distances, [50, 65, 80, 90]))
        min_samples_values = sorted(set([3, 5, max(3, min(10, int(np.sqrt(n_samples))))]))

        for eps in eps_values:
            if eps <= 0:
                continue
            for min_samples_dbscan in min_samples_values:
                dbscan = DBSCAN(eps=float(eps), min_samples=min_samples_dbscan)
                labels_dbscan = dbscan.fit_predict(X_values)
                if not _validar_labels_clustering(labels_dbscan):
                    continue
                metricas_dbscan = _calcular_metricas_clustering(X_values, labels_dbscan)
                labels_validos = labels_dbscan[labels_dbscan != -1]
                centers_dbscan = np.array([
                    X_values[labels_dbscan == label].mean(axis=0)
                    for label in np.unique(labels_validos)
                ])
                candidatos.append({
                    "algoritmo": "DBSCAN",
                    "k": int(len(np.unique(labels_validos))),
                    "labels": labels_dbscan,
                    "centers": centers_dbscan,
                    "estimator": None,
                    "metricas": metricas_dbscan,
                    "parametros": {
                        "eps": float(eps),
                        "min_samples": int(min_samples_dbscan)
                    }
                })

    candidatos_validos = [
        c for c in candidatos
        if c["metricas"]["Silhouette Score"] is not None
    ]
    if not candidatos_validos:
        raise ValueError("No se pudo encontrar una solución de clustering válida con al menos 2 clusters.")

    mejor = sorted(
        candidatos_validos,
        key=lambda c: (
            -c["metricas"]["Silhouette Score"],
            c["metricas"]["Davies-Bouldin Index"]
        )
    )[0]

    model_path = _ruta_con_id_unico(model_filename)
    image_prefix = _obtener_image_prefix(model_path)
    scatter_path, elbow_path, dendrogram_path = _guardar_visualizaciones_clustering(
        X_values,
        mejor["labels"],
        inercias_codo,
        k_values,
        image_prefix
    )

    metricas_por_algoritmo = []
    for candidato in candidatos_validos:
        item = {
            "algoritmo": candidato["algoritmo"],
            "clusters": int(candidato["k"]),
            "metricas": candidato["metricas"]
        }
        if "parametros" in candidato:
            item["parametros"] = candidato["parametros"]
        metricas_por_algoritmo.append(item)

    modelo_cluster = OptimizedClusterModel(
        algoritmo=mejor["algoritmo"],
        estimator=mejor["estimator"],
        centers=mejor["centers"]
    )

    resultados = {
        "tipo_modelo": "Clustering_optimizacion",
        "modelo_seleccionado": mejor["algoritmo"],
        "mejor_numero_clusters": int(mejor["k"]),
        "metricas_precision": mejor["metricas"],
        "metricas_por_algoritmo": metricas_por_algoritmo,
        "metodo_codo": {
            "k_values": [int(k) for k in k_values],
            "inercias": inercias_codo
        },
        "visualizaciones": {
            "scatter_clusters": scatter_path,
            "elbow_chart": elbow_path,
            "dendrograma": dendrogram_path
        },
        "seleccion_variables": {
            "cantidad_original": X.shape[1],
            "cantidad_final": X_numeric.shape[1],
            "variables_utilizadas": X_numeric.columns.tolist()
        },
        "ruta_modelo_guardado": model_path
    }

    paquete_modelo = {
        "modelo": modelo_cluster,
        "columnas": X_numeric.columns.tolist(),
        "labels": mejor["labels"],
        "centros": mejor["centers"],
        "resultados": resultados
    }
    joblib.dump(paquete_modelo, model_path)

    return json.dumps(resultados, indent=4, ensure_ascii=False), modelo_cluster, X_numeric.columns.tolist()

def _es_clasificacion(y):
    unique_values = np.unique(y)
    if len(unique_values) <= 1:
        raise ValueError("La variable objetivo debe tener al menos 2 valores distintos.")

    valores_enteros = np.allclose(unique_values, np.round(unique_values))
    return valores_enteros and len(unique_values) <= min(20, max(2, int(len(y) * 0.2)))

def _cv_seguro(y_train, es_clasificacion, cv_folds):
    if es_clasificacion:
        _, counts = np.unique(y_train, return_counts=True)
        folds = min(cv_folds, int(counts.min()))
        if folds < 2:
            raise ValueError("Redes neuronales requiere al menos 2 registros por clase en entrenamiento para validar.")
        return folds

    folds = min(cv_folds, len(y_train) // 2)
    if folds < 2:
        raise ValueError("Redes neuronales requiere más registros para validación cruzada.")
    return folds

def _guardar_curva_perdida(modelo, image_prefix):
    loss_path = f"{image_prefix}_loss.png"
    plt.figure(figsize=(9, 6))
    if hasattr(modelo, "loss_curve_") and modelo.loss_curve_:
        plt.plot(modelo.loss_curve_)
    plt.title("Curva de pérdida")
    plt.xlabel("Iteración")
    plt.ylabel("Loss")
    plt.tight_layout()
    plt.savefig(loss_path)
    plt.close()
    return loss_path

def _guardar_roc(y_test, y_prob, image_prefix):
    roc_path = f"{image_prefix}_roc.png"
    plt.figure(figsize=(8, 6))
    RocCurveDisplay.from_predictions(y_test, y_prob)
    plt.title("Curva ROC")
    plt.tight_layout()
    plt.savefig(roc_path)
    plt.close()
    return roc_path

def _guardar_matriz_confusion(y_test, y_pred, labels, image_prefix):
    confusion_path = f"{image_prefix}_confusion.png"
    matriz = confusion_matrix(y_test, y_pred, labels=labels)
    disp = ConfusionMatrixDisplay(confusion_matrix=matriz, display_labels=labels)
    disp.plot(cmap="Blues", values_format="d")
    plt.title("Matriz de confusión")
    plt.tight_layout()
    plt.savefig(confusion_path)
    plt.close()
    return confusion_path, matriz

def _guardar_real_vs_pred(y_test, y_pred, image_prefix):
    real_vs_pred_path = f"{image_prefix}_real_vs_pred.png"
    plt.figure(figsize=(8, 6))
    plt.scatter(y_test, y_pred, alpha=0.75)
    min_val = min(np.min(y_test), np.min(y_pred))
    max_val = max(np.max(y_test), np.max(y_pred))
    plt.plot([min_val, max_val], [min_val, max_val], color="red", linestyle="--")
    plt.title("Comparación real vs predicción")
    plt.xlabel("Valor real")
    plt.ylabel("Predicción")
    plt.tight_layout()
    plt.savefig(real_vs_pred_path)
    plt.close()
    return real_vs_pred_path

def Redes_neuronales(X, y, test_size=0.3, cv_folds=5, model_filename='modelo_redes_neuronales.pkl', seed=123):
    if y is None:
        raise ValueError("Redes_neuronales requiere una variable objetivo.")

    X_numeric = X.select_dtypes(include=[np.number]).copy()
    columnas_validas = X_numeric.columns[X_numeric.nunique() > 1]
    X_numeric = X_numeric[columnas_validas]

    if X_numeric.empty:
        raise ValueError("Redes_neuronales requiere columnas numéricas no constantes.")

    y_array = np.asarray(y)
    es_clf = _es_clasificacion(y_array)
    model_path = _ruta_con_id_unico(model_filename)
    image_prefix = _obtener_image_prefix(model_path)

    if es_clf:
        _, class_counts = np.unique(y_array, return_counts=True)
        if np.min(class_counts) < 2:
            raise ValueError("Clasificación con redes neuronales requiere al menos 2 registros por clase.")

    stratify = y_array if es_clf and np.min(np.unique(y_array, return_counts=True)[1]) >= 2 else None
    X_train, X_test, y_train, y_test = train_test_split(
        X_numeric,
        y_array,
        test_size=test_size,
        random_state=seed,
        stratify=stratify
    )

    cv_seguro = _cv_seguro(y_train, es_clf, cv_folds)

    if es_clf:
        modelo_base = MLPClassifier(random_state=seed, early_stopping=True, max_iter=500)
        parametros = {
            "hidden_layer_sizes": [(32,), (64,), (32, 16)],
            "activation": ["relu", "tanh"],
            "alpha": [0.0001, 0.001],
            "learning_rate_init": [0.001, 0.01]
        }
        scoring = "accuracy"
    else:
        modelo_base = MLPRegressor(random_state=seed, early_stopping=True, max_iter=700)
        parametros = {
            "hidden_layer_sizes": [(32,), (64,), (32, 16)],
            "activation": ["relu", "tanh"],
            "alpha": [0.0001, 0.001],
            "learning_rate_init": [0.001, 0.01]
        }
        scoring = "neg_root_mean_squared_error"

    grid_search = GridSearchCV(
        estimator=modelo_base,
        param_grid=parametros,
        scoring=scoring,
        cv=cv_seguro,
        n_jobs=-1
    )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        grid_search.fit(X_train, y_train)

    mejor_modelo = grid_search.best_estimator_
    y_pred = mejor_modelo.predict(X_test)
    loss_path = _guardar_curva_perdida(mejor_modelo, image_prefix)

    if es_clf:
        unique_values = np.sort(np.unique(y_array))
        average_mode = "binary" if len(unique_values) == 2 else "weighted"

        acc = accuracy_score(y_test, y_pred)
        if len(unique_values) == 2:
            pos_label = unique_values[-1]
            prec = precision_score(y_test, y_pred, pos_label=pos_label, zero_division=0)
            rec = recall_score(y_test, y_pred, pos_label=pos_label, zero_division=0)
            f1 = f1_score(y_test, y_pred, pos_label=pos_label, zero_division=0)
        else:
            prec = precision_score(y_test, y_pred, average=average_mode, zero_division=0)
            rec = recall_score(y_test, y_pred, average=average_mode, zero_division=0)
            f1 = f1_score(y_test, y_pred, average=average_mode, zero_division=0)

        y_prob = mejor_modelo.predict_proba(X_test)
        if len(unique_values) == 2:
            auc = roc_auc_score(y_test, y_prob[:, 1])
            roc_path = _guardar_roc(y_test, y_prob[:, 1], image_prefix)
        else:
            auc = roc_auc_score(y_test, y_prob, multi_class="ovr", average="weighted")
            roc_path = None

        confusion_path, matriz = _guardar_matriz_confusion(y_test, y_pred, unique_values, image_prefix)

        resultados = {
            "tipo_modelo": "Redes_neuronales",
            "tipo_problema": "clasificacion",
            "metricas_precision": {
                "Accuracy": float(acc),
                "Precision": float(prec),
                "Recall": float(rec),
                "F1-Score": float(f1),
                "ROC-AUC": float(auc)
            },
            "matriz_confusion": matriz.tolist(),
            "visualizaciones": {
                "curva_roc": roc_path,
                "matriz_confusion": confusion_path,
                "perdida": loss_path,
                "real_vs_prediccion": None
            },
            "validacion": {
                "test_size": test_size,
                "cv_folds": cv_seguro,
                "mejor_score_grid": float(grid_search.best_score_)
            },
            "seleccion_variables": {
                "cantidad_original": X.shape[1],
                "cantidad_final": len(columnas_validas),
                "variables_utilizadas": columnas_validas.tolist()
            },
            "mejores_hiperparametros": grid_search.best_params_,
        "ruta_modelo_guardado": model_path
        }
    else:
        y_pred = np.asarray(y_pred)
        r2 = r2_score(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae = mean_absolute_error(y_test, y_pred)
        real_vs_pred_path = _guardar_real_vs_pred(y_test, y_pred, image_prefix)

        resultados = {
            "tipo_modelo": "Redes_neuronales",
            "tipo_problema": "regresion",
            "metricas_precision": {
                "R2": float(r2),
                "RMSE": float(rmse),
                "MAE": float(mae)
            },
            "visualizaciones": {
                "curva_roc": None,
                "matriz_confusion": None,
                "perdida": loss_path,
                "real_vs_prediccion": real_vs_pred_path
            },
            "validacion": {
                "test_size": test_size,
                "cv_folds": cv_seguro,
                "mejor_score_grid": float(grid_search.best_score_)
            },
            "seleccion_variables": {
                "cantidad_original": X.shape[1],
                "cantidad_final": len(columnas_validas),
                "variables_utilizadas": columnas_validas.tolist()
            },
            "mejores_hiperparametros": grid_search.best_params_,
            "ruta_modelo_guardado": model_path
        }

    paquete_modelo = {
        "modelo": mejor_modelo,
        "columnas": columnas_validas.tolist(),
        "resultados": resultados
    }
    joblib.dump(paquete_modelo, model_path)

    return json.dumps(resultados, indent=4, ensure_ascii=False), mejor_modelo, columnas_validas.tolist()

def _guardar_error_vs_k(k_values, errores_por_distancia, image_prefix):
    error_path = f"{image_prefix}_error_vs_k.png"
    plt.figure(figsize=(9, 6))
    for distancia, errores in errores_por_distancia.items():
        plt.plot(k_values, errores, marker="o", label=distancia)
    plt.title("Error vs K")
    plt.xlabel("Número de vecinos")
    plt.ylabel("Error")
    plt.legend()
    plt.tight_layout()
    plt.savefig(error_path)
    plt.close()
    return error_path

def _comparar_distancias_knn(X_train, X_test, y_train, y_test, es_clf, k_values, weights, distancias):
    comparacion = []
    errores_por_distancia = {distancia: [] for distancia in distancias}

    for distancia in distancias:
        for k in k_values:
            if es_clf:
                modelo = KNeighborsClassifier(n_neighbors=k, weights=weights, metric=distancia)
                modelo.fit(X_train, y_train)
                y_pred = modelo.predict(X_test)
                error = 1 - accuracy_score(y_test, y_pred)
                score = accuracy_score(y_test, y_pred)
            else:
                modelo = KNeighborsRegressor(n_neighbors=k, weights=weights, metric=distancia)
                modelo.fit(X_train, y_train)
                y_pred = modelo.predict(X_test)
                error = np.sqrt(mean_squared_error(y_test, y_pred))
                score = r2_score(y_test, y_pred)

            errores_por_distancia[distancia].append(float(error))
            comparacion.append({
                "distancia": distancia,
                "k": int(k),
                "weights": weights,
                "error": float(error),
                "score": float(score)
            })

    return comparacion, errores_por_distancia

def KNN_model(X, y, test_size=0.3, cv_folds=5, model_filename='modelo_knn.pkl', seed=123):
    if y is None:
        raise ValueError("KNN requiere una variable objetivo.")

    X_numeric = X.select_dtypes(include=[np.number]).copy()
    columnas_validas = X_numeric.columns[X_numeric.nunique() > 1]
    X_numeric = X_numeric[columnas_validas]

    if X_numeric.empty:
        raise ValueError("KNN requiere columnas numéricas no constantes.")

    y_array = np.asarray(y)
    es_clf = _es_clasificacion(y_array)
    model_path = _ruta_con_id_unico(model_filename)
    image_prefix = _obtener_image_prefix(model_path)

    if es_clf:
        _, class_counts = np.unique(y_array, return_counts=True)
        if np.min(class_counts) < 2:
            raise ValueError("Clasificación con KNN requiere al menos 2 registros por clase.")
        stratify = y_array
    else:
        stratify = None

    X_train, X_test, y_train, y_test = train_test_split(
        X_numeric,
        y_array,
        test_size=test_size,
        random_state=seed,
        stratify=stratify
    )

    max_neighbors = min(15, len(X_train))
    if max_neighbors < 1:
        raise ValueError("KNN requiere más registros de entrenamiento.")

    k_values = sorted(set([k for k in [1, 3, 5, 7, 9, 11, 15] if k <= max_neighbors]))
    if not k_values:
        k_values = [1]

    cv_seguro = _cv_seguro(y_train, es_clf, cv_folds)
    distancias = ["euclidean", "manhattan", "minkowski"]

    if es_clf:
        modelo_base = KNeighborsClassifier()
        scoring = "accuracy"
    else:
        modelo_base = KNeighborsRegressor()
        scoring = "neg_root_mean_squared_error"

    parametros = {
        "n_neighbors": k_values,
        "weights": ["uniform", "distance"],
        "metric": distancias
    }

    grid_search = GridSearchCV(
        estimator=modelo_base,
        param_grid=parametros,
        scoring=scoring,
        cv=cv_seguro,
        n_jobs=-1
    )
    grid_search.fit(X_train, y_train)

    mejor_modelo = grid_search.best_estimator_
    y_pred = mejor_modelo.predict(X_test)
    mejor_weights = grid_search.best_params_["weights"]
    comparacion_distancias, errores_por_distancia = _comparar_distancias_knn(
        X_train,
        X_test,
        y_train,
        y_test,
        es_clf,
        k_values,
        mejor_weights,
        distancias
    )
    error_vs_k_path = _guardar_error_vs_k(k_values, errores_por_distancia, image_prefix)

    if es_clf:
        unique_values = np.sort(np.unique(y_array))
        average_mode = "binary" if len(unique_values) == 2 else "weighted"
        acc = accuracy_score(y_test, y_pred)

        if len(unique_values) == 2:
            pos_label = unique_values[-1]
            prec = precision_score(y_test, y_pred, pos_label=pos_label, zero_division=0)
            rec = recall_score(y_test, y_pred, pos_label=pos_label, zero_division=0)
            f1 = f1_score(y_test, y_pred, pos_label=pos_label, zero_division=0)
        else:
            prec = precision_score(y_test, y_pred, average=average_mode, zero_division=0)
            rec = recall_score(y_test, y_pred, average=average_mode, zero_division=0)
            f1 = f1_score(y_test, y_pred, average=average_mode, zero_division=0)

        confusion_path, matriz = _guardar_matriz_confusion(y_test, y_pred, unique_values, image_prefix)

        resultados = {
            "tipo_modelo": "KNN",
            "tipo_problema": "clasificacion",
            "metricas_precision": {
                "Accuracy": float(acc),
                "Precision": float(prec),
                "Recall": float(rec),
                "F1-Score": float(f1)
            },
            "matriz_confusion": matriz.tolist(),
            "comparacion_distancias": comparacion_distancias,
            "visualizaciones": {
                "error_vs_k": error_vs_k_path,
                "matriz_confusion": confusion_path,
                "real_vs_prediccion": None
            },
            "validacion": {
                "test_size": test_size,
                "cv_folds": cv_seguro,
                "mejor_score_grid": float(grid_search.best_score_)
            },
            "seleccion_variables": {
                "cantidad_original": X.shape[1],
                "cantidad_final": len(columnas_validas),
                "variables_utilizadas": columnas_validas.tolist()
            },
            "mejores_hiperparametros": grid_search.best_params_,
            "ruta_modelo_guardado": model_path
        }
    else:
        r2 = r2_score(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae = mean_absolute_error(y_test, y_pred)
        real_vs_pred_path = _guardar_real_vs_pred(y_test, y_pred, image_prefix)

        resultados = {
            "tipo_modelo": "KNN",
            "tipo_problema": "regresion",
            "metricas_precision": {
                "R2": float(r2),
                "RMSE": float(rmse),
                "MAE": float(mae)
            },
            "comparacion_distancias": comparacion_distancias,
            "visualizaciones": {
                "error_vs_k": error_vs_k_path,
                "matriz_confusion": None,
                "real_vs_prediccion": real_vs_pred_path
            },
            "validacion": {
                "test_size": test_size,
                "cv_folds": cv_seguro,
                "mejor_score_grid": float(grid_search.best_score_)
            },
            "seleccion_variables": {
                "cantidad_original": X.shape[1],
                "cantidad_final": len(columnas_validas),
                "variables_utilizadas": columnas_validas.tolist()
            },
            "mejores_hiperparametros": grid_search.best_params_,
            "ruta_modelo_guardado": model_path
        }

    paquete_modelo = {
        "modelo": mejor_modelo,
        "columnas": columnas_validas.tolist(),
        "resultados": resultados
    }
    joblib.dump(paquete_modelo, model_path)

    return json.dumps(resultados, indent=4, ensure_ascii=False), mejor_modelo, columnas_validas.tolist()

def KNN(*args, **kwargs):
    return KNN_model(*args, **kwargs)

def _ks_statistic(y_true_binary, y_prob):
    order = np.argsort(-y_prob)
    y_sorted = np.asarray(y_true_binary)[order]
    total_bad = y_sorted.sum()
    total_good = len(y_sorted) - total_bad
    if total_bad == 0 or total_good == 0:
        return 0.0

    cum_bad = np.cumsum(y_sorted) / total_bad
    cum_good = np.cumsum(1 - y_sorted) / total_good
    return float(np.max(np.abs(cum_bad - cum_good)))

def _guardar_score_distribution(scores, risk_segments, image_prefix):
    path = f"{image_prefix}_score_distribution.png"
    plt.figure(figsize=(10, 6))
    for segmento in ["bajo riesgo", "medio riesgo", "alto riesgo"]:
        valores = np.asarray(scores)[np.asarray(risk_segments) == segmento]
        if len(valores) > 0:
            plt.hist(valores, bins=18, alpha=0.65, label=segmento)
    plt.title("Distribución de score")
    plt.xlabel("Score")
    plt.ylabel("Frecuencia")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path)
    plt.close()
    return path

def _guardar_segmentos_riesgo(risk_segments, image_prefix):
    path = f"{image_prefix}_risk_segments.png"
    segmentos, counts = np.unique(risk_segments, return_counts=True)
    orden = ["bajo riesgo", "medio riesgo", "alto riesgo"]
    counts_ordenados = [int(counts[list(segmentos).index(seg)]) if seg in segmentos else 0 for seg in orden]
    plt.figure(figsize=(8, 6))
    plt.bar(orden, counts_ordenados, color=["#10b981", "#f59e0b", "#ef4444"])
    plt.title("Segmentos de riesgo")
    plt.xlabel("Segmento")
    plt.ylabel("Clientes")
    plt.tight_layout()
    plt.savefig(path)
    plt.close()
    return path

def _guardar_roc_credit(y_true_binary, y_prob, image_prefix):
    roc_path = f"{image_prefix}_roc.png"
    plt.figure(figsize=(8, 6))
    RocCurveDisplay.from_predictions(y_true_binary, y_prob)
    plt.title("Curva ROC - Credit Scoring")
    plt.tight_layout()
    plt.savefig(roc_path)
    plt.close()
    return roc_path

def Credit_scoring(X, y, test_size=0.3, cv_folds=5, model_filename='modelo_credit_scoring.pkl', seed=123, p_value_threshold=0.05):
    if y is None:
        raise ValueError("Credit_scoring requiere una variable objetivo binaria.")

    y_array = np.asarray(y)
    unique_values = np.sort(np.unique(y_array))
    if len(unique_values) != 2:
        raise ValueError(f"Credit_scoring requiere target binario. Se encontraron {len(unique_values)} valores: {unique_values}")

    _, class_counts = np.unique(y_array, return_counts=True)
    if np.min(class_counts) < 2:
        raise ValueError("Credit_scoring requiere al menos 2 registros por clase.")

    X_numeric = X.select_dtypes(include=[np.number]).copy()
    columnas_validas = X_numeric.columns[X_numeric.nunique() > 1]
    X_numeric = X_numeric[columnas_validas]
    if X_numeric.empty:
        raise ValueError("Credit_scoring requiere columnas numéricas no constantes.")

    X_train, X_test, y_train, y_test = train_test_split(
        X_numeric,
        y_array,
        test_size=test_size,
        random_state=seed,
        stratify=y_array
    )

    kb = SelectKBest(k="all", score_func=f_classif)
    kb.fit(X_train, y_train)
    pvalues = np.nan_to_num(kb.pvalues_, nan=1.0)
    columnas_score = X_train.columns[pvalues < p_value_threshold].tolist()
    if len(columnas_score) == 0:
        kb_fallback = SelectKBest(k=min(10, X_train.shape[1]), score_func=f_classif)
        kb_fallback.fit(X_train, y_train)
        columnas_score = X_train.columns[kb_fallback.get_support()].tolist()

    X_train_sel = X_train[columnas_score]
    X_test_sel = X_test[columnas_score]
    cv_seguro = _cv_seguro(y_train, True, cv_folds)

    parametros = {
        "C": [0.01, 0.1, 1, 10, 100],
        "penalty": ["l1", "l2"],
        "solver": ["liblinear", "saga"],
        "class_weight": ["balanced", None],
        "max_iter": [500, 1000]
    }
    modelo_base = LogisticRegression(random_state=seed)
    grid_search = GridSearchCV(
        estimator=modelo_base,
        param_grid=parametros,
        scoring="roc_auc",
        cv=cv_seguro,
        n_jobs=-1
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        grid_search.fit(X_train_sel, y_train)

    mejor_modelo = grid_search.best_estimator_
    probas = mejor_modelo.predict_proba(X_test_sel)
    if probas.shape[1] != 2:
        raise ValueError("Credit_scoring esperaba exactamente 2 columnas de probabilidad.")

    positive_label = unique_values[-1]
    y_test_binary = (y_test == positive_label).astype(int)
    y_pred = mejor_modelo.predict(X_test_sel)
    y_prob = probas[:, 1]

    auc = roc_auc_score(y_test_binary, y_prob)
    gini = float(2 * auc - 1)
    ks = _ks_statistic(y_test_binary, y_prob)
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, pos_label=positive_label, zero_division=0)
    rec = recall_score(y_test, y_pred, pos_label=positive_label, zero_division=0)

    eps = 1e-6
    odds = np.clip(y_prob, eps, 1 - eps) / np.clip(1 - y_prob, eps, 1 - eps)
    base_score = 600
    pdo = 50
    factor = pdo / np.log(2)
    offset = base_score
    scores = offset - factor * np.log(odds)

    q_low, q_high = np.quantile(scores, [0.33, 0.66])
    risk_segments = np.where(
        scores <= q_low,
        "alto riesgo",
        np.where(scores <= q_high, "medio riesgo", "bajo riesgo")
    )

    model_path = _ruta_con_id_unico(model_filename)
    image_prefix = _obtener_image_prefix(model_path)
    roc_path = _guardar_roc_credit(y_test_binary, y_prob, image_prefix)
    score_distribution_path = _guardar_score_distribution(scores, risk_segments, image_prefix)
    risk_segments_path = _guardar_segmentos_riesgo(risk_segments, image_prefix)
    coeficientes_path = _guardar_coeficientes_logisticos(mejor_modelo, columnas_score, image_prefix)
    confusion_path, matriz = _guardar_matriz_confusion(y_test, y_pred, unique_values, image_prefix)

    coeficientes = mejor_modelo.coef_[0]
    scorecard = [
        {
            "variable": col,
            "coeficiente_logistico": float(coef),
            "puntos_por_unidad": float(-factor * coef)
        }
        for col, coef in zip(columnas_score, coeficientes)
    ]

    segmentos_resumen = {
        segmento: int(np.sum(risk_segments == segmento))
        for segmento in ["bajo riesgo", "medio riesgo", "alto riesgo"]
    }

    resultados = {
        "tipo_modelo": "Credit_scoring",
        "tipo_problema": "clasificacion_binaria",
        "metricas_precision": {
            "KS statistic": float(ks),
            "Gini": float(gini),
            "ROC-AUC": float(auc),
            "Accuracy": float(acc),
            "Precision": float(prec),
            "Recall": float(rec)
        },
        "scorecard": {
            "base_score": base_score,
            "pdo": pdo,
            "factor": float(factor),
            "offset": float(offset),
            "positive_label_alto_riesgo": str(positive_label),
            "coeficientes": scorecard
        },
        "segmentos_riesgo": segmentos_resumen,
        "umbrales_score": {
            "alto_riesgo_hasta": float(q_low),
            "medio_riesgo_hasta": float(q_high),
            "bajo_riesgo_desde": float(q_high)
        },
        "matriz_confusion": matriz.tolist(),
        "visualizaciones": {
            "score_distribution": score_distribution_path,
            "segmentos_riesgo": risk_segments_path,
            "curva_roc": roc_path,
            "coeficientes": coeficientes_path,
            "matriz_confusion": confusion_path
        },
        "seleccion_variables": {
            "cantidad_original": X.shape[1],
            "cantidad_final": len(columnas_score),
            "variables_utilizadas": columnas_score
        },
        "validacion": {
            "test_size": test_size,
            "cv_folds": cv_seguro,
            "mejor_score_grid": float(grid_search.best_score_),
            "class_distribution": {
                str(label): int(count)
                for label, count in zip(unique_values, class_counts)
            }
        },
        "mejores_hiperparametros": grid_search.best_params_,
        "ruta_modelo_guardado": model_path
    }

    paquete_modelo = {
        "modelo": mejor_modelo,
        "columnas": columnas_score,
        "scorecard": resultados["scorecard"],
        "resultados": resultados
    }
    joblib.dump(paquete_modelo, model_path)

    return json.dumps(resultados, indent=4, ensure_ascii=False), mejor_modelo, columnas_score
