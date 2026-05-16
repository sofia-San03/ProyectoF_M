# Modelos

import json
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.cluster import KMeans
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error,
    mean_absolute_percentage_error, r2_score,
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, silhouette_score,
    confusion_matrix,
)
from sklearn.feature_selection import SelectKBest, f_regression, f_classif
from sklearn.compose import TransformedTargetRegressor

try:
    from sklearn_extra.cluster import KMedoids as _KMedoids
    _KMEDOIDS_AVAILABLE = True
except ImportError:
    _KMEDOIDS_AVAILABLE = False

def _seleccionar_variables_regresion(X_train, y_train, p_value_threshold=0.05):
    kb = SelectKBest(k="all", score_func=f_regression)
    kb.fit(X_train, y_train)
    mascara = kb.pvalues_ < p_value_threshold
    cols = X_train.columns[mascara].tolist()
    if not cols:
        kb2 = SelectKBest(k=min(3, X_train.shape[1]), score_func=f_regression)
        kb2.fit(X_train, y_train)
        cols = X_train.columns[kb2.get_support()].tolist()
    return cols

def _seleccionar_variables_clasificacion(X_train, y_train, p_value_threshold=0.05):
    kb = SelectKBest(k="all", score_func=f_classif)
    kb.fit(X_train, y_train)
    mascara = kb.pvalues_ < p_value_threshold
    cols = X_train.columns[mascara].tolist()
    if not cols:
        kb2 = SelectKBest(k=min(3, X_train.shape[1]), score_func=f_classif)
        kb2.fit(X_train, y_train)
        cols = X_train.columns[kb2.get_support()].tolist()
    return cols

def _limpiar_columnas_constantes(X_train, X_test):
    cols_validas = X_train.columns[X_train.nunique() > 1]
    return X_train[cols_validas], X_test[cols_validas]

def _metricas_clasificacion(y_test, y_pred, y_prob, unique_values):
    return {
        "Accuracy":  float(accuracy_score(y_test, y_pred)),
        "Precision": float(precision_score(y_test, y_pred, pos_label=unique_values[1], zero_division=0)),
        "Recall":    float(recall_score(y_test, y_pred,    pos_label=unique_values[1], zero_division=0)),
        "F1-Score":  float(f1_score(y_test, y_pred,        pos_label=unique_values[1], zero_division=0)),
        "ROC-AUC":   float(roc_auc_score(y_test, y_prob)) if y_prob is not None else 0.0,
    }

def _metricas_regresion(y_test, y_pred):
    mse = mean_squared_error(y_test, y_pred)
    return {
        "MSE":  float(mse),
        "RMSE": float(np.sqrt(mse)),
        "MAE":  float(mean_absolute_error(y_test, y_pred)),
        "MAPE": float(mean_absolute_percentage_error(y_test, y_pred)),
        "R2":   float(r2_score(y_test, y_pred)),
    }

# REGRESIÓN

# Regresión Lineal
def Regresion_lineal(X, y, test_size=0.3, cv_folds=5,
                     model_filename='modelo_lineal.pkl', seed=123,
                     p_value_threshold=0.05):

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed)

    X_train, X_test = _limpiar_columnas_constantes(X_train, X_test)
    cols = _seleccionar_variables_regresion(X_train, y_train, p_value_threshold)

    X_train_s, X_test_s = X_train[cols], X_test[cols]

    modelo_base = TransformedTargetRegressor(
        regressor=LinearRegression(),
        func=np.log1p, inverse_func=np.expm1
    )
    grid = GridSearchCV(
        modelo_base,
        {'regressor__fit_intercept': [True, False],
         'regressor__copy_X':        [True, False]},
        scoring='neg_mean_squared_error', cv=cv_folds, n_jobs=-1
    )
    grid.fit(X_train_s, y_train)
    mejor = grid.best_estimator_

    score  = mejor.score(X_train_s, y_train)
    y_pred = mejor.predict(X_test_s)
    metricas = _metricas_regresion(y_test, y_pred)

    cv_r2 = cross_val_score(mejor, X_train_s, y_train,
                            cv=cv_folds, scoring='r2', n_jobs=-1)

    resultados = {
        "seleccion_variables": {
            "cantidad_original": X.shape[1],
            "cantidad_final": len(cols),
            "variables_utilizadas": cols
        },
        "metricas_precision": {**metricas, "modelo.score": float(score)},
        "cross_validation_train_R2": {
            "media": float(cv_r2.mean()),
            "desviacion_estandar": float(cv_r2.std()),
            "folds": cv_folds
        },
        "mejores_hiperparametros": grid.best_params_,
        "ruta_modelo_guardado": model_filename
    }

    joblib.dump({"modelo": mejor, "columnas": cols}, model_filename)
    return json.dumps(resultados, indent=4, ensure_ascii=False), mejor, cols

# Regresor de Bosque Aleatorio
def Random_Forest_Regressor(X, y, test_size=0.3, cv_folds=5,
                              model_filename='modelo_rf_regressor.pkl',
                              seed=123, p_value_threshold=0.05):

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed)
    X_train, X_test = _limpiar_columnas_constantes(X_train, X_test)
    cols = _seleccionar_variables_regresion(X_train, y_train, p_value_threshold)
    X_train_s, X_test_s = X_train[cols], X_test[cols]

    param_grid = {
        'n_estimators': [100, 200],
        'max_depth':    [None, 10, 20],
        'min_samples_split': [2, 5],
    }
    grid = GridSearchCV(
        RandomForestRegressor(random_state=seed),
        param_grid, scoring='neg_mean_squared_error', cv=cv_folds, n_jobs=-1
    )
    grid.fit(X_train_s, y_train)
    mejor = grid.best_estimator_

    y_pred   = mejor.predict(X_test_s)
    metricas = _metricas_regresion(y_test, y_pred)
    cv_r2    = cross_val_score(mejor, X_train_s, y_train,
                               cv=cv_folds, scoring='r2', n_jobs=-1)

    importancias = dict(zip(cols, mejor.feature_importances_.tolist()))

    resultados = {
        "seleccion_variables": {
            "cantidad_original": X.shape[1],
            "cantidad_final": len(cols),
            "variables_utilizadas": cols
        },
        "metricas_precision": metricas,
        "feature_importances": importancias,
        "cross_validation_train_R2": {
            "media": float(cv_r2.mean()),
            "desviacion_estandar": float(cv_r2.std()),
            "folds": cv_folds
        },
        "mejores_hiperparametros": grid.best_params_,
        "ruta_modelo_guardado": model_filename
    }

    joblib.dump({"modelo": mejor, "columnas": cols}, model_filename)
    return json.dumps(resultados, indent=4, ensure_ascii=False), mejor, cols

# Regresor de Red Neuronal (MLP)
def Red_Neuronal_Regressor(X, y, test_size=0.3, cv_folds=5,
                            model_filename='modelo_mlp_regressor.pkl',
                            seed=123, p_value_threshold=0.05):

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed)
    X_train, X_test = _limpiar_columnas_constantes(X_train, X_test)
    cols = _seleccionar_variables_regresion(X_train, y_train, p_value_threshold)
    X_train_s, X_test_s = X_train[cols], X_test[cols]

    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train_s)
    X_test_sc  = scaler.transform(X_test_s)

    param_grid = {
        'hidden_layer_sizes': [(64,), (128,), (64, 32)],
        'activation':         ['relu', 'tanh'],
        'alpha':              [0.0001, 0.001],
        'max_iter':           [500],
    }
    grid = GridSearchCV(
        MLPRegressor(random_state=seed, early_stopping=True),
        param_grid, scoring='neg_mean_squared_error', cv=cv_folds, n_jobs=-1
    )
    grid.fit(X_train_sc, y_train)
    mejor = grid.best_estimator_

    y_pred   = mejor.predict(X_test_sc)
    mse      = round(float(mean_squared_error(y_test, y_pred)), 4)
    mae      = round(float(mean_absolute_error(y_test, y_pred)), 4)
    r2       = round(float(r2_score(y_test, y_pred)), 4)
    rmse     = round(float(np.sqrt(mse)), 4)

    cv_r2 = cross_val_score(mejor, X_train_sc, y_train,
                            cv=cv_folds, scoring='r2', n_jobs=-1)

    resultados = {
        "seleccion_variables": {
            "cantidad_original": X.shape[1],
            "cantidad_final": len(cols),
            "variables_utilizadas": cols
        },
        "metricas_precision": {
            "R2":   r2,
            "MAE":  mae,
            "MSE":  mse,
            "RMSE": rmse,
        },
        "cross_validation_train_R2": {
            "media": round(float(cv_r2.mean()), 4),
            "desviacion_estandar": round(float(cv_r2.std()), 4),
            "folds": cv_folds
        },
        "mejores_hiperparametros": grid.best_params_,
        "ruta_modelo_guardado": model_filename
    }

    joblib.dump({"modelo": mejor, "scaler": scaler, "columnas": cols}, model_filename)
    return json.dumps(resultados, indent=4, ensure_ascii=False), mejor, cols

# Regresor K-Vecinos más Cercanos
def KNN_Regressor(X, y, test_size=0.3, cv_folds=5,
                  model_filename='modelo_knn_regressor.pkl',
                  seed=123, p_value_threshold=0.05):

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed)
    X_train, X_test = _limpiar_columnas_constantes(X_train, X_test)
    cols = _seleccionar_variables_regresion(X_train, y_train, p_value_threshold)
    X_train_s, X_test_s = X_train[cols], X_test[cols]

    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train_s)
    X_test_sc  = scaler.transform(X_test_s)

    param_grid = {
        'n_neighbors': [3, 5, 7, 11, 15],
        'weights':     ['uniform', 'distance'],
        'metric':      ['euclidean', 'manhattan'],
    }
    grid = GridSearchCV(
        KNeighborsRegressor(),
        param_grid, scoring='neg_mean_squared_error', cv=cv_folds, n_jobs=-1
    )
    grid.fit(X_train_sc, y_train)
    mejor = grid.best_estimator_

    y_pred = mejor.predict(X_test_sc)
    mse    = round(float(mean_squared_error(y_test, y_pred)), 4)
    mae    = round(float(mean_absolute_error(y_test, y_pred)), 4)
    r2     = round(float(r2_score(y_test, y_pred)), 4)
    rmse   = round(float(np.sqrt(mse)), 4)

    cv_r2 = cross_val_score(mejor, X_train_sc, y_train,
                            cv=cv_folds, scoring='r2', n_jobs=-1)

    resultados = {
        "seleccion_variables": {
            "cantidad_original": X.shape[1],
            "cantidad_final": len(cols),
            "variables_utilizadas": cols
        },
        "metricas_precision": {
            "R2":   r2,
            "MAE":  mae,
            "MSE":  mse,
            "RMSE": rmse,
        },
        "cross_validation_train_R2": {
            "media": round(float(cv_r2.mean()), 4),
            "desviacion_estandar": round(float(cv_r2.std()), 4),
            "folds": cv_folds
        },
        "mejores_hiperparametros": grid.best_params_,
        "ruta_modelo_guardado": model_filename
    }

    joblib.dump({"modelo": mejor, "scaler": scaler, "columnas": cols}, model_filename)
    return json.dumps(resultados, indent=4, ensure_ascii=False), mejor, cols

# CLASIFICACIÓN

# Árbol de Decisión
def Arbol_decision(X, y, test_size=0.3, cv_folds=5,
                   model_filename='modelo_arbol.pkl', seed=123,
                   p_value_threshold=0.05):
    unique_values = np.sort(np.unique(y))
    if len(unique_values) != 2:
        raise ValueError(
            f"Árbol de Decisión requiere variable binaria. "
            f"Se encontraron {len(unique_values)} valores."
        )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed, stratify=y)
    cols = _seleccionar_variables_clasificacion(X_train, y_train, p_value_threshold)
    X_train_s, X_test_s = X_train[cols], X_test[cols]

    param_grid = {
        'criterion':          ['gini', 'entropy'],
        'max_depth':          [None, 5, 10, 20],
        'min_samples_split':  [2, 5, 10],
        'min_samples_leaf':   [1, 2, 4],
    }
    grid = GridSearchCV(
        DecisionTreeClassifier(random_state=seed),
        param_grid, scoring='accuracy', cv=cv_folds, n_jobs=-1
    )
    grid.fit(X_train_s, y_train)
    mejor = grid.best_estimator_

    plt.figure(figsize=(20, 10))
    plot_tree(mejor, feature_names=cols,
              class_names=[str(v) for v in unique_values],
              filled=True, rounded=True)
    image_path = model_filename.replace('.pkl', '.png')
    plt.savefig(image_path, dpi=100, bbox_inches='tight')
    plt.close()

    y_pred = mejor.predict(X_test_s)
    y_prob = mejor.predict_proba(X_test_s)[:, 1]
    metricas = _metricas_clasificacion(y_test, y_pred, y_prob, unique_values)
    cv_acc = cross_val_score(mejor, X_train_s, y_train,
                             cv=cv_folds, scoring='accuracy', n_jobs=-1)

    resultados = {
        "seleccion_variables": {
            "cantidad_original": X.shape[1],
            "cantidad_final": len(cols),
            "variables_utilizadas": cols
        },
        "metricas_precision": metricas,
        "cross_validation_train_Accuracy": {
            "media": float(cv_acc.mean()),
            "desviacion_estandar": float(cv_acc.std()),
            "folds": cv_folds
        },
        "mejores_hiperparametros": grid.best_params_,
        "ruta_modelo_guardado": model_filename,
        "ruta_imagen_arbol": image_path
    }
    joblib.dump({"modelo": mejor, "columnas": cols}, model_filename)
    return json.dumps(resultados, indent=4, ensure_ascii=False), mejor, cols

# Regresión Logística
def Regresion_logistica(X, y, test_size=0.3, cv_folds=5,
                         model_filename='modelo_logistico.pkl', seed=123,
                         p_value_threshold=0.05):
    unique_values = np.sort(np.unique(y))
    if len(unique_values) != 2:
        raise ValueError(
            f"Regresión Logística requiere variable binaria (2 valores). "
            f"Se encontraron {len(unique_values)}: {unique_values}"
        )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed, stratify=y)
    X_train, X_test = _limpiar_columnas_constantes(X_train, X_test)
    cols = _seleccionar_variables_clasificacion(X_train, y_train, p_value_threshold)
    X_train_s, X_test_s = X_train[cols], X_test[cols]

    param_grid = {
        'C':        [0.01, 0.1, 1, 10],
        'solver':   ['liblinear', 'lbfgs'],
        'max_iter': [200, 500],
    }
    grid = GridSearchCV(
        LogisticRegression(random_state=seed),
        param_grid, scoring='accuracy', cv=cv_folds, n_jobs=-1
    )
    grid.fit(X_train_s, y_train)
    mejor = grid.best_estimator_

    y_pred = mejor.predict(X_test_s)
    y_prob = mejor.predict_proba(X_test_s)[:, 1] \
             if hasattr(mejor, "predict_proba") else None
    metricas = _metricas_clasificacion(y_test, y_pred, y_prob, unique_values)
    cv_acc = cross_val_score(mejor, X_train_s, y_train,
                             cv=cv_folds, scoring='accuracy', n_jobs=-1)

    resultados = {
        "seleccion_variables": {
            "cantidad_original": X.shape[1],
            "cantidad_final": len(cols),
            "variables_utilizadas": cols
        },
        "metricas_precision": metricas,
        "cross_validation_train_Accuracy": {
            "media": float(cv_acc.mean()),
            "desviacion_estandar": float(cv_acc.std()),
            "folds": cv_folds
        },
        "mejores_hiperparametros": grid.best_params_,
        "ruta_modelo_guardado": model_filename
    }
    joblib.dump({"modelo": mejor, "columnas": cols}, model_filename)
    return json.dumps(resultados, indent=4, ensure_ascii=False), mejor, cols

# Clasificador de Bosque Aleatorio
def Random_Forest_Clasificador(X, y, test_size=0.3, cv_folds=5,
                                 model_filename='modelo_rf_clf.pkl', seed=123,
                                 p_value_threshold=0.05):
    unique_values = np.sort(np.unique(y))
    stratify_y = y if len(unique_values) > 1 else None
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed, stratify=stratify_y)
    X_train, X_test = _limpiar_columnas_constantes(X_train, X_test)
    cols = _seleccionar_variables_clasificacion(X_train, y_train, p_value_threshold)
    X_train_s, X_test_s = X_train[cols], X_test[cols]

    param_grid = {
        'n_estimators': [100, 200],
        'max_depth':    [None, 10, 20],
        'min_samples_split': [2, 5],
    }
    grid = GridSearchCV(
        RandomForestClassifier(random_state=seed),
        param_grid, scoring='accuracy', cv=cv_folds, n_jobs=-1
    )
    grid.fit(X_train_s, y_train)
    mejor = grid.best_estimator_

    y_pred = mejor.predict(X_test_s)
    es_binario = len(unique_values) == 2
    y_prob = mejor.predict_proba(X_test_s)[:, 1] if es_binario else None

    avg = 'binary' if es_binario else 'weighted'
    metricas = {
        "Accuracy":  float(accuracy_score(y_test, y_pred)),
        "Precision": float(precision_score(y_test, y_pred, average=avg, zero_division=0)),
        "Recall":    float(recall_score(y_test, y_pred,    average=avg, zero_division=0)),
        "F1-Score":  float(f1_score(y_test, y_pred,        average=avg, zero_division=0)),
        "ROC-AUC":   float(roc_auc_score(y_test, y_prob)) if y_prob is not None else None,
    }

    importancias = dict(zip(cols, mejor.feature_importances_.tolist()))
    cv_acc = cross_val_score(mejor, X_train_s, y_train,
                             cv=cv_folds, scoring='accuracy', n_jobs=-1)

    resultados = {
        "seleccion_variables": {
            "cantidad_original": X.shape[1],
            "cantidad_final": len(cols),
            "variables_utilizadas": cols
        },
        "metricas_precision": metricas,
        "feature_importances": importancias,
        "cross_validation_train_Accuracy": {
            "media": float(cv_acc.mean()),
            "desviacion_estandar": float(cv_acc.std()),
            "folds": cv_folds
        },
        "mejores_hiperparametros": grid.best_params_,
        "ruta_modelo_guardado": model_filename
    }
    joblib.dump({"modelo": mejor, "columnas": cols}, model_filename)
    return json.dumps(resultados, indent=4, ensure_ascii=False), mejor, cols

# Clasificador de Red Neuronal (MLP)
def Red_Neuronal_Clasificador(X, y, test_size=0.3, cv_folds=5,
                               model_filename='modelo_mlp_clf.pkl', seed=123,
                               p_value_threshold=0.05):
    unique_values = np.sort(np.unique(y))
    if len(unique_values) != 2:
        raise ValueError(
            f"Red Neuronal Clasificador requiere variable binaria (2 valores). "
            f"Se encontraron {len(unique_values)}: {unique_values}"
        )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed, stratify=y)
    X_train, X_test = _limpiar_columnas_constantes(X_train, X_test)
    cols = _seleccionar_variables_clasificacion(X_train, y_train, p_value_threshold)
    X_train_s, X_test_s = X_train[cols], X_test[cols]

    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train_s)
    X_test_sc  = scaler.transform(X_test_s)

    param_grid = {
        'hidden_layer_sizes': [(64,), (128,), (64, 32)],
        'activation':         ['relu', 'tanh'],
        'alpha':              [0.0001, 0.001],
        'max_iter':           [500],
    }
    grid = GridSearchCV(
        MLPClassifier(random_state=seed, early_stopping=True),
        param_grid, scoring='accuracy', cv=cv_folds, n_jobs=-1
    )
    grid.fit(X_train_sc, y_train)
    mejor = grid.best_estimator_

    y_pred  = mejor.predict(X_test_sc)
    y_prob  = mejor.predict_proba(X_test_sc)[:, 1] \
              if hasattr(mejor, "predict_proba") else None
    cm      = confusion_matrix(y_test, y_pred).tolist()
    cv_acc  = cross_val_score(mejor, X_train_sc, y_train,
                              cv=cv_folds, scoring='accuracy', n_jobs=-1)

    resultados = {
        "seleccion_variables": {
            "cantidad_original": X.shape[1],
            "cantidad_final": len(cols),
            "variables_utilizadas": cols
        },
        "metricas_precision": {
            "Accuracy":        round(float(accuracy_score(y_test, y_pred)), 4),
            "Precision":       round(float(precision_score(y_test, y_pred, pos_label=unique_values[1], zero_division=0)), 4),
            "Recall":          round(float(recall_score(y_test, y_pred,    pos_label=unique_values[1], zero_division=0)), 4),
            "F1-Score":        round(float(f1_score(y_test, y_pred,        pos_label=unique_values[1], zero_division=0)), 4),
            "Matriz_Confusion": cm,
        },
        "cross_validation_train_Accuracy": {
            "media": round(float(cv_acc.mean()), 4),
            "desviacion_estandar": round(float(cv_acc.std()), 4),
            "folds": cv_folds
        },
        "mejores_hiperparametros": grid.best_params_,
        "ruta_modelo_guardado": model_filename
    }

    joblib.dump({"modelo": mejor, "scaler": scaler, "columnas": cols}, model_filename)
    return json.dumps(resultados, indent=4, ensure_ascii=False), mejor, cols

# Clasificador K-Vecinos más Cercanos
def KNN_Clasificador(X, y, test_size=0.3, cv_folds=5,
                     model_filename='modelo_knn_clf.pkl', seed=123,
                     p_value_threshold=0.05):
    unique_values = np.sort(np.unique(y))
    if len(unique_values) != 2:
        raise ValueError(
            f"KNN Clasificador requiere variable binaria (2 valores). "
            f"Se encontraron {len(unique_values)}: {unique_values}"
        )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed, stratify=y)
    X_train, X_test = _limpiar_columnas_constantes(X_train, X_test)
    cols = _seleccionar_variables_clasificacion(X_train, y_train, p_value_threshold)
    X_train_s, X_test_s = X_train[cols], X_test[cols]

    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train_s)
    X_test_sc  = scaler.transform(X_test_s)

    param_grid = {
        'n_neighbors': [3, 5, 7, 11, 15],
        'weights':     ['uniform', 'distance'],
        'metric':      ['euclidean', 'manhattan'],
    }
    grid = GridSearchCV(
        KNeighborsClassifier(),
        param_grid, scoring='accuracy', cv=cv_folds, n_jobs=-1
    )
    grid.fit(X_train_sc, y_train)
    mejor = grid.best_estimator_

    y_pred  = mejor.predict(X_test_sc)
    y_prob  = mejor.predict_proba(X_test_sc)[:, 1] \
              if hasattr(mejor, "predict_proba") else None
    cm      = confusion_matrix(y_test, y_pred).tolist()
    cv_acc  = cross_val_score(mejor, X_train_sc, y_train,
                              cv=cv_folds, scoring='accuracy', n_jobs=-1)

    resultados = {
        "seleccion_variables": {
            "cantidad_original": X.shape[1],
            "cantidad_final": len(cols),
            "variables_utilizadas": cols
        },
        "metricas_precision": {
            "Accuracy":        round(float(accuracy_score(y_test, y_pred)), 4),
            "Precision":       round(float(precision_score(y_test, y_pred, pos_label=unique_values[1], zero_division=0)), 4),
            "Recall":          round(float(recall_score(y_test, y_pred,    pos_label=unique_values[1], zero_division=0)), 4),
            "F1-Score":        round(float(f1_score(y_test, y_pred,        pos_label=unique_values[1], zero_division=0)), 4),
            "Matriz_Confusion": cm,
        },
        "cross_validation_train_Accuracy": {
            "media": round(float(cv_acc.mean()), 4),
            "desviacion_estandar": round(float(cv_acc.std()), 4),
            "folds": cv_folds
        },
        "mejores_hiperparametros": grid.best_params_,
        "ruta_modelo_guardado": model_filename
    }

    joblib.dump({"modelo": mejor, "scaler": scaler, "columnas": cols}, model_filename)
    return json.dumps(resultados, indent=4, ensure_ascii=False), mejor, cols

# CLUSTERING

# Agrupamiento por K-Medios
def KMeans_Clustering(X, n_clusters_range=(2, 8),
                       model_filename='modelo_kmeans.pkl', seed=123):
    X_num = X.select_dtypes(include='number').copy()
    if X_num.empty:
        raise ValueError("KMeans requiere columnas numéricas.")

    scaler   = StandardScaler()
    X_scaled = scaler.fit_transform(X_num)

    best_k, best_score, best_modelo = 2, -1, None
    silhouette_scores = {}

    for k in range(n_clusters_range[0], n_clusters_range[1] + 1):
        km = KMeans(n_clusters=k, random_state=seed, n_init=10)
        labels = km.fit_predict(X_scaled)
        score = silhouette_score(X_scaled, labels)
        silhouette_scores[k] = round(float(score), 4)
        if score > best_score:
            best_k, best_score, best_modelo = k, score, km

    cols_usadas = X_num.columns.tolist()

    labels_finales = best_modelo.labels_
    distribucion = {
        f"cluster_{i}": int(np.sum(labels_finales == i))
        for i in range(best_k)
    }

    resultados = {
        "modelo": "KMeans",
        "k_optimo": best_k,
        "silhouette_optimo": round(best_score, 4),
        "silhouette_por_k": silhouette_scores,
        "distribucion_clusters": distribucion,
        "variables_utilizadas": cols_usadas,
        "n_features": len(cols_usadas),
        "ruta_modelo_guardado": model_filename
    }

    joblib.dump(
        {"modelo": best_modelo, "scaler": scaler, "columnas": cols_usadas},
        model_filename
    )
    return json.dumps(resultados, indent=4, ensure_ascii=False), best_modelo, cols_usadas

# Agrupamiento por K-Medioides
def KMedoids_Clustering(X, n_clusters_range=(2, 8),
                         model_filename='modelo_kmedoids.pkl', seed=123):
    X_num = X.select_dtypes(include='number').copy()
    if X_num.empty:
        raise ValueError("KMedoids requiere columnas numéricas.")

    scaler   = StandardScaler()
    X_scaled = scaler.fit_transform(X_num)
    cols_usadas = X_num.columns.tolist()

    best_k, best_score, best_modelo = 2, -1, None
    silhouette_scores = {}
    inercia_scores    = {}

    for k in range(n_clusters_range[0], n_clusters_range[1] + 1):
        if _KMEDOIDS_AVAILABLE:
            modelo_k = _KMedoids(n_clusters=k, random_state=seed, metric='euclidean')
        else:
            modelo_k = KMeans(n_clusters=k, random_state=seed, n_init=10)

        labels = modelo_k.fit_predict(X_scaled)
        score  = silhouette_score(X_scaled, labels)
        silhouette_scores[k] = round(float(score), 4)

        if _KMEDOIDS_AVAILABLE:
            disparidad = 0.0
            for idx, centro_idx in enumerate(modelo_k.medoid_indices_):
                mask = labels == idx
                disparidad += float(np.sum(
                    np.linalg.norm(X_scaled[mask] - X_scaled[centro_idx], axis=1)
                ))
            inercia_scores[k] = round(disparidad, 4)
        else:
            inercia_scores[k] = round(float(modelo_k.inertia_), 4)

        if score > best_score:
            best_k, best_score, best_modelo = k, score, modelo_k

    labels_finales = best_modelo.labels_
    distribucion = {
        f"cluster_{i}": int(np.sum(labels_finales == i))
        for i in range(best_k)
    }

    resultados = {
        "modelo": "KMedoids" if _KMEDOIDS_AVAILABLE else "KMedoids_fallback_KMeans",
        "sklearn_extra_disponible": _KMEDOIDS_AVAILABLE,
        "k_optimo": best_k,
        "Silueta": round(best_score, 4),
        "silhouette_por_k": silhouette_scores,
        "disparidad_por_k": inercia_scores,
        "distribucion_clusters": distribucion,
        "variables_utilizadas": cols_usadas,
        "n_features": len(cols_usadas),
        "ruta_modelo_guardado": model_filename
    }

    joblib.dump(
        {"modelo": best_modelo, "scaler": scaler, "columnas": cols_usadas},
        model_filename
    )
    return json.dumps(resultados, indent=4, ensure_ascii=False), best_modelo, cols_usadas

# Agrupamiento por K-Medianas
def KMedianas_Clustering(X, n_clusters_range=(2, 8),
                          model_filename='modelo_kmedianas.pkl', seed=123):
    X_num = X.select_dtypes(include='number').copy()
    if X_num.empty:
        raise ValueError("KMedianas requiere columnas numéricas.")

    scaler   = StandardScaler()
    X_arr    = scaler.fit_transform(X_num)
    cols_usadas = X_num.columns.tolist()

    class KMedianasModel:
        def __init__(self, n_clusters=3, max_iter=300, tol=1e-4, random_state=None):
            self.n_clusters   = n_clusters
            self.max_iter     = max_iter
            self.tol          = tol
            self.random_state = random_state
            self.medianas_    = None
            self.labels_      = None
            self.inercia_     = None

        def fit(self, X):
            rng = np.random.default_rng(self.random_state)
            idx_init = rng.choice(len(X), self.n_clusters, replace=False)
            medianas = X[idx_init].copy()

            for _ in range(self.max_iter):
                distancias = np.array([
                    np.sum(np.abs(X - med), axis=1) for med in medianas
                ]).T                                          # (n_samples, k)
                labels = np.argmin(distancias, axis=1)

                nuevas = np.array([
                    np.median(X[labels == k], axis=0)
                    if np.any(labels == k) else medianas[k]
                    for k in range(self.n_clusters)
                ])

                cambio = np.max(np.sum(np.abs(nuevas - medianas), axis=1))
                medianas = nuevas
                if cambio < self.tol:
                    break

            self.medianas_ = medianas
            self.labels_   = labels
            self.inercia_  = float(np.sum([
                np.sum(np.abs(X[labels == k] - medianas[k]))
                for k in range(self.n_clusters)
                if np.any(labels == k)
            ]))
            return self

        def fit_predict(self, X):
            self.fit(X)
            return self.labels_

        def predict(self, X):
            distancias = np.array([
                np.sum(np.abs(X - med), axis=1) for med in self.medianas_
            ]).T
            return np.argmin(distancias, axis=1)

    best_k, best_score, best_modelo = 2, -1, None
    silhouette_scores = {}
    inercia_scores    = {}

    for k in range(n_clusters_range[0], n_clusters_range[1] + 1):
        modelo_k = KMedianasModel(n_clusters=k, random_state=seed)
        labels   = modelo_k.fit_predict(X_arr)

        # Silhouette requiere al menos 2 etiquetas distintas
        n_etiquetas = len(np.unique(labels))
        if n_etiquetas < 2:
            silhouette_scores[k] = None
            inercia_scores[k]    = round(modelo_k.inercia_, 4)
            continue

        score = silhouette_score(X_arr, labels)
        silhouette_scores[k] = round(float(score), 4)
        inercia_scores[k]    = round(modelo_k.inercia_, 4)

        if score > best_score:
            best_k, best_score, best_modelo = k, score, modelo_k

    if best_modelo is None:
        best_modelo = KMedianasModel(n_clusters=2, random_state=seed)
        best_modelo.fit_predict(X_arr)
        best_k    = 2
        best_score = 0.0

    labels_finales = best_modelo.labels_
    distribucion = {
        f"cluster_{i}": int(np.sum(labels_finales == i))
        for i in range(best_k)
    }

    resultados = {
        "modelo": "KMedianas",
        "k_optimo": best_k,
        "Silueta": round(best_score, 4),
        "silhouette_por_k": silhouette_scores,
        "inercia_L1_por_k": inercia_scores,
        "distribucion_clusters": distribucion,
        "variables_utilizadas": cols_usadas,
        "n_features": len(cols_usadas),
        "ruta_modelo_guardado": model_filename
    }

    joblib.dump(
        {"modelo": best_modelo, "scaler": scaler, "columnas": cols_usadas},
        model_filename
    )
    return json.dumps(resultados, indent=4, ensure_ascii=False), best_modelo, cols_usadas

# Orquestador
def orquestador_modelos_interno(X, y, tipo_modelo):
    clave = tipo_modelo.lower().strip()

    if clave == "regresion_lineal":
        return Regresion_lineal(X, y)

    elif clave == "random_forest_regressor":
        return Random_Forest_Regressor(X, y)

    elif clave == "red_neuronal_regressor":
        return Red_Neuronal_Regressor(X, y)

    elif clave == "knn_regressor":
        return KNN_Regressor(X, y)

    elif clave == "arbol_decision":
        return Arbol_decision(X, y)

    elif clave == "regresion_logistica":
        return Regresion_logistica(X, y)

    elif clave == "random_forest_clasificador":
        return Random_Forest_Clasificador(X, y)

    elif clave == "red_neuronal_clasificador":
        return Red_Neuronal_Clasificador(X, y)

    elif clave == "knn_clasificador":
        return KNN_Clasificador(X, y)

    elif clave == "kmeans":
        return KMeans_Clustering(X)

    elif clave == "kmedoids":
        return KMedoids_Clustering(X)

    elif clave == "kmedianas":
        return KMedianas_Clustering(X)

    else:
        raise ValueError(
            f"ERROR: El modelo solicitado '{tipo_modelo}' no coincide "
            f"con ninguna arquitectura registrada en el orquestador. "
            f"Modelos válidos: regresion_lineal, random_forest_regressor, "
            f"red_neuronal_regressor, knn_regressor, arbol_decision, "
            f"regresion_logistica, random_forest_clasificador, "
            f"red_neuronal_clasificador, knn_clasificador, "
            f"kmeans, kmedoids, kmedianas."
        )
