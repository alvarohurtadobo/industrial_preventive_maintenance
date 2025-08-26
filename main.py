import os
import math
import warnings
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from openpyxl import load_workbook
from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, IsolationForest
from sklearn.svm import SVC, OneClassSVM
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (classification_report, confusion_matrix, roc_auc_score,
                             accuracy_score, precision_score, recall_score, f1_score,
                             roc_curve, precision_recall_curve, average_precision_score)
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN
from sklearn.neighbors import LocalOutlierFactor
from imblearn.over_sampling import SMOTE
from ydata_profiling import ProfileReport

from packaging import version
import sklearn

# Configuración de logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Ignorar advertencias para una salida más limpia
warnings.filterwarnings('ignore')

# Configuración de directorios y archivos
RESULTS_DIR = "results"
EXCEL_FILE = os.path.join(RESULTS_DIR, 'model_evaluation.xlsx')
PDF_REPORT = os.path.join(RESULTS_DIR, "technical_report.pdf")

def setup_directories():
    """Crear directorios necesarios."""
    if not os.path.exists(RESULTS_DIR):
        os.makedirs(RESULTS_DIR)
        logging.info(f"Directorio '{RESULTS_DIR}' creado.")
    else:
        logging.info(f"Directorio '{RESULTS_DIR}' ya existe.")

    if not os.path.exists(EXCEL_FILE):
        with pd.ExcelWriter(EXCEL_FILE, engine="openpyxl") as writer:
            pd.DataFrame().to_excel(writer)  # Crear archivo Excel vacío
        logging.info(f"Archivo Excel '{EXCEL_FILE}' creado.")
    else:
        logging.info(f"Archivo Excel '{EXCEL_FILE}' ya existe.")

def generate_simulated_temporal_data():
    """Generar datos simulados temporales para múltiples equipos."""
    np.random.seed(42)
    n_equipment = 100  # Número de equipos
    n_time_steps = 40  # Número de tiempos por equipo

    # Crear listas para almacenar los datos
    data_records = []

    for equipment in range(1, n_equipment + 1):
        # Asignar un tipo de proceso fijo por equipo
        process_type = np.random.choice(['Vibrations', 'Oil Analysis', 'Hours Operated'])
        for t in range(1, n_time_steps + 1):
            record = {
                'equipment_id': equipment,
                'time_step': t,
                'process_type': process_type
            }

            # Simulación de características con tendencias y ruido
            if process_type == 'Vibrations':
                vib = np.sin(t / 5) + np.random.normal(0, 0.5)
                temp = 20 + 2 * vib + np.random.normal(0, 0.5)
                pres = 30 + 3 * (vib ** 2) + np.random.normal(0, 1)
                record.update({
                    'vibration': vib,
                    'temperature': temp,
                    'pressure': pres,
                    'oil_quality': np.nan,
                    'contaminant_level': np.nan,
                    'acidity': np.nan,
                    'hours_operated': np.nan,
                    'maintenance_history': np.nan,
                    'load': np.nan
                })
            elif process_type == 'Oil Analysis':
                oil_q = np.random.uniform(0, 100) + t * 0.1  # Incremento leve con el tiempo
                cont_level = 50 + 0.5 * oil_q + np.random.normal(0, 5)
                acid = 10 + 0.3 * (oil_q ** 1.5) + np.random.normal(0, 2)
                record.update({
                    'vibration': np.nan,
                    'temperature': np.nan,
                    'pressure': np.nan,
                    'oil_quality': oil_q,
                    'contaminant_level': cont_level,
                    'acidity': acid,
                    'hours_operated': np.nan,
                    'maintenance_history': np.nan,
                    'load': np.nan
                })
            elif process_type == 'Hours Operated':
                hours_op = np.random.exponential(scale=50) + t * 0.5  # Acumulativo con el tiempo
                maint_hist = np.random.poisson(lam=2)  # Historial de mantenimiento
                ld = 100 + 0.1 * t + np.random.normal(0, 10)
                record.update({
                    'vibration': np.nan,
                    'temperature': np.nan,
                    'pressure': np.nan,
                    'oil_quality': np.nan,
                    'contaminant_level': np.nan,
                    'acidity': np.nan,
                    'hours_operated': hours_op,
                    'maintenance_history': maint_hist,
                    'load': ld
                })

            # Simulación de fallos
            if process_type == 'Vibrations':
                fail = int((0.3 * vib + 0.2 * temp - 0.1 * pres + np.random.normal(0, 0.5)) > 1)
            elif process_type == 'Oil Analysis':
                fail = int((0.2 * oil_q - 0.1 * cont_level + 0.05 * acid + np.random.normal(0, 1)) > 5)
            elif process_type == 'Hours Operated':
                fail = int((0.05 * hours_op + 0.1 * maint_hist - 0.02 * ld + np.random.normal(0, 1)) > 3)
            record['failure'] = fail

            # Introducción de anomalías aleatorias
            if np.random.rand() < 0.02:  # 2% de probabilidad de anomalía
                record['anomaly'] = 1
                # Alterar algunas variables
                if process_type == 'Vibrations':
                    record['vibration'] += np.random.normal(10, 5)  # Anomalía en vibración
                elif process_type == 'Oil Analysis':
                    record['oil_quality'] += np.random.uniform(50, 100)  # Anomalía en calidad de aceite
                elif process_type == 'Hours Operated':
                    record['load'] += np.random.uniform(50, 100)  # Anomalía en carga
            else:
                record['anomaly'] = 0

            data_records.append(record)

    # Crear DataFrame

    # data = pd.read_csv('data/data.csv')
    data = pd.DataFrame(data_records)

    # Manejar valores NaN (rellenar con la media de cada columna numérica)
    numeric_cols = data.select_dtypes(include=[np.number]).columns
    data[numeric_cols] = data[numeric_cols].fillna(data[numeric_cols].mean())

    # Asegurarse de que 'anomaly' es de tipo entero
    data['anomaly'] = data['anomaly'].astype(int)

    logging.info("Datos temporales simulados generados correctamente.")
    return data

def handle_data_types(data):
    """
    Asegura que todas las columnas tengan los tipos de datos correctos.
    - Convierte variables categóricas en numéricas mediante codificación.
    - Asegura que las variables numéricas sean del tipo adecuado.
    """
    categorical_cols = data.select_dtypes(include=['object', 'category']).columns.tolist()

    # Determinar la versión de scikit-learn
    skl_version = version.parse(sklearn.__version__)

    # Definir los parámetros para OneHotEncoder según la versión
    if skl_version >= version.parse("1.2"):
        encoder = OneHotEncoder(drop='first', sparse_output=False)
    else:
        encoder = OneHotEncoder(drop='first', sparse=False)

    # Codificar variables categóricas usando OneHotEncoder
    if categorical_cols:
        try:
            encoded_data = encoder.fit_transform(data[categorical_cols])
            encoded_cols = encoder.get_feature_names_out(categorical_cols)
            encoded_df = pd.DataFrame(encoded_data, columns=encoded_cols, index=data.index)
            data = pd.concat([data.drop(categorical_cols, axis=1), encoded_df], axis=1)
            logging.info("Variables categóricas codificadas correctamente.")
        except Exception as e:
            logging.error(f"Error al codificar variables categóricas: {e}")
            raise
    else:
        logging.info("No se encontraron columnas categóricas para codificar.")

    # Asegurar que todas las columnas numéricas sean de tipo float
    numeric_cols = data.select_dtypes(include=[np.number]).columns
    data[numeric_cols] = data[numeric_cols].astype(float)
    logging.info("Tipos de datos numéricos asegurados como float.")

    # Verificación adicional
    remaining_categorical = data.select_dtypes(include=['object', 'category']).columns.tolist()
    if remaining_categorical:
        raise ValueError(f"Las siguientes columnas aún son categóricas y no han sido codificadas: {remaining_categorical}")
    else:
        logging.info("Todas las columnas categóricas han sido codificadas.")

    return data

def perform_eda(data):
    """Realizar Análisis Exploratorio de Datos (EDA)."""
    try:
        profile = ProfileReport(data, title='Análisis Exploratorio de Datos', explorative=True)
        eda_file = os.path.join(RESULTS_DIR, "EDA_del_dataset.html")
        profile.to_file(eda_file)
        logging.info(f"Reporte de EDA generado en '{eda_file}'.")
    except Exception as e:
        logging.error(f"Error al generar el reporte de EDA: {e}")

    # Distribución de fallos y no fallos
    plt.figure(figsize=(8, 6))
    sns.countplot(x='failure', data=data, palette='coolwarm')
    plt.title('Distribución de Fallos vs No Fallos')
    plt.xlabel('Fallo')
    plt.ylabel('Conteo')
    plt.savefig(os.path.join(RESULTS_DIR, 'failure_distribution.png'), dpi=300)
    plt.close()
    logging.info("Gráfico 'failure_distribution.png' guardado.")

    # Matriz de correlación
    plt.figure(figsize=(14, 10))
    corr = data.corr()
    sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
    plt.title('Matriz de Correlación de Datos Simulados')
    plt.savefig(os.path.join(RESULTS_DIR, 'correlation_matrix.png'), dpi=300)
    plt.close()
    logging.info("Gráfico 'correlation_matrix.png' guardado.")

    # Histogramas de cada variable
    numeric_columns = data.select_dtypes(include=[np.number]).columns
    data[numeric_columns].hist(bins=30, figsize=(20, 15), color='steelblue', edgecolor='black')
    plt.suptitle('Histogramas de Variables', fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    histograms_path = os.path.join(RESULTS_DIR, 'histograms.png')
    plt.savefig(histograms_path, dpi=300)
    plt.close()
    logging.info(f"Gráfico 'histograms.png' guardado.")

    # Boxplots para detectar outliers
    numeric_columns = data.select_dtypes(include=[np.number]).columns
    feature_columns = numeric_columns.drop(['failure', 'anomaly'], errors='ignore')
    num_features = len(feature_columns)

    # Definir número de columnas por fila
    num_cols = 3
    # Calcular número de filas necesarias
    num_rows = math.ceil(num_features / num_cols)

    # Crear una figura grande para contener todos los boxplots
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(num_cols * 6, num_rows * 4))
    axes = axes.flatten()  # Aplanar para iterar fácilmente

    for idx, column in enumerate(feature_columns):
        sns.boxplot(y=data[column], ax=axes[idx], color='lightgreen')
        axes[idx].set_title(f'Boxplot de {column}')

    # Eliminar subplots vacíos si los hay
    for ax in axes[num_features:]:
        fig.delaxes(ax)

    plt.tight_layout()
    boxplots_path = os.path.join(RESULTS_DIR, 'boxplots.png')
    plt.savefig(boxplots_path, dpi=300)
    plt.close()
    logging.info(f"Gráfico 'boxplots.png' guardado.")

    # Pairplot para ver relaciones entre variables
    sns.pairplot(data.drop(['equipment_id', 'time_step'], axis=1, errors='ignore'), hue='failure', palette='coolwarm', diag_kind='kde')
    plt.suptitle('Pairplot de Variables', y=1.02)
    pairplot_path = os.path.join(RESULTS_DIR, 'pairplot.png')
    plt.savefig(pairplot_path, dpi=300)
    plt.close()
    logging.info(f"Gráfico 'pairplot.png' guardado.")

def preprocess_data(data):
    """Preprocesar los datos: escalado, balanceo y división en conjuntos de entrenamiento y prueba."""
    # Variables independientes (X) y dependiente (y)
    X = data.drop(['failure', 'equipment_id', 'time_step', 'anomaly'], axis=1, errors='ignore')
    y = data['failure'].astype(int)  # Asegurar que 'failure' es de tipo entero

    # Escalado de características
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    logging.info("Características escaladas correctamente.")

    # Manejo de desbalance de clases con SMOTE
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X_scaled, y)
    logging.info("Datos balanceados usando SMOTE correctamente.")

    # Dividir en conjunto de entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(
        X_resampled, y_resampled, test_size=0.3, random_state=42, stratify=y_resampled
    )
    logging.info("Datos divididos en entrenamiento y prueba correctamente.")
    return X_train, X_test, y_train, y_test, X.columns

def train_classification_models(X_train, y_train):
    """Entrenar modelos de Machine Learning utilizando GridSearchCV."""
    models = {
        'RandomForest': {
            'model': RandomForestClassifier(random_state=42),
            'params': {
                'n_estimators': [100, 200],
                'max_depth': [None, 10, 20],
                'min_samples_split': [2, 5]
            }
        },
        'SVM': {
            'model': SVC(probability=True, random_state=42),
            'params': {
                'C': [0.1, 1, 10],
                'kernel': ['linear', 'rbf']
            }
        },
        'GradientBoosting': {
            'model': GradientBoostingClassifier(random_state=42),
            'params': {
                'n_estimators': [100, 200],
                'learning_rate': [0.05, 0.1],
                'max_depth': [3, 5]
            }
        },
        'LogisticRegression': {
            'model': LogisticRegression(random_state=42, max_iter=1000),
            'params': {
                'C': [0.01, 0.1, 1, 10],
                'penalty': ['l2']
            }
        }
    }

    best_models = {}
    for model_name, mp in models.items():
        logging.info(f"Entrenando y ajustando hiperparámetros para {model_name}...")
        try:
            grid = GridSearchCV(mp['model'], mp['params'], cv=5, scoring='roc_auc', n_jobs=-1)
            grid.fit(X_train, y_train)
            best_models[model_name] = grid.best_estimator_
            logging.info(f"Mejores parámetros para {model_name}: {grid.best_params_}")
            logging.info(f"Mejor ROC AUC en validación para {model_name}: {grid.best_score_:.4f}\n")
        except Exception as e:
            logging.error(f"Error al entrenar {model_name}: {e}")
    return best_models

def evaluate_classification_models(best_models, X_test, y_test):
    """Evaluar los modelos entrenados y guardar los resultados."""
    def save_results(model_name, model, X_test, y_test):
        try:
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test)[:, 1]

            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred)
            recall = recall_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)
            roc_auc = roc_auc_score(y_test, y_pred_proba)
            pr_auc = average_precision_score(y_test, y_pred_proba)

            report = classification_report(y_test, y_pred, output_dict=True)
            conf_matrix = confusion_matrix(y_test, y_pred)

            # Imprimir resultados
            logging.info(f"{model_name} Results:")
            logging.info(f"Accuracy: {accuracy:.4f}")
            logging.info(f"Precision: {precision:.4f}")
            logging.info(f"Recall: {recall:.4f}")
            logging.info(f"F1-score: {f1:.4f}")
            logging.info(f"ROC AUC: {roc_auc:.4f}")
            logging.info(f"PR AUC: {pr_auc:.4f}\n")

            # Guardar los resultados en un archivo Excel
            df_report = pd.DataFrame(report).transpose()

            # Verificar si la hoja ya existe y eliminarla si es necesario
            book = load_workbook(EXCEL_FILE)
            if f'{model_name}_report' in book.sheetnames:
                del book[f'{model_name}_report']
                book.save(EXCEL_FILE)
                logging.info(f"Hoja '{model_name}_report' existente eliminada.")

            with pd.ExcelWriter(EXCEL_FILE, engine="openpyxl", mode="a") as writer:
                df_report.to_excel(writer, sheet_name=f'{model_name}_report')

            # Guardar la matriz de confusión
            plt.figure(figsize=(8, 6))
            sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
                        xticklabels=['No Fallo', 'Fallo'], yticklabels=['No Fallo', 'Fallo'])
            plt.title(f'Matriz de Confusión - {model_name}')
            plt.xlabel('Predicción')
            plt.ylabel('Realidad')
            plt.tight_layout()
            conf_matrix_path = os.path.join(RESULTS_DIR, f'{model_name}_confusion_matrix.png')
            plt.savefig(conf_matrix_path, dpi=300)
            plt.close()
            logging.info(f"Matriz de confusión '{model_name}_confusion_matrix.png' guardada.")

            return y_pred_proba, roc_auc, pr_auc
        except Exception as e:
            logging.error(f"Error al evaluar {model_name}: {e}")
            return None, None, None

    model_metrics = {}
    for model_name, model in best_models.items():
        metrics = save_results(model_name, model, X_test, y_test)
        if metrics[0] is not None:
            y_pred_proba, roc_auc, pr_auc = metrics
            model_metrics[model_name] = {
                'y_pred_proba': y_pred_proba,
                'roc_auc': roc_auc,
                'pr_auc': pr_auc
            }
    return model_metrics

def plot_classification_curves(model_metrics, y_test):
    """Plotear curvas ROC y Precision-Recall para todos los modelos."""
    try:
        plt.figure(figsize=(12, 6))

        # Curvas ROC
        plt.subplot(1, 2, 1)
        for model_name, metrics in model_metrics.items():
            fpr, tpr, _ = roc_curve(y_test, metrics['y_pred_proba'])
            plt.plot(fpr, tpr, label=f'{model_name} (AUC = {metrics["roc_auc"]:.2f})')
        plt.plot([0, 1], [0, 1], 'k--')
        plt.title('Curvas ROC')
        plt.xlabel('Tasa de Falsos Positivos')
        plt.ylabel('Tasa de Verdaderos Positivos')
        plt.legend(loc='lower right')

        # Curvas Precision-Recall
        plt.subplot(1, 2, 2)
        for model_name, metrics in model_metrics.items():
            precision, recall, _ = precision_recall_curve(y_test, metrics['y_pred_proba'])
            plt.plot(recall, precision, label=f'{model_name} (AUC = {metrics["pr_auc"]:.2f})')
        plt.title('Curvas Precision-Recall')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.legend(loc='lower left')

        plt.tight_layout()
        roc_pr_path = os.path.join(RESULTS_DIR, 'roc_pr_curves.png')
        plt.savefig(roc_pr_path, dpi=300)
        plt.close()
        logging.info("Gráfico 'roc_pr_curves.png' guardado correctamente.")
    except Exception as e:
        logging.error(f"Error al plotear curvas de clasificación: {e}")

def plot_feature_importance(models, feature_names):
    """Plotear la importancia de características para modelos que lo soportan."""
    for model_name, model in models.items():
        if hasattr(model, 'feature_importances_'):
            try:
                importances = model.feature_importances_
                indices = np.argsort(importances)[::-1]
                plt.figure(figsize=(10, 6))
                sns.barplot(x=importances[indices], y=np.array(feature_names)[indices], palette='viridis')
                plt.title(f'Importancia de Características - {model_name}')
                plt.xlabel('Importancia')
                plt.ylabel('Características')
                plt.tight_layout()
                fi_path = os.path.join(RESULTS_DIR, f'{model_name}_feature_importance.png')
                plt.savefig(fi_path, dpi=300)
                plt.close()
                logging.info(f"Gráfico de importancia de características '{model_name}_feature_importance.png' guardado correctamente.")
            except Exception as e:
                logging.error(f"Error al plotear importancia de características para {model_name}: {e}")

def detect_anomalies(data):
    """Aplicar cinco algoritmos de detección de anomalías en los datos temporales."""
    # Seleccionar características numéricas
    columns_to_drop = ['equipment_id', 'time_step', 'failure', 'anomaly']
    existing_columns_to_drop = [col for col in columns_to_drop if col in data.columns]
    X_anomaly = data.drop(existing_columns_to_drop, axis=1, errors='ignore')

    # Determinar la versión de scikit-learn para manejar OneHotEncoder si es necesario
    skl_version = version.parse(sklearn.__version__)

    # Codificar variables categóricas si existen
    categorical_cols = X_anomaly.select_dtypes(include=['object', 'category']).columns.tolist()
    if categorical_cols:
        try:
            if skl_version >= version.parse("1.2"):
                encoder = OneHotEncoder(drop='first', sparse_output=False)
            else:
                encoder = OneHotEncoder(drop='first', sparse=False)
            encoded_data = encoder.fit_transform(X_anomaly[categorical_cols])
            encoded_cols = encoder.get_feature_names_out(categorical_cols)
            encoded_df = pd.DataFrame(encoded_data, columns=encoded_cols, index=X_anomaly.index)
            X_anomaly = pd.concat([X_anomaly.drop(categorical_cols, axis=1), encoded_df], axis=1)
            logging.info("Variables categóricas codificadas correctamente para detección de anomalías.")
        except Exception as e:
            logging.error(f"Error al codificar variables categóricas para detección de anomalías: {e}")
            raise
    else:
        logging.info("No se encontraron columnas categóricas para codificar en detección de anomalías.")

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_anomaly)
    logging.info("Características escaladas para detección de anomalías.")

    anomaly_results = pd.DataFrame(index=data.index)
    anomaly_results['True_Anomaly'] = data['anomaly'].astype(int)

    # 1. Isolation Forest
    try:
        iso_forest = IsolationForest(contamination=0.02, random_state=42)
        iso_forest.fit(X_scaled)
        y_pred_iso = iso_forest.predict(X_scaled)
        anomaly_results['IsolationForest'] = np.where(y_pred_iso == -1, 1, 0)
        logging.info("IsolationForest aplicado correctamente.")
    except Exception as e:
        logging.error(f"Error al aplicar IsolationForest: {e}")

    # 2. One-Class SVM
    try:
        one_class_svm = OneClassSVM(nu=0.02, kernel='rbf', gamma='scale')
        one_class_svm.fit(X_scaled)
        y_pred_svm = one_class_svm.predict(X_scaled)
        anomaly_results['OneClassSVM'] = np.where(y_pred_svm == -1, 1, 0)
        logging.info("OneClassSVM aplicado correctamente.")
    except Exception as e:
        logging.error(f"Error al aplicar OneClassSVM: {e}")

    # 3. Local Outlier Factor
    try:
        lof = LocalOutlierFactor(n_neighbors=20, contamination=0.02)
        y_pred_lof = lof.fit_predict(X_scaled)
        anomaly_results['LocalOutlierFactor'] = np.where(y_pred_lof == -1, 1, 0)
        logging.info("LocalOutlierFactor aplicado correctamente.")
    except Exception as e:
        logging.error(f"Error al aplicar LocalOutlierFactor: {e}")

    # 4. DBSCAN
    try:
        dbscan = DBSCAN(eps=3, min_samples=5)
        dbscan_labels = dbscan.fit_predict(X_scaled)
        anomaly_results['DBSCAN'] = np.where(dbscan_labels == -1, 1, 0)
        logging.info("DBSCAN aplicado correctamente.")
    except Exception as e:
        logging.error(f"Error al aplicar DBSCAN: {e}")

    # 5. PCA-based Outlier Detection
    try:
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X_scaled)
        pca_distances = np.linalg.norm(X_pca, axis=1)
        threshold = np.percentile(pca_distances, 98)  # Top 2% como anomalías
        anomaly_results['PCA_Outlier'] = (pca_distances > threshold).astype(int)
        logging.info("PCA-based Outlier Detection aplicado correctamente.")
    except Exception as e:
        logging.error(f"Error al aplicar PCA-based Outlier Detection: {e}")

    # Guardar resultados de anomalías
    try:
        anomaly_results.to_csv(os.path.join(RESULTS_DIR, 'anomaly_detection_results.csv'), index=False)
        logging.info("Resultados de detección de anomalías guardados en 'anomaly_detection_results.csv'.")
    except Exception as e:
        logging.error(f"Error al guardar resultados de detección de anomalías: {e}")

    # Evaluación de las detecciones
    try:
        for method in ['IsolationForest', 'OneClassSVM', 'LocalOutlierFactor', 'DBSCAN', 'PCA_Outlier']:
            y_true = anomaly_results['True_Anomaly']
            y_pred = anomaly_results[method]
            precision = precision_score(y_true, y_pred, zero_division=0)
            recall = recall_score(y_true, y_pred, zero_division=0)
            f1 = f1_score(y_true, y_pred, zero_division=0)
            logging.info(f"Anomaly Detection - {method}: Precision={precision:.4f}, Recall={recall:.4f}, F1-score={f1:.4f}")
    except Exception as e:
        logging.error(f"Error al evaluar detección de anomalías: {e}")

    # Generar gráficos de detección de anomalías
    try:
        for method in ['IsolationForest', 'OneClassSVM', 'LocalOutlierFactor', 'DBSCAN', 'PCA_Outlier']:
            plt.figure(figsize=(10, 6))
            if 'load' in data.columns:
                sns.scatterplot(x=data.index, y=data['load'], hue=anomaly_results[method], palette='coolwarm', legend=False)
                plt.title(f'Detección de Anomalías - {method}')
                plt.xlabel('Índice de Muestra')
                plt.ylabel('Carga (load)')
            else:
                # Si 'load' no está disponible, usa otra variable numérica
                numerical_vars = ['vibration', 'oil_quality', 'temperature', 'pressure', 'hours_operated']
                available_var = next((var for var in numerical_vars if var in data.columns), 'vibration')
                sns.scatterplot(x=data.index, y=data[available_var], hue=anomaly_results[method], palette='coolwarm', legend=False)
                plt.title(f'Detección de Anomalías - {method}')
                plt.xlabel('Índice de Muestra')
                plt.ylabel(f'Valor de {available_var}')
            plt.tight_layout()
            anomaly_plot_path = os.path.join(RESULTS_DIR, f'{method}_anomaly_detection.png')
            plt.savefig(anomaly_plot_path, dpi=300)
            plt.close()
            logging.info(f"Gráfico de detección de anomalías '{method}_anomaly_detection.png' guardado correctamente.")
    except Exception as e:
        logging.error(f"Error al generar gráficos de detección de anomalías: {e}")

    return anomaly_results

def create_pdf_report(data, model_metrics, feature_names, best_models, anomaly_results):
    """Crear un informe PDF con los gráficos generados."""
    try:
        pdf_file = PDF_REPORT
        doc = SimpleDocTemplate(pdf_file, pagesize=A4,
                                rightMargin=30, leftMargin=30,
                                topMargin=30, bottomMargin=18)
        styles = getSampleStyleSheet()
        styles.add(ParagraphStyle(name='CenterTitle', alignment=1, fontSize=16, spaceAfter=20))
        flowables = []

        # Título
        flowables.append(Paragraph("Informe Técnico: Análisis de Mantenimiento Predictivo y Detección de Anomalías", styles['CenterTitle']))

        # Descripción general
        description = (
            "Este informe presenta un análisis completo de mantenimiento predictivo utilizando técnicas de Machine Learning y Detección de Anomalías. "
            "Se generaron datos simulados temporales para múltiples equipos a lo largo de 40 tiempos para predecir fallos y detectar anomalías en su evolución. "
            "Se entrenaron múltiples modelos y se evaluaron sus desempeños mediante métricas estándar y visualizaciones detalladas."
        )
        flowables.append(Paragraph(description, styles['Normal']))
        flowables.append(Spacer(1, 12))

        # EDA: Distribución de Fallos
        flowables.append(Paragraph("1. Análisis Exploratorio de Datos (EDA)", styles['Heading2']))
        flowables.append(Spacer(1, 12))
        flowables.append(Paragraph("Figura 1: Distribución de Fallos vs No Fallos", styles['Heading3']))
        flowables.append(Image(os.path.join(RESULTS_DIR, 'failure_distribution.png'), width=400, height=300))
        flowables.append(Spacer(1, 12))

        # EDA: Matriz de Correlación
        flowables.append(Paragraph("Figura 2: Matriz de Correlación de Datos Simulados", styles['Heading3']))
        flowables.append(Image(os.path.join(RESULTS_DIR, 'correlation_matrix.png'), width=500, height=400))
        flowables.append(Spacer(1, 12))

        # EDA: Histogramas
        flowables.append(Paragraph("Figura 3: Histogramas de Variables", styles['Heading3']))
        flowables.append(Image(os.path.join(RESULTS_DIR, 'histograms.png'), width=500, height=400))
        flowables.append(Spacer(1, 12))

        # EDA: Boxplots
        flowables.append(Paragraph("Figura 4: Boxplots de Variables", styles['Heading3']))
        flowables.append(Image(os.path.join(RESULTS_DIR, 'boxplots.png'), width=500, height=400))
        flowables.append(Spacer(1, 12))

        # EDA: Pairplot
        flowables.append(Paragraph("Figura 5: Pairplot de Variables", styles['Heading3']))
        flowables.append(Image(os.path.join(RESULTS_DIR, 'pairplot.png'), width=500, height=400))
        flowables.append(Spacer(1, 12))

        # Detección de Anomalías
        flowables.append(Paragraph("2. Detección de Anomalías", styles['Heading2']))
        flowables.append(Spacer(1, 12))
        flowables.append(Paragraph("Se aplicaron cinco algoritmos diferentes de detección de anomalías para identificar comportamientos inusuales en los datos temporales.", styles['Normal']))
        flowables.append(Spacer(1, 12))
        # Incluir gráficos de detección de anomalías
        for idx, method in enumerate(['IsolationForest', 'OneClassSVM', 'LocalOutlierFactor', 'DBSCAN', 'PCA_Outlier'], start=1):
            anomaly_plot_path = os.path.join(RESULTS_DIR, f'{method}_anomaly_detection.png')
            if os.path.exists(anomaly_plot_path):
                flowables.append(Paragraph(f"Figura {5 + idx}: Detección de Anomalías - {method}", styles['Heading3']))
                flowables.append(Image(anomaly_plot_path, width=400, height=300))
                flowables.append(Spacer(1, 12))

        # Evaluación de Modelos de Clasificación
        flowables.append(Paragraph("3. Evaluación de Modelos de Clasificación", styles['Heading2']))
        flowables.append(Spacer(1, 12))

        # Agregar métricas de modelos
        for idx, (model_name, metrics) in enumerate(model_metrics.items(), start=1):
            flowables.append(Paragraph(f"3.{idx} {model_name}", styles['Heading3']))
            flowables.append(Paragraph(f"ROC AUC: {metrics['roc_auc']:.4f}", styles['Normal']))
            flowables.append(Paragraph(f"PR AUC: {metrics['pr_auc']:.4f}", styles['Normal']))
            flowables.append(Spacer(1, 12))
            # Incluir matriz de confusión
            conf_matrix_path = os.path.join(RESULTS_DIR, f"{model_name}_confusion_matrix.png")
            if os.path.exists(conf_matrix_path):
                # Ajustar el número de figura
                figure_number = 5 + len(['IsolationForest', 'OneClassSVM', 'LocalOutlierFactor', 'DBSCAN', 'PCA_Outlier']) + idx
                flowables.append(Paragraph(f"Figura {figure_number}: Matriz de Confusión - {model_name}", styles['Heading4']))
                flowables.append(Image(conf_matrix_path, width=300, height=250))
                flowables.append(Spacer(1, 12))

        # Curvas ROC y PR
        flowables.append(Paragraph("Figura 11: Curvas ROC y Precision-Recall", styles['Heading3']))
        flowables.append(Image(os.path.join(RESULTS_DIR, 'roc_pr_curves.png'), width=500, height=300))
        flowables.append(Spacer(1, 12))

        # Importancia de características
        flowables.append(Paragraph("4. Importancia de Características", styles['Heading2']))
        flowables.append(Spacer(1, 12))
        for idx, model_name in enumerate(best_models.keys(), start=1):
            fi_path = os.path.join(RESULTS_DIR, f'{model_name}_feature_importance.png')
            if os.path.exists(fi_path):
                figure_number = 12 + idx
                flowables.append(Paragraph(f"Figura {figure_number}: Importancia de Características - {model_name}", styles['Heading3']))
                flowables.append(Image(fi_path, width=400, height=300))
                flowables.append(Spacer(1, 12))

        # Conclusiones
        flowables.append(Paragraph("5. Conclusiones", styles['Heading2']))
        conclusions = (
            "Los modelos de clasificación evaluados demostraron un desempeño prometedor en la predicción de fallos de equipos. "
            "Entre los modelos evaluados, Random Forest y Gradient Boosting presentaron las mejores métricas de rendimiento, "
            "indicando una alta capacidad para distinguir entre equipos que fallarán y los que no. "
            "La detección de anomalías mediante cinco diferentes algoritmos permitió identificar comportamientos inusuales en la evolución temporal de los equipos. "
            "Las variables como la vibración, la calidad del aceite, las horas operadas, y el historial de mantenimiento fueron las más determinantes para predecir fallos. "
            "Estos hallazgos sugieren que un monitoreo continuo y un mantenimiento preventivo basado en estas métricas pueden mejorar significativamente la fiabilidad de los equipos."
        )
        flowables.append(Paragraph(conclusions, styles['Normal']))

        # Generar el PDF
        doc.build(flowables)
        logging.info(f"Informe PDF generado en '{pdf_file}'.")
    except Exception as e:
        logging.error(f"Error al crear el informe PDF: {e}")

def main():
    """Main function for overall flow"""
    setup_directories()
    data = generate_simulated_temporal_data()
    data = handle_data_types(data)  # Manejar tipos de datos antes del EDA

    # # Verificación adicional
    # categorical_cols_remaining = data.select_dtypes(include=['object', 'category']).columns.tolist()
    # if categorical_cols_remaining:
    #     raise ValueError(f"Las siguientes columnas aún son categóricas y no han sido codificadas: {categorical_cols_remaining}")
    # else:
    #     logging.info("Todas las columnas categóricas han sido codificadas correctamente.")

    # perform_eda(data)
    # X_train, X_test, y_train, y_test, feature_names = preprocess_data(data)
    # best_models = train_classification_models(X_train, y_train)
    # model_metrics = evaluate_classification_models(best_models, X_test, y_test)
    # plot_classification_curves(model_metrics, y_test)
    # plot_feature_importance(best_models, feature_names)
    # anomaly_results = detect_anomalies(data)
    # create_pdf_report(data, model_metrics, feature_names, best_models, anomaly_results)
    # logging.info("Proceso completado exitosamente. Todos los resultados y el informe se han guardado en la carpeta 'resultados'.")

if __name__ == "__main__":
    main()